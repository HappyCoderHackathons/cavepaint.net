"""Single-camera gesture drawing demo using StrokeStore.

Controls:
  - q / ESC : quit
  - c       : clear canvas
  - u       : undo last stroke
  - [ / ]   : decrease/increase brush radius
  - p       : cycle palette color

Gestures:
  - point : draw
  - fist  : erase
  - peace : resize brush (pinch index-middle)
  - open_hand + swipe left/right : cycle color
"""

import argparse
import math
import time

import cv2
import numpy as np

from swipe_detect import SwipeDetector
from stroke import StrokeStore
from stereo_drawing.constants import (
    GESTURE_CONFIDENCE,
    PALETTE,
    SWIPE_DISPLAY_FRAMES,
    SWIPE_META_PATH,
    SWIPE_MODEL_PATH,
    _PALM_INDICES,
)
from stereo_drawing.gesture import GestureClassifier
from stereo_drawing.landmarker import detect, draw_hand, make_landmarker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera gesture drawing")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    return parser.parse_args()


def overlay_canvas(frame, canvas):
    mask = canvas.any(axis=2)
    frame[mask] = canvas[mask]
    return frame


def draw_swipe_events(frame, events, palette):
    if not events:
        return
    h, w = frame.shape[:2]
    y = h // 2 - 40
    for label, color_idx, frames_left in events:
        text = label.upper().replace("_", " ")
        color = palette[color_idx]
        alpha = min(frames_left / 15.0, 1.0)
        tcolor = tuple(int(c * alpha) for c in color)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        x = (w - size[0]) // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, tcolor, 2, cv2.LINE_AA)
        y += size[1] + 10


def draw_palette(frame, palette, active_idx):
    h, w = frame.shape[:2]
    swatch = 30
    gap = 6
    total = len(palette) * (swatch + gap) - gap
    x0 = (w - total) // 2
    y0 = h - swatch - 10
    cv2.rectangle(frame, (x0 - 8, y0 - 8), (x0 + total + 8, y0 + swatch + 8), (28, 28, 28), -1)
    for i, color in enumerate(palette):
        x = x0 + i * (swatch + gap)
        cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), color, -1)
        if i == active_idx:
            cv2.rectangle(frame, (x - 2, y0 - 2), (x + swatch + 2, y0 + swatch + 2), (255, 255, 255), 2)
            cx = x + swatch // 2
            pts = np.array([[cx, y0 - 5], [cx - 5, y0 - 13], [cx + 5, y0 - 13]], dtype=np.int32)
            cv2.fillPoly(frame, [pts], (255, 255, 255))
        else:
            cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), (70, 70, 70), 1)


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.cam}")

    try:
        gesture_clf = GestureClassifier()
        print(f"[gesture] loaded classes: {gesture_clf.gestures}")
    except Exception as exc:
        print(f"[gesture] classifier unavailable ({exc}); gesture actions disabled")
        gesture_clf = None

    try:
        swipe_det = SwipeDetector(SWIPE_MODEL_PATH, SWIPE_META_PATH)
        print("[swipe] detector loaded")
    except Exception as exc:
        print(f"[swipe] detector unavailable ({exc}); swipe color cycling disabled")
        swipe_det = None

    strokes = StrokeStore()
    color_idx = 0
    was_drawing = False
    was_erasing = False
    smooth_pinch = 0.5
    frame_ts_ms = 0
    fps = 0.0
    fps_count = 0
    fps_t0 = time.monotonic()
    swipe_events = []

    with make_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            frame_ts_ms += 33
            result = detect(landmarker, frame, frame_ts_ms)

            tip = None
            tip8 = None
            tip12 = None
            gesture = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                tip = draw_hand(frame, hand, w, h)
                tip8 = hand[8]
                tip12 = hand[12]
                if gesture_clf:
                    g, conf = gesture_clf.classify(hand)
                    if conf >= GESTURE_CONFIDENCE:
                        gesture = g

                if swipe_det and gesture == "open_hand":
                    xs = [hand[i].x for i in _PALM_INDICES]
                    ys = [hand[i].y for i in _PALM_INDICES]
                    palm_x = sum(xs) / len(xs)
                    palm_y = sum(ys) / len(ys)
                    palm_sc = float(math.hypot(hand[12].x - hand[0].x, hand[12].y - hand[0].y))
                    swipe = swipe_det.update(palm_x, palm_y, palm_sc)
                    if swipe:
                        label, _ = swipe
                        if label == "swipe_right":
                            color_idx = (color_idx + 1) % len(PALETTE)
                            swipe_events.append((label, color_idx, SWIPE_DISPLAY_FRAMES))
                        elif label == "swipe_left":
                            color_idx = (color_idx - 1) % len(PALETTE)
                            swipe_events.append((label, color_idx, SWIPE_DISPLAY_FRAMES))

            drawing = gesture == "point"
            erasing = gesture == "fist"
            resizing = gesture == "peace"

            if erasing and tip:
                if was_drawing:
                    strokes.end()
                strokes.erase_near(tip[0], tip[1], radius=strokes.current_radius, z=0.0)
                was_erasing = True
                was_drawing = False
            elif drawing and tip:
                if not was_drawing:
                    active_color = PALETTE[color_idx]
                    active_radius = int(max(1, round(float(strokes.current_radius))))
                    active_min_radius = strokes.stroke_min_radius(active_radius)
                    strokes.begin(
                        color=active_color,
                        max_radius=active_radius,
                        min_radius=active_min_radius,
                    )
                strokes.add_point(tip[0], tip[1], 0.0)
                was_drawing = True
                was_erasing = False
            elif resizing and tip and tip8 and tip12:
                if was_drawing:
                    strokes.end()
                pinch_max = 0.25
                raw_dist = math.hypot(tip12.x - tip8.x, tip12.y - tip8.y)
                pinch_norm = 1.0 - min(raw_dist / pinch_max, 1.0)
                smooth_pinch = 0.85 * smooth_pinch + 0.15 * pinch_norm
                next_radius = strokes._radius(smooth_pinch)
                strokes.current_radius = int(max(strokes.min_radius, min(strokes.max_radius, next_radius)))
                was_drawing = False
                was_erasing = False
            else:
                if was_drawing:
                    strokes.end()
                was_drawing = False
                was_erasing = False

            canvas = strokes.render(frame.shape)
            frame = overlay_canvas(frame, canvas)

            if erasing and tip:
                cv2.circle(frame, tip, int(strokes.current_radius), (0, 0, 255), 2, cv2.LINE_AA)
            if resizing and tip8 and tip12:
                px8 = (int(tip8.x * w), int(tip8.y * h))
                px12 = (int(tip12.x * w), int(tip12.y * h))
                mid = ((px8[0] + px12[0]) // 2, (px8[1] + px12[1]) // 2)
                cv2.line(frame, px8, px12, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(frame, mid, int(strokes.current_radius), PALETTE[color_idx], -1, cv2.LINE_AA)
                cv2.circle(frame, mid, int(strokes.current_radius), (255, 255, 255), 1, cv2.LINE_AA)

            draw_swipe_events(frame, swipe_events, PALETTE)
            draw_palette(frame, PALETTE, color_idx)

            hud = [
                f"gesture: {gesture or '-'}",
                f"brush: {int(strokes.current_radius)}",
                f"color: {color_idx + 1}/{len(PALETTE)}",
                f"fps: {fps:.1f}",
                "point=draw, fist=erase, peace=resize",
                "open_hand+swipe = color cycle",
            ]
            y = 28
            for line in hud:
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y += 26

            cv2.imshow("track_draw (single cam)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                strokes.clear()
                was_drawing = False
                was_erasing = False
            elif key == ord("u"):
                strokes.undo()
            elif key == ord("["):
                strokes.current_radius = max(strokes.min_radius, int(strokes.current_radius) - 1)
            elif key == ord("]"):
                strokes.current_radius = min(strokes.max_radius, int(strokes.current_radius) + 1)
            elif key == ord("p"):
                color_idx = (color_idx + 1) % len(PALETTE)

            swipe_events = [(lbl, ci, f - 1) for lbl, ci, f in swipe_events if f > 1]

            fps_count += 1
            now = time.monotonic()
            if now - fps_t0 >= 0.5:
                fps = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
