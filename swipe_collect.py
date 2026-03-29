"""Collect swipe motion training data.

Usage:
    uv run swipe_collect.py
    uv run swipe_collect.py -g swipe_left swipe_right   # specific motions
    uv run swipe_collect.py -c 2                        # 2 cycles
"""

import argparse
import csv
import math
import random
import time
from collections import deque
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MOTIONS = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "none"]

INSTRUCTIONS = {
    "swipe_left":  "Swipe hand LEFT",
    "swipe_right": "Swipe hand RIGHT",
    "swipe_up":    "Swipe hand UP",
    "swipe_down":  "Swipe hand DOWN",
    "none":        "Hold hand STILL or drift slowly",
}

WINDOW_SIZE    = 30
STEP_SIZE      = 5
REPS           = 12
CYCLES         = 3
COUNTDOWN_SECS = 2.0
RECORD_SECS    = 2.0
RESET_SECS     = 1.5
NONE_SECS      = 6.0

HAND_MODEL_PATH = Path("hand_landmarker.task")
PALM_INDICES = [0, 5, 9, 13, 17]

HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions           = mp.tasks.BaseOptions
RunningMode           = mp.tasks.vision.RunningMode


class Phase(Enum):
    COUNTDOWN = auto()
    RECORD    = auto()
    RESET     = auto()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-g", "--motions", nargs="+", choices=MOTIONS, default=MOTIONS)
    p.add_argument("-c", "--cycles", type=int, default=CYCLES)
    p.add_argument("-r", "--record-seconds", type=float, default=RECORD_SECS)
    p.add_argument("--reset-seconds", type=float, default=RESET_SECS)
    p.add_argument("--none-seconds", type=float, default=NONE_SECS)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--step-size", type=int, default=STEP_SIZE)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("-o", "--output-dir", default="swipe_dataset")
    return p.parse_args()


def extract_features(landmarks):
    """Palm center (avg of wrist + 4 MCP bases) and hand scale (wrist→middle tip)."""
    xs = [landmarks[i].x for i in PALM_INDICES]
    ys = [landmarks[i].y for i in PALM_INDICES]
    px = sum(xs) / len(xs)
    py = sum(ys) / len(ys)
    scale = math.hypot(landmarks[12].x - landmarks[0].x,
                       landmarks[12].y - landmarks[0].y)
    return px, py, scale


def csv_header(window_size):
    h = ["cycle", "motion", "seq_index", "timestamp"]
    for i in range(window_size):
        h += [f"f{i}_px", f"f{i}_py", f"f{i}_scale"]
    return h


def flush_window(writer, buf, cycle, motion, seq_idx):
    row = [cycle, motion, seq_idx, f"{time.time():.6f}"]
    for px, py, sc in buf:
        row += [f"{px:.6f}", f"{py:.6f}", f"{sc:.6f}"]
    writer.writerow(row)


def draw_center(frame, text, y, scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (frame.shape[1] - sz[0]) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def run_phase(cap, landmarker, frame_ts, duration, record,
              writer, cycle, motion, seq_counter,
              buf, step_size, target_fps, display_cb):
    """Run one timed phase. Returns (frame_ts, seq_counter, buf, quit)."""
    frame_delay = 1.0 / target_fps
    window_size = buf.maxlen
    phase_start = time.perf_counter()
    frame_count = 0

    while True:
        loop_start = time.perf_counter()
        remaining = duration - (loop_start - phase_start)
        if remaining <= 0:
            break

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        result = landmarker.detect_for_video(mp_image, frame_ts)

        # Draw skeleton
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            h, w = frame.shape[:2]
            pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lms)}
            for conn in conns:
                cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
            for pt in pts.values():
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        if record and result.hand_landmarks and len(result.hand_landmarks) == 1:
            lms = result.hand_landmarks[0]
            if len(lms) == 21:
                px, py, sc = extract_features(lms)
                buf.append((px, py, sc))
                frame_count += 1
                if frame_count % step_size == 0 and len(buf) == window_size:
                    flush_window(writer, buf, cycle, motion, seq_counter)
                    seq_counter += 1

        display_cb(frame, remaining, seq_counter)
        cv2.imshow("Swipe Collector", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return frame_ts, seq_counter, buf, True

        sleep = frame_delay - (time.perf_counter() - loop_start)
        if sleep > 0:
            time.sleep(sleep)

    return frame_ts, seq_counter, buf, False


def main():
    args = parse_args()
    motions  = args.motions
    cycles   = args.cycles
    window_size = args.window_size
    step_size   = args.step_size
    shuffle = not args.no_shuffle
    target_fps = 30

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / f"session_{timestamp}"
    session_dir.mkdir(parents=True)
    csv_path = session_dir / "sequences.csv"

    print(f"Motions: {', '.join(motions)}")
    print(f"Cycles: {cycles}  Reps: {args.reps}  Record: {args.record_seconds}s")
    print(f"Output: {csv_path}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    frame_ts = 0
    total_seqs = 0
    total_slots = len(motions) * cycles

    with open(csv_path, "w", newline="") as f, \
         HandLandmarker.create_from_options(options) as landmarker:

        writer = csv.writer(f)
        writer.writerow(csv_header(window_size))

        slot = 0
        for cycle in range(1, cycles + 1):
            order = motions[:]
            if shuffle:
                random.shuffle(order)

            for motion in order:
                slot += 1
                instr    = INSTRUCTIONS[motion]
                is_none  = motion == "none"
                reps     = 1 if is_none else args.reps
                rec_dur  = args.none_seconds if is_none else args.record_seconds

                # Countdown
                def countdown_cb(frame, remaining, _seq, instr=instr, cycle=cycle,
                                  slot=slot, total_slots=total_slots, cycles=cycles):
                    h = frame.shape[0]
                    draw_center(frame, instr, h // 2 - 50, 0.9, (255, 255, 255))
                    draw_center(frame, "Get ready...", h // 2, 0.7, (180, 180, 180))
                    draw_center(frame, str(math.ceil(remaining)), h // 2 + 60, 2.0, (0, 255, 255), 4)
                    cv2.putText(frame, f"Cycle {cycle}/{cycles}  {slot}/{total_slots}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                buf = deque(maxlen=window_size)
                frame_ts, _, buf, quit = run_phase(
                    cap, landmarker, frame_ts, args.record_seconds * 0 + 2.0,
                    False, writer, cycle, motion, total_seqs, buf,
                    step_size, target_fps, countdown_cb,
                )
                if quit:
                    print(f"Saved {total_seqs} sequences to {csv_path}")
                    return

                # Reps
                for rep in range(1, reps + 1):
                    def record_cb(frame, remaining, seq_count, instr=instr,
                                  rep=rep, reps=reps, is_none=is_none):
                        h, w = frame.shape[:2]
                        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
                        msg = instr if is_none else f"DO IT: {instr}"
                        draw_center(frame, msg, h // 2 - 30, 0.8, (0, 200, 255))
                        draw_center(frame, f"Rep {rep}/{reps}  {remaining:.1f}s",
                                    h // 2 + 20, 0.7, (0, 0, 255))
                        sz = cv2.getTextSize(f"Seqs: {seq_count}",
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.putText(frame, f"Seqs: {seq_count}",
                                    (w - sz[0] - 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

                    buf = deque(maxlen=window_size)
                    frame_ts, total_seqs, buf, quit = run_phase(
                        cap, landmarker, frame_ts, rec_dur, True,
                        writer, cycle, motion, total_seqs, buf,
                        step_size, target_fps, record_cb,
                    )
                    if quit:
                        print(f"Saved {total_seqs} sequences to {csv_path}")
                        return

                    if rep < reps:
                        def reset_cb(frame, remaining, _seq):
                            h = frame.shape[0]
                            draw_center(frame, "Reset hand to center", h // 2 - 20, 0.8, (180, 180, 180))
                            draw_center(frame, f"{remaining:.1f}s", h // 2 + 30, 0.7, (140, 140, 140))

                        frame_ts, _, _, quit = run_phase(
                            cap, landmarker, frame_ts, args.reset_seconds,
                            False, writer, cycle, motion, total_seqs,
                            deque(maxlen=window_size), step_size, target_fps, reset_cb,
                        )
                        if quit:
                            print(f"Saved {total_seqs} sequences to {csv_path}")
                            return

    print(f"Done! {total_seqs} sequences saved to {csv_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
