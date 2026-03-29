"""Collect hand landmark data for gesture training.

Usage:
    uv run collect.py
    uv run collect.py -g open_hand point   # specific gestures only
    uv run collect.py -c 3                 # 3 cycles instead of default 5
"""

import argparse
import csv
import math
import os
import random
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

GESTURES = ["open_hand", "point", "fist", "peace"]
CYCLES = 5
COUNTDOWN_SECONDS = 3
RECORD_SECONDS = 8
PAUSE_SECONDS = 2

HAND_MODEL_PATH = Path("hand_landmarker.task")

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


class Phase(Enum):
    COUNTDOWN = auto()
    RECORD = auto()
    PAUSE = auto()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gestures", nargs="+", choices=GESTURES, default=GESTURES)
    parser.add_argument("-c", "--cycles", type=int, default=CYCLES)
    parser.add_argument("-r", "--record-seconds", type=float, default=RECORD_SECONDS)
    parser.add_argument("-p", "--pause-seconds", type=float, default=PAUSE_SECONDS)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def normalize_landmarks(raw):
    wrist = np.array(raw[0])
    translated = [np.array(p) - wrist for p in raw]
    max_dist = max(np.linalg.norm(t) for t in translated[1:])
    if max_dist < 1e-6:
        return translated
    return [t / max_dist for t in translated]


def draw_text_center(frame, text, y, scale, color, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (frame.shape[1] - sz[0]) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    gestures = args.gestures
    cycles = args.cycles

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path("dataset") / f"session_{timestamp}"
    session_dir.mkdir(parents=True)
    csv_path = session_dir / "landmarks.csv"

    # CSV header: cycle, gesture, frame, timestamp + 21 raw + 21 norm landmarks
    header = ["cycle", "gesture", "frame", "timestamp"]
    for i in range(21):
        header += [f"raw_{i}_x", f"raw_{i}_y", f"raw_{i}_z"]
    for i in range(21):
        header += [f"norm_{i}_x", f"norm_{i}_y", f"norm_{i}_z"]

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

    frame_ts_ms = 0

    with open(csv_path, "w", newline="") as f, \
         HandLandmarker.create_from_options(options) as landmarker:

        writer = csv.writer(f)
        writer.writerow(header)

        for cycle in range(1, cycles + 1):
            order = gestures[:]
            if not args.no_shuffle:
                random.shuffle(order)

            for gesture in order:
                phases = [
                    (Phase.COUNTDOWN, COUNTDOWN_SECONDS),
                    (Phase.RECORD,    args.record_seconds),
                    (Phase.PAUSE,     args.pause_seconds),
                ]

                for phase, duration in phases:
                    phase_start = time.perf_counter()
                    frame_index = 0
                    frames_saved = 0

                    while True:
                        elapsed = time.perf_counter() - phase_start
                        remaining = duration - elapsed
                        if remaining <= 0:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            continue

                        frame = cv2.flip(frame, 1)
                        h, w = frame.shape[:2]

                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        frame_ts_ms += 33
                        result = landmarker.detect_for_video(mp_image, frame_ts_ms)

                        # Draw landmarks
                        if result.hand_landmarks:
                            lms = result.hand_landmarks[0]
                            conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                            pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lms)}
                            for conn in conns:
                                cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
                            for pt in pts.values():
                                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                        # Record
                        if phase == Phase.RECORD and result.hand_landmarks:
                            lms = result.hand_landmarks[0]
                            if len(lms) == 21:
                                raw = [(lm.x, lm.y, lm.z) for lm in lms]
                                norm = normalize_landmarks(raw)
                                row = [cycle, gesture, frame_index, f"{time.time():.6f}"]
                                for x, y, z in raw:
                                    row += [x, y, z]
                                for v in norm:
                                    row += [v[0], v[1], v[2]]
                                writer.writerow(row)
                                frames_saved += 1

                        # UI
                        if phase == Phase.RECORD:
                            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

                        draw_text_center(frame, f"Show: {gesture}", h // 2 - 30, 1.2, (255, 255, 255), 3)

                        if phase == Phase.COUNTDOWN:
                            draw_text_center(frame, str(math.ceil(remaining)), h // 2 + 40, 2.0, (0, 255, 255), 4)
                        elif phase == Phase.RECORD:
                            draw_text_center(frame, f"RECORDING  {remaining:.1f}s", h // 2 + 40, 1.0, (0, 0, 255), 2)
                            sz = cv2.getTextSize(f"Saved: {frames_saved}", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.putText(frame, f"Saved: {frames_saved}", (w - sz[0] - 10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            draw_text_center(frame, f"Pause  {remaining:.1f}s", h // 2 + 40, 1.0, (180, 180, 180), 2)

                        cv2.putText(frame, f"Cycle {cycle}/{cycles}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        cv2.imshow("Collector", frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            print(f"Saved to {session_dir}")
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                        frame_index += 1

    print(f"Done! Dataset saved to {session_dir}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
