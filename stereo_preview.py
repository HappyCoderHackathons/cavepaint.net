"""Stereo camera preview with fingertip triangulation.

Runs MediaPipe hand tracking on both Brio 101 streams and displays
the estimated 3D position of the index fingertip.

Usage:
    uv run stereo_preview.py
    uv run stereo_preview.py --cam0 1 --cam1 2
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp

from triangulate import depth_inches_to_str, triangulate

HAND_MODEL_PATH = Path("hand_landmarker.task")
INDEX_FINGERTIP = 8

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=2, help="Left camera index")
    parser.add_argument("--cam1", type=int, default=1, help="Right camera index")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def make_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detect(landmarker, frame, ts_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def draw_hand(frame, landmarks, w, h):
    conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for conn in conns:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    # Highlight fingertip
    fp = pts[INDEX_FINGERTIP]
    cv2.circle(frame, fp, 10, (0, 255, 255), -1)
    return pts[INDEX_FINGERTIP]


def main():
    args = parse_args()

    print(f"Opening cameras {args.cam0} (left) and {args.cam1} (right)...")
    cap0 = open_camera(args.cam0, args.width, args.height)
    cap1 = open_camera(args.cam1, args.width, args.height)
    print("Press 'q' to quit.")

    ts0 = ts1 = 0

    with make_landmarker() as lm0, make_landmarker() as lm1:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()

            if not ret0 or not ret1:
                print("Failed to read from one or both cameras.")
                break

            frame0 = cv2.flip(frame0, 1)
            frame1 = cv2.flip(frame1, 1)
            h, w = frame0.shape[:2]

            ts0 += 33
            ts1 += 33
            res0 = detect(lm0, frame0, ts0)
            res1 = detect(lm1, frame1, ts1)

            tip0 = tip1 = None

            if res0.hand_landmarks:
                tip0 = draw_hand(frame0, res0.hand_landmarks[0], w, h)
            if res1.hand_landmarks:
                tip1 = draw_hand(frame1, res1.hand_landmarks[0], w, h)

            # Triangulate if both cameras see the hand
            pos3d = None
            if tip0 and tip1:
                pos3d = triangulate(tip0, tip1)

            depth_str = depth_inches_to_str(pos3d)
            color = (0, 255, 0) if pos3d else (100, 100, 100)

            # Labels on each frame
            cv2.putText(frame0, f"CAM {args.cam0} (left)",  (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame1, f"CAM {args.cam1} (right)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            combined = cv2.hconcat([frame0, frame1])

            # 3D position overlay centered at the bottom
            cv2.putText(combined, depth_str, (20, combined.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            cv2.imshow("Stereo Preview", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
