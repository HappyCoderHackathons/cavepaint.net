"""Measure stereo camera toe-in by finding the convergence distance.

The convergence point is where both cameras see an object at the same X pixel
(disparity = 0). That distance lets us correct depth calculations for toe-in.

Usage:
    uv run calibrate_toein.py

Instructions:
    Hold a finger/pen in front of the cameras and move it closer/farther
    until the vertical green line appears in the same position in both frames.
    Press 'c' to record that distance (measured with a tape measure).
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp

CALIBRATION_PATH = Path("calibration.json")
HAND_MODEL_PATH = Path("hand_landmarker.task")
INDEX_FINGERTIP = 8

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


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


def get_fingertip(result, w, h):
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0][INDEX_FINGERTIP]
    return int(lm.x * w), int(lm.y * h)


def main():
    cap0 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    for cap in (cap0, cap1):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ts0 = ts1 = 0

    print("Move your finger closer/farther until the X markers align in both frames.")
    print("Press 'c' when aligned, then enter the distance. Press 'q' to quit.")

    with make_landmarker() as lm0, make_landmarker() as lm1:
        while True:
            ret0, frame0 = cap0.read()
            ret1, frame1 = cap1.read()
            if not ret0 or not ret1:
                continue

            frame0 = cv2.flip(frame0, 1)
            frame1 = cv2.flip(frame1, 1)
            h, w = frame0.shape[:2]

            ts0 += 33
            ts1 += 33
            res0 = detect(lm0, frame0, ts0)
            res1 = detect(lm1, frame1, ts1)

            tip0 = get_fingertip(res0, w, h)
            tip1 = get_fingertip(res1, w, h)

            disparity = None
            if tip0:
                cv2.circle(frame0, tip0, 8, (0, 255, 255), -1)
                cv2.line(frame0, (tip0[0], 0), (tip0[0], h), (0, 255, 0), 1)
                cv2.putText(frame0, f"x={tip0[0]}", (tip0[0] + 6, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if tip1:
                cv2.circle(frame1, tip1, 8, (0, 255, 255), -1)
                cv2.line(frame1, (tip1[0], 0), (tip1[0], h), (0, 255, 0), 1)
                cv2.putText(frame1, f"x={tip1[0]}", (tip1[0] + 6, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if tip0 and tip1:
                disparity = tip0[0] - tip1[0]
                color = (0, 255, 0) if abs(disparity) < 5 else (0, 100, 255)
                status = f"disparity: {disparity:+d} px  {'<-- ALIGNED, press c!' if abs(disparity) < 5 else ''}"
                cv2.putText(frame0, status, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            cv2.putText(frame0, "CAM 1 (left)",  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame1, "CAM 2 (right)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Toe-in Calibration", cv2.hconcat([frame0, frame1]))
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                if disparity is None:
                    print("No finger detected — make sure both cameras can see your finger.")
                    continue
                print(f"Current disparity: {disparity:+d} px")
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                try:
                    zc = float(input("Measure the distance from the camera mount to your finger (inches): ").strip())
                except ValueError:
                    print("Invalid input.")
                    break

                angle_deg = __import__('math').degrees(__import__('math').atan((12 / 2) / zc))
                print(f"Convergence distance: {zc:.1f}\"")
                print(f"Estimated toe-in angle per camera: {angle_deg:.1f}°")

                cal = {}
                if CALIBRATION_PATH.exists():
                    cal = json.loads(CALIBRATION_PATH.read_text())
                cal["convergence_inches"] = round(zc, 2)
                cal["toein_angle_deg"] = round(angle_deg, 2)
                CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
                print(f"Saved to {CALIBRATION_PATH}")
                break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
