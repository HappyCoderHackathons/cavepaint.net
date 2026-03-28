"""Measure individual toe-in angle for each camera.

Points a single camera at a known distant reference point straight ahead
(e.g. a mark on the wall directly in front of the camera mount center),
then measures how far off-center that point appears in the frame.

Usage:
    uv run calibrate_angles.py --cam 1
    uv run calibrate_angles.py --cam 2

Instructions:
    1. Place a small mark (tape, pen dot) on the wall directly in front of
       the CENTER of the camera bar — not in front of each camera individually.
    2. Run this script for each camera.
    3. Click the mark in the frame.
    4. Enter the distance from the camera bar to the wall.
    5. The toe-in angle is saved to calibration.json.
"""

import argparse
import json
import math
from pathlib import Path

import cv2

CALIBRATION_PATH = Path("calibration.json")

clicked_pt = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, required=True, help="Camera index (1 or 2)")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def mouse_callback(event, x, y, flags, param):
    global clicked_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pt = (x, y)


def main():
    global clicked_pt

    args = parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        return

    cx = args.width  / 2
    cy = args.height / 2

    cv2.namedWindow("Angle Calibration")
    cv2.setMouseCallback("Angle Calibration", mouse_callback)

    print(f"Camera {args.cam}: click the reference mark on the wall, then press 'c'. Press 'q' to quit.")

    frozen = False
    frozen_frame = None

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frozen_frame = frame.copy()

        display = frozen_frame.copy()
        h, w = display.shape[:2]

        # Draw center crosshair
        cv2.line(display, (int(cx), 0), (int(cx), h), (100, 100, 100), 1)
        cv2.line(display, (0, int(cy)), (w, int(cy)), (100, 100, 100), 1)

        if clicked_pt:
            cv2.circle(display, clicked_pt, 8, (0, 255, 255), -1)
            cv2.line(display, (int(cx), int(cy)), clicked_pt, (0, 255, 255), 1)
            offset_x = clicked_pt[0] - cx
            cv2.putText(display, f"offset: {offset_x:+.0f} px", (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        status = "FROZEN | click the mark | c = confirm | r = re-freeze" if frozen else "Live | SPACE to freeze"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("Angle Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" "):
            frozen = True
            clicked_pt = None

        if key == ord("r"):
            frozen = False
            clicked_pt = None

        if key == ord("c") and frozen and clicked_pt:
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            offset_x = clicked_pt[0] - cx
            print(f"\nPixel offset from center: {offset_x:+.1f} px")

            try:
                distance = float(input("Distance from camera bar to the reference mark (inches): ").strip())
            except ValueError:
                print("Invalid input.")
                return

            # Load calibration
            cal = {}
            if CALIBRATION_PATH.exists():
                cal = json.loads(CALIBRATION_PATH.read_text())

            focal = cal.get(f"cam{args.cam}", {}).get("focal_length_px", 807.0)

            # Angle = atan(pixel_offset / focal_length)
            # Positive = camera is rotated inward (toe-in), negative = outward
            angle_rad = math.atan(offset_x / focal)
            angle_deg = math.degrees(angle_rad)

            print(f"Toe-in angle for cam{args.cam}: {angle_deg:.2f}°  ({'inward' if angle_deg > 0 else 'outward'})")

            cam_key = f"cam{args.cam}"
            cal.setdefault(cam_key, {})["toein_angle_deg"] = round(angle_deg, 3)
            cal.setdefault(cam_key, {})["toein_measured_at"] = {
                "offset_px": round(offset_x, 1),
                "distance_in": distance,
                "focal_px": focal,
            }
            CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
            print(f"Saved to {CALIBRATION_PATH}")
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
