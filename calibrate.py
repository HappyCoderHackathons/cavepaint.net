"""Focal length calibration using a known object at a known distance.

Usage:
    uv run calibrate.py
    uv run calibrate.py --cam 1

Instructions:
    1. Hold a flat object face-on to the camera (credit card, paper, phone, ruler)
    2. Measure the distance from the camera lens to the object (inches)
    3. Draw a box around the object by clicking and dragging
    4. Enter the real width and distance when prompted
    5. Focal length is saved to calibration.json
"""

import argparse
import json
from pathlib import Path

import cv2

CALIBRATION_PATH = Path("calibration.json")

drawing = False
box_start = None
box_end = None
frozen_frame = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=1, help="Camera index to calibrate")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def mouse_callback(event, x, y, flags, param):
    global drawing, box_start, box_end

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        box_start = (x, y)
        box_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        box_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        box_end = (x, y)


def draw_box(frame):
    if box_start and box_end:
        cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)
        px_w = abs(box_end[0] - box_start[0])
        px_h = abs(box_end[1] - box_start[1])
        cv2.putText(frame, f"{px_w} x {px_h} px",
                    (min(box_start[0], box_end[0]), min(box_start[1], box_end[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    global frozen_frame, box_start, box_end

    args = parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        return

    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", mouse_callback)

    print("Step 1: Hold your object flat and face-on to the camera.")
    print("        Press SPACE to freeze the frame, then draw a box around the object.")
    print("        Press 'r' to re-freeze. Press 'q' to quit.")

    frozen = False

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frozen_frame = frame.copy()

        display = frozen_frame.copy()
        draw_box(display)

        status = "FROZEN — draw a box around the object  |  r = re-freeze  |  c = confirm" \
                 if frozen else \
                 "Live — press SPACE to freeze"
        cv2.putText(display, status, (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1, cv2.LINE_AA)

        cv2.imshow("Calibrate", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord(" "):
            frozen = True

        if key == ord("r"):
            frozen = False
            box_start = None

        if key != 255:
            print(f"[key] {key} | frozen={frozen} | box_start={box_start} | box_end={box_end}")

        if key == ord("c"):
            print(f"[c pressed] frozen={frozen} box_start={box_start} box_end={box_end}")
            if frozen and box_start and box_end:
                print("[c] breaking out of loop")
                break
            else:
                print("[c] conditions not met — not breaking")

    print("[loop] exited, releasing camera...")
    cap.release()
    print("[loop] camera released, destroying windows...")
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("[loop] windows destroyed")

    if not box_start or not box_end:
        print("No box drawn — exiting.")
        return

    px_width = abs(box_end[0] - box_start[0])
    if px_width == 0:
        print("Box width is zero — try again.")
        return

    print(f"\nBox drawn: {px_width} pixels wide")
    print()

    try:
        real_width  = float(input("Real width of the object (inches): ").strip())
        distance    = float(input("Distance from camera lens to object (inches): ").strip())
    except ValueError:
        print("Invalid input.")
        return

    focal_length = (px_width * distance) / real_width
    print(f"\nCalculated focal length: {focal_length:.1f} px")

    # Load existing calibration if present, update focal length
    cal = {}
    if CALIBRATION_PATH.exists():
        cal = json.loads(CALIBRATION_PATH.read_text())

    cam_key = f"cam{args.cam}"
    cal.setdefault(cam_key, {})["focal_length_px"] = round(focal_length, 2)
    cal.setdefault(cam_key, {})["measured_at"] = {
        "px_width": px_width,
        "real_width_in": real_width,
        "distance_in": distance,
    }

    # Compute average if both cameras have been calibrated
    focal_lengths = [
        v["focal_length_px"]
        for v in cal.values()
        if isinstance(v, dict) and "focal_length_px" in v
    ]
    if len(focal_lengths) > 1:
        cal["focal_length_avg_px"] = round(sum(focal_lengths) / len(focal_lengths), 2)
        print(f"Average focal length across cameras: {cal['focal_length_avg_px']:.1f} px")

    CALIBRATION_PATH.write_text(json.dumps(cal, indent=2))
    print(f"Saved to {CALIBRATION_PATH}")
    print(f"\nUpdate FOCAL_LENGTH_PX in triangulate.py to {focal_length:.1f}")


if __name__ == "__main__":
    main()
