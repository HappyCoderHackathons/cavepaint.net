"""Entry point for: python -m stereo_drawing"""

import argparse
import os
import time

import cv2

from .tracker import StereoDrawingTracker


def _parse_cam_arg(raw: str | int) -> int | str:
    """Accept numeric camera indices plus URL/zmq camera sources."""
    if isinstance(raw, int):
        return raw
    value = str(raw).strip()
    if value.startswith("http") or value.startswith("zmq://"):
        return value
    return int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cam0",
        default=os.getenv("CAM0", "2"),
        help="Left camera index or URL (defaults to CAM0 env var)",
    )
    parser.add_argument(
        "--cam1",
        default=os.getenv("CAM1", "1"),
        help="Right camera index or URL (defaults to CAM1 env var)",
    )
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    cam0 = _parse_cam_arg(args.cam0)
    cam1 = _parse_cam_arg(args.cam1)

    tracker = StereoDrawingTracker(
        cam0=cam0, cam1=cam1, width=args.width, height=args.height
    )
    print(f"[cli] cam0={cam0} cam1={cam1}")
    tracker.start()

    try:
        while True:
            frame = tracker.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imshow("Stereo Drawing", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                tracker.clear_canvas()
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
