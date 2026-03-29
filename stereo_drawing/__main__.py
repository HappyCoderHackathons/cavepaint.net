"""Entry point for: python -m stereo_drawing"""

import argparse
import time

import cv2

from .tracker import StereoDrawingTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=2, help="Left camera index")
    parser.add_argument("--cam1", type=int, default=1, help="Right camera index")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    tracker = StereoDrawingTracker(
        cam0=args.cam0, cam1=args.cam1, width=args.width, height=args.height
    )
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
