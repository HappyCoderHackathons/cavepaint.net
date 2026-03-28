import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=1, help="Left camera index")
    parser.add_argument("--cam1", type=int, default=2, help="Right camera index")
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


def main():
    args = parse_args()

    print(f"Opening cameras {args.cam0} (left) and {args.cam1} (right)...")
    cap0 = open_camera(args.cam0, args.width, args.height)
    cap1 = open_camera(args.cam1, args.width, args.height)
    print("Press 'q' to quit.")

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("Failed to read from one or both cameras.")
            break

        cv2.putText(frame0, f"CAM {args.cam0} (left)",  (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame1, f"CAM {args.cam1} (right)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        combined = cv2.hconcat([frame0, frame1])
        cv2.imshow("Stereo Preview", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
