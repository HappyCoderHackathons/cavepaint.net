import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np

HAND_MODEL_PATH = Path("hand_landmarker.task")
INDEX_FINGERTIP = 8
points = []
current_speed = 0.0
canvas = None
last_radius = 8


HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

def draw_latest_segment(frame_shape):
    global canvas, last_radius
    if canvas is None:
        canvas = np.zeros(frame_shape, dtype=np.uint8)

    if len(points) < 2:
        return

    p1, p2 = np.array(points[-2]), np.array(points[-1])
    direction = p2 - p1
    length = np.linalg.norm(direction)

    steps = max(1, int(length / 10))  # one sub-segment per 10px
    for i in range(steps):
        t1 = i / steps
        t2 = (i + 1) / steps
        sp1 = (p1 + direction * t1).astype(int)
        sp2 = (p1 + direction * t2).astype(int)
        seg = sp2 - sp1
        seg_len = np.linalg.norm(seg)
        if seg_len == 0:
            continue

    # Radius from speed, clamped so it can't change too abruptly
    speed_norm = min(length / 40.0, 1.0)
    target_radius = int(12 - speed_norm * (12 - 4))
    r = int(last_radius * 0.6 + target_radius * 0.4)  # smooth transition
    last_radius = r


    perp = np.array([-direction[1], direction[0]]) / length
    quad = np.array([
        p1 + perp * r, p1 - perp * r,
        p2 - perp * r, p2 + perp * r,
    ], dtype=np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(canvas, [quad], (255, 0, 0))
    cv2.circle(canvas, tuple(p1), r, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(p2), r, (255, 0, 0), -1, cv2.LINE_AA)


def get_fingertip(landmarks, w, h):
    """Return (x, y) pixel coords of the index fingertip."""
    lm = landmarks[INDEX_FINGERTIP]
    return int(lm.x * w), int(lm.y * h)


def draw_hand(frame, landmarks, w, h):
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    points = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for conn in connections:
        cv2.line(frame, points[conn.start], points[conn.end], (0, 200, 0), 2)
    for pt in points.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    print("Hand tracking running. Press 'q' to quit.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    frame_ts_ms = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks, w, h)


                fx, fy = get_fingertip(landmarks, w, h)
                points.append((fx, fy))
                draw_latest_segment(frame.shape)

                # Composite canvas onto live frame
                output = frame.copy()
                mask = canvas.any(axis=2)
                output[mask] = canvas[mask]
                cv2.imshow("Hand Tracking", output)


                cv2.circle(frame, (fx, fy), 10, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"tip: ({fx}, {fy})",
                    (fx + 12, fy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                points.clear()
                cv2.imshow("Hand Tracking", frame)

            #cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
