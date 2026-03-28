import cv2
import mediapipe as mp
from pathlib import Path
import numpy as np

HAND_MODEL_PATH = Path("hand_landmarker.task")
INDEX_FINGERTIP = 8
points = []
current_speed = 0.0


HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

def draw_point(frame, landmarks, w, h): 
    lm = landmarks[INDEX_FINGERTIP]

    if len(points) < 2:
        """Early return if not enough points"""
        return
    
    radii = infer_pressure_from_speed(points, min_radius=4, max_radius=12)
    for i in range(1, len(points)):
        p1, p2 = np.array(points[i-1]), np.array(points[i])
        r1, r2 = radii[i-1], radii[i]

        #cv2.circle(frame, tuple(p1), r1, (0, 0, 0), -1, cv2.LINE_AA)
        #cv2.circle(frame, tuple(p2), r2, (0, 0, 0), -1, cv2.LINE_AA)

        direction = p2 - p1
        length = np.linalg.norm(direction)

        perp = np.array([-direction[1], direction[0]]) / length

        quad = np.array([p1 + perp * r1, p1-perp * r1, p2+perp * r2, p2-perp * r2], dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(frame, [quad], (255,0,0))


def infer_pressure_from_speed(points, min_radius=2, max_radius=12):
    """
    Fast movement  →  low pressure  →  thin stroke
    Slow movement  →  high pressure →  thick stroke
    """
    if len(points) < 2:
        return [max_radius] * len(points)
 
    pts = np.array(points, dtype=float)
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)   # segment lengths
    diffs = np.append(diffs, diffs[-1])                     # match length
 
    # Smooth the speed signal to avoid jitter
    kernel_size = max(3, len(diffs) // 10) | 1              # must be odd
    speed_smooth = np.convolve(diffs, np.ones(kernel_size) / kernel_size, mode='same')
 
    # Invert: high speed → low radius
    speed_norm = (speed_smooth - speed_smooth.min()) / (speed_smooth.max() - speed_smooth.min() + 1e-9)
    radii = max_radius - speed_norm * (max_radius - min_radius)
    print(radii)
    return radii.astype(int).tolist()


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
    cap = cv2.VideoCapture(0)
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
                draw_point(frame, landmarks, w, h)

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
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
