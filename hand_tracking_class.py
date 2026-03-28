import cv2
import mediapipe as mp
import threading
import time
import os
from pathlib import Path

HAND_MODEL_PATH = Path("hand_landmarker.task")
INDEX_FINGERTIP = 8


class HandTracker:
    def __init__(self, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index)
        self.lock = threading.Lock()
        self.output_frame = None
        self.running = False
        self.thread = None
        self.trail_points = []
        self.frame_ts_ms = 0

        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.BaseOptions = mp.tasks.BaseOptions
        self.RunningMode = mp.tasks.vision.RunningMode

    def draw_point(self, frame, landmarks, w, h):
        lm = landmarks[INDEX_FINGERTIP]
        self.trail_points.append((int(lm.x * w), int(lm.y * h)))
        for center in self.trail_points:
            cv2.circle(frame, center, 5, (255, 255, 0), 5)

    def draw_hand(self, frame, landmarks, w, h):
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        landmark_points = {
            i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)
        }

        for conn in connections:
            cv2.line(frame, landmark_points[conn.start], landmark_points[conn.end], (0, 200, 0), 2)

        for pt in landmark_points.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    def process_loop(self):
        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=self.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        with self.HandLandmarker.create_from_options(options) as landmarker:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                self.frame_ts_ms += 33
                result = landmarker.detect_for_video(mp_image, self.frame_ts_ms)

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    self.draw_hand(frame, landmarks, w, h)
                    self.draw_point(frame, landmarks, w, h)
                else:
                    self.trail_points.clear()

                with self.lock:
                    self.output_frame = frame.copy()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.thread.start()

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.cap.release()


def main():
    camera_index = int(os.getenv("CAMERA_INDEX", "1"))
    tracker = HandTracker(camera_index=camera_index)
    print(f"HandTracker class test running on camera {camera_index}. Press 'q' to quit.")

    tracker.start()
    no_frame_count = 0

    try:
        while True:
            frame = tracker.get_frame()
            if frame is None:
                no_frame_count += 1
                if no_frame_count == 300:
                    print("No frames received yet. Check camera index/model path if this continues.")
                if no_frame_count > 900:
                    print("Still no frames. Exiting test.")
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.01)
                continue

            no_frame_count = 0
            cv2.imshow("HandTracker Class Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
