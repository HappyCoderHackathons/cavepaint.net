import os
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

INDEX_FINGERTIP = 8
DEFAULT_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


class HandTracker:
    def __init__(self, camera_index=0, model_path=None):
        self.camera_index = camera_index
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.cap = cv2.VideoCapture(camera_index)

        self.lock = threading.Lock()
        self.output_frame = None
        self.running = False
        self.thread = None

        self.points = []
        self.canvas = None
        self.last_radius = 8
        self.frame_ts_ms = 0
        self.error_message = None

        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.BaseOptions = mp.tasks.BaseOptions
        self.RunningMode = mp.tasks.vision.RunningMode

    def _draw_latest_segment(self, frame_shape):
        if self.canvas is None or self.canvas.shape != frame_shape:
            self.canvas = np.zeros(frame_shape, dtype=np.uint8)

        if len(self.points) < 2:
            return

        p1 = np.array(self.points[-2], dtype=np.float32)
        p2 = np.array(self.points[-1], dtype=np.float32)
        direction = p2 - p1
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            return

        speed_norm = min(length / 40.0, 1.0)
        target_radius = int(12 - speed_norm * (12 - 4))
        radius = int(self.last_radius * 0.6 + target_radius * 0.4)
        self.last_radius = radius

        perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
        quad = np.array(
            [p1 + perp * radius, p1 - perp * radius, p2 - perp * radius, p2 + perp * radius],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        p1i = tuple(p1.astype(int))
        p2i = tuple(p2.astype(int))
        cv2.fillPoly(self.canvas, [quad], (255, 0, 0))
        cv2.circle(self.canvas, p1i, radius, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.canvas, p2i, radius, (255, 0, 0), -1, cv2.LINE_AA)

    @staticmethod
    def _get_fingertip(landmarks, width, height):
        lm = landmarks[INDEX_FINGERTIP]
        return int(lm.x * width), int(lm.y * height)

    @staticmethod
    def _draw_hand(frame, landmarks, width, height):
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        hand_points = {i: (int(lm.x * width), int(lm.y * height)) for i, lm in enumerate(landmarks)}
        for conn in connections:
            cv2.line(frame, hand_points[conn.start], hand_points[conn.end], (0, 200, 0), 2)
        for pt in hand_points.values():
            cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    @staticmethod
    def _build_message_frame(message, width=960, height=540):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (20, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def process_loop(self):
        if not self.cap.isOpened():
            self.error_message = f"Error: cannot open camera index {self.camera_index}"
            with self.lock:
                self.output_frame = self._build_message_frame(self.error_message)
            return

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=self.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        try:
            with self.HandLandmarker.create_from_options(options) as landmarker:
                while self.running:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    self.frame_ts_ms += 33
                    result = landmarker.detect_for_video(mp_image, self.frame_ts_ms)

                    output = frame.copy()
                    if result.hand_landmarks:
                        landmarks = result.hand_landmarks[0]
                        self._draw_hand(output, landmarks, width, height)

                        fx, fy = self._get_fingertip(landmarks, width, height)
                        self.points.append((fx, fy))
                        self._draw_latest_segment(output.shape)

                        if self.canvas is not None:
                            mask = self.canvas.any(axis=2)
                            output[mask] = self.canvas[mask]

                        cv2.circle(output, (fx, fy), 10, (0, 255, 255), -1)
                        cv2.putText(
                            output,
                            f"tip: ({fx}, {fy})",
                            (fx + 12, fy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        self.points.clear()

                    with self.lock:
                        self.output_frame = output
        except Exception as exc:
            self.error_message = f"Hand tracker error: {exc}"
            with self.lock:
                self.output_frame = self._build_message_frame(self.error_message)

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
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    tracker = HandTracker(camera_index=camera_index)
    print(f"Hand tracking class test running. Camera index={camera_index}. Press 'q' to quit.")
    tracker.start()

    try:
        while True:
            frame = tracker.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
