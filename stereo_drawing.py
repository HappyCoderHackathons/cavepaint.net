"""Stereo camera drawing with fingertip tracking.

Provides StereoDrawingTracker — a threaded class that captures both cameras,
runs hand detection, draws on a shared canvas, and exposes get_frame() for
MJPEG streaming.

Also runnable standalone:
    uv run stereo_drawing.py
    uv run stereo_drawing.py --cam0 2 --cam1 1
    Press 'q' to quit, 'c' to clear canvas.
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from triangulate import depth_inches_to_str, triangulate

HAND_MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
INDEX_FINGERTIP = 8

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


class _CameraReader(threading.Thread):
    """Continuously reads from a camera on its own thread so reads never block the main loop."""

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False


def _open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {index}")
    return cap


def find_cameras(max_index=8) -> list[int]:
    """Return indices of all cameras that open successfully."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def _make_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def _detect(landmarker, frame, ts_ms):
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def _draw_hand(frame, landmarks, w, h):
    conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for conn in conns:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    fp = pts[INDEX_FINGERTIP]
    cv2.circle(frame, fp, 10, (0, 255, 255), -1)
    return pts[INDEX_FINGERTIP]


def _draw_segment(canvas, points, last_radius):
    if len(points) < 2:
        return last_radius

    p1 = np.array(points[-2], dtype=np.float32)
    p2 = np.array(points[-1], dtype=np.float32)
    direction = p2 - p1
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return last_radius

    speed_norm = min(length / 40.0, 1.0)
    target_radius = int(12 - speed_norm * (12 - 4))
    radius = int(last_radius * 0.6 + target_radius * 0.4)

    perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
    quad = np.array(
        [p1 + perp * radius, p1 - perp * radius, p2 - perp * radius, p2 + perp * radius],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.fillPoly(canvas, [quad], (255, 0, 0))
    cv2.circle(canvas, tuple(p1.astype(int)), radius, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(p2.astype(int)), radius, (255, 0, 0), -1, cv2.LINE_AA)
    return radius


class StereoDrawingTracker:
    def __init__(self, cam0=2, cam1=1, width=640, height=480):
        self.cam0 = cam0
        self.cam1 = cam1
        self.width = width
        self.height = height

        self.lock = threading.Lock()
        self.output_frame = None
        self.canvas = None
        self.running = False
        self.thread = None
        self._points = []
        self._last_radius = 8

    def clear_canvas(self):
        with self.lock:
            self.canvas = None
            self._points = []

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()

    def start(self):
        if self.running:
            return
        self._points = []
        self._last_radius = 8
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _process_loop(self):
        try:
            cap0 = _open_camera(self.cam0, self.width, self.height)
        except RuntimeError:
            available = find_cameras()
            if not available:
                with self.lock:
                    self.output_frame = self._error_frame("No cameras found")
                return
            self.cam0 = available[0]
            cap0 = _open_camera(self.cam0, self.width, self.height)

        try:
            cap1 = _open_camera(self.cam1, self.width, self.height)
        except RuntimeError:
            # Try next available camera that isn't cam0
            available = [i for i in find_cameras() if i != self.cam0]
            cap1 = _open_camera(available[0], self.width, self.height) if available else None
            if cap1 is not None:
                self.cam1 = available[0]

        single_cam = cap1 is None

        reader0 = _CameraReader(cap0)
        reader0.start()
        if single_cam:
            reader1 = reader0
        else:
            reader1 = _CameraReader(cap1)
            reader1.start()

        try:
            with _make_landmarker() as lm0, _make_landmarker() as lm1:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    while self.running:
                        frame0 = reader0.get()
                        frame1 = reader1.get()

                        if frame0 is None or frame1 is None:
                            time.sleep(0.005)
                            continue

                        frame0 = cv2.flip(frame0, 1)
                        frame1 = cv2.flip(frame1, 1)
                        h, w = frame0.shape[:2]
                        ts = int(time.time() * 1000)

                        if single_cam:
                            res0 = _detect(lm0, frame0, ts)
                            res1 = res0
                        else:
                            f0 = pool.submit(_detect, lm0, frame0, ts)
                            f1 = pool.submit(_detect, lm1, frame1, ts)
                            res0, res1 = f0.result(), f1.result()

                        tip0 = tip1 = None
                        if res0.hand_landmarks:
                            tip0 = _draw_hand(frame0, res0.hand_landmarks[0], w, h)
                        if res1.hand_landmarks:
                            tip1 = _draw_hand(frame1, res1.hand_landmarks[0], w, h)

                        pos3d = triangulate(tip0, tip1) if (not single_cam and tip0 and tip1) else None

                        label0 = f"CAM {self.cam0}" + (" (left)" if not single_cam else "")
                        label1 = f"CAM {self.cam1 if not single_cam else self.cam0}" + (" (right)" if not single_cam else "")
                        cv2.putText(frame0, label0, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame1, label1, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        combined = cv2.hconcat([frame0, frame1])

                        if not single_cam:
                            depth_str = depth_inches_to_str(pos3d)
                            depth_color = (0, 255, 0) if pos3d else (100, 100, 100)
                            cv2.putText(combined, depth_str, (20, combined.shape[0] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, depth_color, 2, cv2.LINE_AA)

                        with self.lock:
                            if self.canvas is None or self.canvas.shape != combined.shape:
                                self.canvas = np.zeros(combined.shape, dtype=np.uint8)

                            if tip0:
                                self._points.append(tip0)
                                self._last_radius = _draw_segment(
                                    self.canvas, self._points, self._last_radius
                                )
                            else:
                                self._points.clear()

                            mask = self.canvas.any(axis=2)
                            combined[mask] = self.canvas[mask]
                            self.output_frame = combined

        except Exception as exc:
            with self.lock:
                self.output_frame = self._error_frame(f"Stereo tracker error: {exc}")
        finally:
            reader0.stop()
            if not single_cam:
                reader1.stop()
            cap0.release()
            if cap1 is not None:
                cap1.release()

    @staticmethod
    def _error_frame(message, width=1280, height=480):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, message, (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam0", type=int, default=2, help="Left camera index")
    parser.add_argument("--cam1", type=int, default=1, help="Right camera index")
    parser.add_argument("--width", type=int, default=640)
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
