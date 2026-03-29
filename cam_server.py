"""Dual-camera MJPEG HTTP server for the Libre Computer.

Serves two cameras as MJPEG streams:
  GET /cam0  →  /dev/video1
  GET /cam1  →  /dev/video3

Run on the Libre Computer:
  python3 cam_server.py

Environment variables:
  CAM0_DEV   camera device for cam0 (default: 1)
  CAM1_DEV   camera device for cam1 (default: 3)
  CAM_WIDTH  capture width  (default: 640)
  CAM_HEIGHT capture height (default: 480)
  CAM_FPS    capture fps    (default: 30)
  PORT       HTTP port      (default: 8080)
"""

import os
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

CAM0_DEV   = int(os.getenv("CAM0_DEV",   "1"))
CAM1_DEV   = int(os.getenv("CAM1_DEV",   "3"))
CAM_WIDTH  = int(os.getenv("CAM_WIDTH",  "640"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))
CAM_FPS    = int(os.getenv("CAM_FPS",    "30"))
PORT       = int(os.getenv("PORT",       "8080"))

BOUNDARY = b"frame"


class CameraSource:
    """Continuously grabs frames from a camera device in a background thread."""

    def __init__(self, dev: int):
        self._dev = dev
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._cap: cv2.VideoCapture | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._cap = cv2.VideoCapture(self._dev)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{self._dev}")
        self._thread.start()
        print(f"[cam{self._dev}] started")

    def _loop(self):
        while True:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self._lock:
                self._frame = buf.tobytes()

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._frame


# Global camera sources
cam0 = CameraSource(CAM0_DEV)
cam1 = CameraSource(CAM1_DEV)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # silence per-request logs

    def do_GET(self):
        if self.path == "/cam0":
            self._stream(cam0)
        elif self.path == "/cam1":
            self._stream(cam1)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def _stream(self, source: CameraSource):
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg = source.get_jpeg()
                if jpeg is None:
                    time.sleep(0.01)
                    continue
                header = (
                    f"--{BOUNDARY.decode()}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode()
                self.wfile.write(header + jpeg + b"\r\n")
                self.wfile.flush()
                time.sleep(1.0 / CAM_FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    cam0.start()
    cam1.start()
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[cam_server] Streaming on http://0.0.0.0:{PORT}/cam0 and /cam1")
    server.serve_forever()
