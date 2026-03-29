"""Camera utilities: threaded reader, opener, and discovery."""

import platform
import threading

import cv2
import numpy as np


class CameraReader(threading.Thread):
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


class ZmqCameraReader(threading.Thread):
    """Receives JPEG frames from a ZMQ PUSH socket (cam_server.py on Libre Computer).

    address: e.g. "zmq://10.14.232.18:5555"
    Each camera uses its own port — no topic prefix needed.
    upscale_to: optional (width, height) to resize received frames.
    """

    def __init__(self, address: str, upscale_to: tuple[int, int] | None = None):
        super().__init__(daemon=True)
        # Parse "zmq://host:port"
        rest = address[len("zmq://"):]
        self._endpoint = f"tcp://{rest}"
        self._upscale_to = upscale_to
        self._frame = None
        self._lock = threading.Lock()
        self.running = True

    def run(self):
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.set_hwm(4)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(self._endpoint)
        sock.setsockopt(zmq.SUBSCRIBE, b"")

        while self.running:
            try:
                jpeg = sock.recv()
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if self._upscale_to is not None:
                        frame = cv2.resize(frame, self._upscale_to)
                    with self._lock:
                        self._frame = frame
            except Exception:
                pass

        sock.close()
        ctx.term()

    def get(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self.running = False


def open_camera(index, width, height):
    """Open a camera by integer index, HTTP URL, or zmq:// URL."""
    if isinstance(index, str) and index.startswith("zmq://"):
        # ZMQ stream — return a ZmqCameraReader instead of a VideoCapture
        return None, index  # sentinel handled in tracker
    if isinstance(index, str):
        cap = cv2.VideoCapture(index, cv2.CAP_FFMPEG)
    elif platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not isinstance(index, str):
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
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found
