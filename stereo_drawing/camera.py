"""Camera utilities: threaded reader, opener, and discovery."""

import platform
import threading

import cv2


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


def open_camera(index, width, height):
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
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
