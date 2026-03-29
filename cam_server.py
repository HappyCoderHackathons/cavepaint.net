"""Dual-camera ZeroMQ publisher + motor control API for the Libre Computer.

Camera streams (ZMQ PUSH):
  cam0 → tcp://host:5555   (/dev/video1)
  cam1 → tcp://host:5556   (/dev/video3)

Motor API (HTTP):
  POST /rotate   {"degrees": 90}   rotate by degrees (positive=forward, negative=backward)
  GET  /rotate?degrees=90          same, via query string
  GET  /status                     {"rotating": bool, "last_degrees": float}

Environment variables:
  CAM0_DEV        camera device for cam0 (default: 1)
  CAM1_DEV        camera device for cam1 (default: 3)
  CAM0_PORT       ZMQ port for cam0      (default: 5555)
  CAM1_PORT       ZMQ port for cam1      (default: 5556)
  CAM_WIDTH       capture width          (default: 640)
  CAM_HEIGHT      capture height         (default: 480)
  CAM_FPS         capture fps            (default: 30)
  JPEG_QUALITY    JPEG encode quality    (default: 75)
  API_PORT        HTTP port for motor API (default: 8080)
"""

import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import zmq

CAM0_DEV     = int(os.getenv("CAM0_DEV",     "1"))
CAM1_DEV     = int(os.getenv("CAM1_DEV",     "3"))
CAM0_PORT    = int(os.getenv("CAM0_PORT",    "5555"))
CAM1_PORT    = int(os.getenv("CAM1_PORT",    "5556"))
CAM_WIDTH    = int(os.getenv("CAM_WIDTH",    "640"))
CAM_HEIGHT   = int(os.getenv("CAM_HEIGHT",   "480"))
CAM_FPS      = int(os.getenv("CAM_FPS",      "30"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))
STREAM_WIDTH  = int(os.getenv("STREAM_WIDTH",  "0"))
STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "0"))
API_PORT     = int(os.getenv("API_PORT",     "8080"))

# ---------------------------------------------------------------------------
# Motor control
# ---------------------------------------------------------------------------

_motor_lock = threading.Lock()
_motor_status = {"rotating": False, "last_degrees": 0.0}

def _do_rotate(degrees: float):
    try:
        import lgpio
        PULSES_PER_REV = 661
        OVERSHOOT_COMPENSATION = 330
        IN1, IN2, C1 = 98, 91, 92

        chip = lgpio.gpiochip_open(1)
        lgpio.gpio_claim_output(chip, IN1)
        lgpio.gpio_claim_output(chip, IN2)
        lgpio.gpio_claim_input(chip, C1)

        target_pulses = int((abs(degrees) / 360) * PULSES_PER_REV)
        # Cap compensation so it never exceeds half the target (avoids zeroing small moves)
        compensation = min(OVERSHOOT_COMPENSATION, target_pulses // 2)
        target_pulses = max(0, target_pulses - compensation)
        pulse_count = 0
        last_state = lgpio.gpio_read(chip, C1)

        if degrees > 0:
            lgpio.gpio_write(chip, IN1, 1); lgpio.gpio_write(chip, IN2, 0)
        else:
            lgpio.gpio_write(chip, IN1, 0); lgpio.gpio_write(chip, IN2, 1)

        while pulse_count < target_pulses:
            current = lgpio.gpio_read(chip, C1)
            if current == 1 and last_state == 0:
                pulse_count += 1
            last_state = current

        lgpio.gpio_write(chip, IN1, 0); lgpio.gpio_write(chip, IN2, 0)
        lgpio.gpiochip_close(chip)
        print(f"[motor] rotated {degrees}° ({pulse_count} pulses)", flush=True)
    except Exception as exc:
        print(f"[motor] ERROR: {exc}", flush=True)
    finally:
        with _motor_lock:
            _motor_status["rotating"] = False


def rotate_async(degrees: float) -> bool:
    """Start rotation in background. Returns False if already rotating."""
    degrees = max(-720, min(720, degrees))
    with _motor_lock:
        if _motor_status["rotating"]:
            return False
        _motor_status["rotating"] = True
        _motor_status["last_degrees"] = degrees
    threading.Thread(target=_do_rotate, args=(degrees,), daemon=True).start()
    return True


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _parse_degrees(self, params) -> float | None:
        vals = params.get("degrees", [])
        if not vals:
            return None
        try:
            return float(vals[0])
        except ValueError:
            return None

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/status":
            with _motor_lock:
                self._send_json(200, dict(_motor_status))

        elif parsed.path == "/rotate":
            degrees = self._parse_degrees(params)
            if degrees is None:
                self._send_json(400, {"error": "degrees required"})
                return
            started = rotate_async(degrees)
            if started:
                self._send_json(200, {"ok": True, "degrees": degrees})
            else:
                self._send_json(409, {"error": "already rotating"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/rotate":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
            degrees = float(data["degrees"])
        except (KeyError, ValueError, json.JSONDecodeError):
            # fall back to query string
            params = parse_qs(parsed.query)
            degrees = self._parse_degrees(params)
            if degrees is None:
                self._send_json(400, {"error": "degrees required"})
                return

        started = rotate_async(degrees)
        if started:
            self._send_json(200, {"ok": True, "degrees": degrees})
        else:
            self._send_json(409, {"error": "already rotating"})


# ---------------------------------------------------------------------------
# Camera streaming
# ---------------------------------------------------------------------------

def camera_publisher(dev: int, port: int, name: str):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.set_hwm(4)  # small HWM — drop old frames, keep latency low
    sock.bind(f"tcp://0.0.0.0:{port}")

    cap = cv2.VideoCapture(dev)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"[{name}] ERROR: Cannot open /dev/video{dev}", flush=True)
        sys.exit(1)

    print(f"[{name}] /dev/video{dev} → tcp://0.0.0.0:{port}", flush=True)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    sw = STREAM_WIDTH  or CAM_WIDTH
    sh = STREAM_HEIGHT or CAM_HEIGHT
    do_resize = (sw != CAM_WIDTH or sh != CAM_HEIGHT)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.001)
            continue
        if do_resize:
            frame = cv2.resize(frame, (sw, sh))
        _, buf = cv2.imencode(".jpg", frame, encode_params)
        try:
            sock.send(buf.tobytes(), zmq.NOBLOCK)
        except zmq.Again:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = threading.Thread(target=camera_publisher, args=(CAM0_DEV, CAM0_PORT, "cam0"), daemon=True)
    t1 = threading.Thread(target=camera_publisher, args=(CAM1_DEV, CAM1_PORT, "cam1"), daemon=True)
    t0.start()
    t1.start()

    api = HTTPServer(("0.0.0.0", API_PORT), APIHandler)
    api_thread = threading.Thread(target=api.serve_forever, daemon=True)
    api_thread.start()
    print(f"[cam_server] cameras started, API on http://0.0.0.0:{API_PORT}", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[cam_server] stopped")
