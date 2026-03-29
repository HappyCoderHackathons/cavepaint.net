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
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import platform

from pymongo import MongoClient
from pymongo.errors import PyMongoError
import os
from dotenv import load_dotenv

from mongo_whiteboard import MongoWhiteboardReplay
from stroke import Stroke, StrokeStore
from triangulate import depth_inches_to_str, triangulate
from swipe_detect import SwipeDetector

HAND_MODEL_PATH    = Path(__file__).with_name("hand_landmarker.task")
GESTURE_MODEL_PATH = Path(__file__).with_name("gesture_model.pth")
GESTURE_META_PATH  = Path(__file__).with_name("gesture_meta.json")
SWIPE_MODEL_PATH   = Path(__file__).with_name("swipe_model.pth")
SWIPE_META_PATH    = Path(__file__).with_name("swipe_meta.json")
INDEX_FINGERTIP    = 8
GESTURE_CONFIDENCE = 0.5

# 10-color palette (BGR) cycled by swipe left/right when palm is open
PALETTE = [
    (255, 255, 255),  # white
    ( 20,  20,  20),  # near-black
    (  0,   0, 220),  # red
    (  0, 140, 255),  # orange
    (  0, 220, 220),  # yellow
    (  0, 200,  60),  # green
    (200, 200,   0),  # cyan
    (220,  60,   0),  # blue
    (200,   0, 200),  # magenta
    (130,   0, 180),  # purple
    ( 19,  69, 139),  # brown
]
_PALM_INDICES       = [0, 5, 9, 13, 17]
SWIPE_DISPLAY_FRAMES = 35   # how long a swipe label stays on screen (~1 s)

# ---------------------------------------------------------------------------
# Gesture classifier (mirrors preview.py)
# ---------------------------------------------------------------------------

_FINGERTIP_INDICES = [4, 8, 12, 16, 20]
_INPUT_DIM = 63 + 15  # 21 landmarks × 3 coords + 15 distance features

# mongoDB setup

load_dotenv()
uri = os.getenv("MONGO_URI")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


WHITEBOARD_DB_REFRESH_MS = _env_int("WHITEBOARD_DB_REFRESH_MS", 250)
drawings_col = None
points_col = None
erases_col = None

if uri:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        db = client["cavepainting"]
        drawings_col = db["drawings"]
        points_col = db["points"]
        erases_col = db["erases"]
        client.admin.command("ping")
        drawings_col.create_index([("sessionId", 1), ("createdAt", 1)])
        points_col.create_index([("sessionId", 1), ("_id", 1)])
        erases_col.create_index([("sessionId", 1), ("_id", 1)])
    except Exception as exc:
        print(f"[mongo] Disabled (connection failed): {exc}")
        drawings_col = None
        points_col = None
        erases_col = None

class _GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _compute_features(landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords -= coords[0].copy()
    ft = coords[_FINGERTIP_INDICES]
    scale = max(np.linalg.norm(ft, axis=1).max(), 1e-6)
    coords /= scale
    ft = coords[_FINGERTIP_INDICES]
    pair_dists = [np.linalg.norm(ft[i] - ft[j]) for i in range(5) for j in range(i + 1, 5)]
    tip_dists = np.linalg.norm(ft, axis=1).tolist()
    return np.concatenate([coords.flatten(), pair_dists, tip_dists]).astype(np.float32)


class _GestureClassifier:
    def __init__(self):
        import json
        meta = json.loads(GESTURE_META_PATH.read_text())
        self.gestures = meta["gestures"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = _GestureMLP(num_classes=len(self.gestures)).to(device)
        model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        self.model = model

    def classify(self, landmarks) -> tuple[str, float]:
        """Return (gesture_name, confidence)."""
        features = _compute_features(landmarks)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)
            conf, idx = probs.max(1)
        return self.gestures[idx.item()], conf.item()

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
    # Make backend compatible for mac
    if platform.system() == 'Windows': 
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
        # Make backend compatible for mac
        if platform.system() == 'Windows': 
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)

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




class StereoDrawingTracker:
    def __init__(self, cam0=2, cam1=1, width=640, height=480):
        self.cam0 = cam0
        self.cam1 = cam1
        self.width = width
        self.height = height

        self.lock = threading.Lock()
        self.output_frame = None
        self._strokes = StrokeStore()
        self._was_drawing = False
        self.running = False
        self.thread = None
        self._active_drawing_id = None
        self._seq = 0
        self._session_id = uuid4().hex
        self._mongo_enabled = drawings_col is not None and points_col is not None
        self._mongo_whiteboard = (
            MongoWhiteboardReplay(
                points_col=points_col,
                erases_col=erases_col,
                session_id=self._session_id,
                refresh_interval_ms=WHITEBOARD_DB_REFRESH_MS,
            )
            if points_col is not None and erases_col is not None
            else None
        )
        self._color_idx = 0
        print(f"[session] Drawing session: {self._session_id}")

    @staticmethod
    def _utcnow():
        return datetime.now(timezone.utc)

    @staticmethod
    def _serialize_color(color):
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return [int(max(0, min(255, c))) for c in color]
        if isinstance(color, str):
            return color
        return "black"

    def _start_drawing_doc(self, color="black"):
        if not self._mongo_enabled:
            return
        now = self._utcnow()
        doc = {
            "createdAt": now,
            "updatedAt": now,
            "endedAt": None,
            "status": "active",
            "pointCount": 0,
            "color": self._serialize_color(color),
            "rigRotationDeg": 0,
            "sessionId": self._session_id,
        }
        try:
            result = drawings_col.insert_one(doc)
            self._active_drawing_id = result.inserted_id
            self._seq = 0
        except PyMongoError as exc:
            print(f"[mongo] drawing insert failed: {exc}")
            self._active_drawing_id = None

    def _insert_point_doc(self, x, y, z, color="black"):
        if not self._mongo_enabled or self._active_drawing_id is None:
            return
        point_doc = {
            "drawingId": self._active_drawing_id,
            "seq": self._seq,
            "t": self._utcnow(),
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "rigRotationDeg": 0,
            "color": self._serialize_color(color),
            "sessionId": self._session_id,
        }
        try:
            points_col.insert_one(point_doc)
            self._seq += 1
        except PyMongoError as exc:
            print(f"[mongo] point insert failed: {exc}")

    def _insert_erase_doc(self, x, y, radius, z=0.0):
        if erases_col is None:
            return
        erase_doc = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "radius": float(radius),
            "t": self._utcnow(),
            "sessionId": self._session_id,
        }
        if self._active_drawing_id is not None:
            erase_doc["drawingId"] = self._active_drawing_id
        try:
            erases_col.insert_one(erase_doc)
        except PyMongoError as exc:
            print(f"[mongo] erase insert failed: {exc}")

    def _finish_drawing_doc(self, status="completed"):
        if not self._mongo_enabled or self._active_drawing_id is None:
            return
        now = self._utcnow()
        try:
            drawings_col.update_one(
                {"_id": self._active_drawing_id},
                {"$set": {"updatedAt": now, "endedAt": now, "status": status, "pointCount": self._seq}},
            )
        except PyMongoError as exc:
            print(f"[mongo] drawing finalize failed: {exc}")
        finally:
            self._active_drawing_id = None
            self._seq = 0

    def clear_canvas(self):
        with self.lock:
            self._strokes.clear()

    def undo(self):
        with self.lock:
            self._strokes.undo()

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()

    @staticmethod
    def _project_whiteboard_point(
        x: float,
        y: float,
        z: float,
        yaw_rad: float,
        fov_rad: float,
        width: int,
        height: int,
    ):
        # Normalize camera-space points around the original frame center.
        xw = float(x) - 320.0
        yw = float(y) - 240.0
        zw = float(z) * 25.0 + 400.0

        c = float(np.cos(yaw_rad))
        s = float(np.sin(yaw_rad))
        xr = c * xw + s * zw
        zr = -s * xw + c * zw
        if zr <= 5.0:
            return None

        half_fov = fov_rad * 0.5
        horiz_angle = float(np.arctan2(xr, zr))
        if abs(horiz_angle) > half_fov:
            return None

        focal = (width * 0.5) / np.tan(max(half_fov, 1e-4))
        u = (width * 0.5) + (xr * focal / zr)
        v = (height * 0.5) + (yw * focal / zr)
        if u < -2 or u > width + 2 or v < -2 or v > height + 2:
            return None
        return int(u), int(v)

    def _snapshot_strokes(self):
        strokes = []
        for src in self._strokes._completed:
            dst = Stroke(color=src.color, max_radius=src.max_radius, min_radius=src.min_radius)
            dst.pts = list(src.pts)
            dst.times = list(src.times)
            strokes.append(dst)
        active = self._strokes._active
        if active and not active.empty():
            dst = Stroke(color=active.color, max_radius=active.max_radius, min_radius=active.min_radius)
            dst.pts = list(active.pts)
            dst.times = list(active.times)
            strokes.append(dst)
        return strokes

    @staticmethod
    def _project_erase_radius(
        x: float,
        y: float,
        z: float,
        radius: float,
        yaw_rad: float,
        fov_rad: float,
        width: int,
        height: int,
    ) -> int:
        c0 = StereoDrawingTracker._project_whiteboard_point(
            x, y, z, yaw_rad, fov_rad, width, height
        )
        c1 = StereoDrawingTracker._project_whiteboard_point(
            x + float(radius), y, z, yaw_rad, fov_rad, width, height
        )
        if c0 is not None and c1 is not None:
            r = int(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            return max(1, r)
        # Fallback scale from original camera width (640px).
        return max(1, int(float(radius) * (float(width) / 640.0)))

    def render_whiteboard(self, yaw_deg=0.0, fov_deg=80.0, width=960, height=260):
        width = int(np.clip(width, 320, 1920))
        height = int(np.clip(height, 120, 1080))
        yaw_deg = float(np.clip(yaw_deg, -85.0, 85.0))
        fov_deg = float(np.clip(fov_deg, 30.0, 150.0))
        yaw_rad = float(np.deg2rad(yaw_deg))
        fov_rad = float(np.deg2rad(fov_deg))

        ops = None
        if self._mongo_whiteboard is not None:
            ops = self._mongo_whiteboard.load_state()

        if ops is None:
            with self.lock:
                ops = [{"kind": "stroke", "stroke": s} for s in self._snapshot_strokes()]

        def project(px, py, pz):
            return self._project_whiteboard_point(px, py, pz, yaw_rad, fov_rad, width, height)

        board = np.full((height, width, 3), 255, dtype=np.uint8)
        for op in ops:
            if op.get("kind") == "stroke":
                stroke = op.get("stroke")
                if stroke is not None:
                    stroke.render(board, project=project)
                continue

            if op.get("kind") == "erase":
                cx = float(op.get("x", 0.0))
                cy = float(op.get("y", 0.0))
                cz = float(op.get("z", 0.0))
                cr = float(op.get("radius", 30.0))
                center = self._project_whiteboard_point(
                    cx, cy, cz, yaw_rad, fov_rad, width, height
                )
                if center is None:
                    continue
                draw_r = self._project_erase_radius(
                    cx, cy, cz, cr, yaw_rad, fov_rad, width, height
                )
                cv2.circle(board, center, draw_r, (255, 255, 255), -1, cv2.LINE_AA)

        cv2.putText(
            board,
            f"Yaw {yaw_deg:+.1f} deg | FOV {fov_deg:.0f} deg",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (40, 40, 40),
            2,
            cv2.LINE_AA,
        )
        return board

    def start(self):
        if self.running:
            return
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

        # Load gesture classifier if model exists
        gesture_clf = None
        if GESTURE_MODEL_PATH.exists() and GESTURE_META_PATH.exists():
            try:
                gesture_clf = _GestureClassifier()
            except Exception:
                pass  # no model — draw always

        # Load swipe detector if model exists
        swipe_det    = None
        swipe_events = []   # list of (label, new_color_idx, frames_remaining) — thread-local
        if SWIPE_MODEL_PATH.exists() and SWIPE_META_PATH.exists():
            try:
                swipe_det = SwipeDetector(SWIPE_MODEL_PATH, SWIPE_META_PATH)
            except Exception:
                pass

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

                        if frame0.shape != frame1.shape:
                            frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))

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
                        gesture = None
                        if res0.hand_landmarks:
                            tip0 = _draw_hand(frame0, res0.hand_landmarks[0], w, h)
                            if gesture_clf:
                                gesture, conf = gesture_clf.classify(res0.hand_landmarks[0])
                                if conf < GESTURE_CONFIDENCE:
                                    gesture = None
                        if res1.hand_landmarks:
                            tip1 = _draw_hand(frame1, res1.hand_landmarks[0], w, h)

                        drawing = (gesture == "point") if gesture_clf else (tip0 is not None)
                        erasing = (gesture == "fist") if gesture_clf else False

                        # Swipe detection — feed every frame, act only when open_hand
                        if res0.hand_landmarks and swipe_det:
                            lms = res0.hand_landmarks[0]
                            xs = [lms[i].x for i in _PALM_INDICES]
                            ys = [lms[i].y for i in _PALM_INDICES]
                            palm_x = sum(xs) / len(xs)
                            palm_y = sum(ys) / len(ys)
                            palm_sc = float(np.hypot(lms[12].x - lms[0].x,
                                                     lms[12].y - lms[0].y))
                            swipe = swipe_det.update(palm_x, palm_y, palm_sc)
                            if swipe and gesture == "open_hand":
                                label, _ = swipe
                                if label == "swipe_right":
                                    self._color_idx = (self._color_idx + 1) % len(PALETTE)
                                elif label == "swipe_left":
                                    self._color_idx = (self._color_idx - 1) % len(PALETTE)
                                if label in ("swipe_left", "swipe_right"):
                                    swipe_events.append((label, self._color_idx,
                                                         SWIPE_DISPLAY_FRAMES))

                        pos3d = triangulate(tip0, tip1) if (not single_cam and tip0 and tip1) else None

                        label0 = f"CAM {self.cam0}" + (" (left)" if not single_cam else "")
                        label1 = f"CAM {self.cam1 if not single_cam else self.cam0}" + (" (right)" if not single_cam else "")
                        cv2.putText(frame0, label0, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame1, label1, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        # Gesture label on left frame
                        if gesture:
                            g_color = (0, 255, 0) if gesture == "point" else (0, 165, 255)
                            cv2.putText(frame0, gesture, (10, 65),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, g_color, 2, cv2.LINE_AA)
                            
                        combined = cv2.hconcat([frame0, frame1])

                        if not single_cam:
                            depth_str = depth_inches_to_str(pos3d)
                            depth_color = (0, 255, 0) if pos3d else (100, 100, 100)
                            cv2.putText(combined, depth_str, (20, combined.shape[0] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, depth_color, 2, cv2.LINE_AA)

                        # Use triangulated Z when available, else 0
                        z = pos3d[2] if pos3d else 0.0

                        with self.lock:
                            if erasing and tip0:
                                if self._was_drawing:
                                    self._strokes.end()
                                    self._finish_drawing_doc(status="completed")
                                erase_radius = 30
                                self._strokes.erase_near(tip0[0], tip0[1], radius=erase_radius)
                                self._insert_erase_doc(tip0[0], tip0[1], erase_radius, z)
                                self._was_drawing = False
                            elif drawing and tip0:
                                if not self._was_drawing:
                                    active_color = PALETTE[self._color_idx]
                                    self._strokes.begin(color=active_color)
                                    self._start_drawing_doc(color=active_color)

                                active = self._strokes._active
                                before_len = len(active.pts) if active is not None else 0
                                self._strokes.add_point(tip0[0], tip0[1], z)

                                active = self._strokes._active
                                if active is not None and len(active.pts) > before_len:
                                    sx, sy, sz = active.pts[-1]
                                    self._insert_point_doc(sx, sy, sz, color=active.color)
                                self._was_drawing = True
                            else:
                                if self._was_drawing:
                                    self._strokes.end()
                                    self._finish_drawing_doc(status="completed")
                                self._was_drawing = False

                            stroke_canvas = self._strokes.render(combined.shape)
                            mask = stroke_canvas.any(axis=2)
                            combined[mask] = stroke_canvas[mask]

                            # Erase cursor
                            if erasing and tip0:
                                cv2.circle(combined, tip0, 40, (0, 0, 255), 2, cv2.LINE_AA)

                            self._draw_swipe_events(combined, swipe_events, PALETTE)
                            self._draw_palette(combined, PALETTE, self._color_idx)
                            self.output_frame = combined

                        swipe_events = [(lbl, ci, f - 1)
                                        for lbl, ci, f in swipe_events if f > 1]

        except Exception as exc:
            with self.lock:
                self.output_frame = self._error_frame(f"Stereo tracker error: {exc}")
        finally:
            if self._was_drawing:
                self._strokes.end()
                self._finish_drawing_doc(status="interrupted")
                self._was_drawing = False
            reader0.stop()
            if not single_cam:
                reader1.stop()
            cap0.release()
            if cap1 is not None:
                cap1.release()

    @staticmethod
    def _draw_swipe_events(frame, events, palette):
        """Overlay recent swipe labels, fading out as frames_remaining drops."""
        if not events:
            return
        h, w = frame.shape[:2]
        y = h // 2 - 60
        for label, color_idx, frames_left in events:
            text   = label.upper().replace('_', ' ')
            color  = palette[color_idx]
            # Dim the label proportionally as it expires
            alpha  = min(frames_left / 15.0, 1.0)
            tcolor = tuple(int(c * alpha) for c in color)
            font   = cv2.FONT_HERSHEY_SIMPLEX
            scale  = 1.2
            thick  = 3
            sz     = cv2.getTextSize(text, font, scale, thick)[0]
            x      = (w - sz[0]) // 2
            cv2.putText(frame, text, (x, y), font, scale, tcolor, thick, cv2.LINE_AA)
            y     += sz[1] + 12

    @staticmethod
    def _draw_palette(frame, palette, active_idx):
        """Draw a color palette strip at the bottom center of the frame."""
        h, w = frame.shape[:2]
        swatch = 36
        gap    = 6
        total  = len(palette) * (swatch + gap) - gap
        x0     = (w - total) // 2
        y0     = h - swatch - 10

        # Dim background strip
        cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + total + 8, y0 + swatch + 8),
                      (30, 30, 30), -1)

        for i, color in enumerate(palette):
            x = x0 + i * (swatch + gap)
            cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), color, -1)
            if i == active_idx:
                cv2.rectangle(frame, (x - 2, y0 - 2),
                              (x + swatch + 2, y0 + swatch + 2), (255, 255, 255), 2)
                # Indicator triangle above active swatch
                cx = x + swatch // 2
                pts = np.array([[cx, y0 - 5], [cx - 5, y0 - 13], [cx + 5, y0 - 13]],
                               dtype=np.int32)
                cv2.fillPoly(frame, [pts], (255, 255, 255))
            else:
                cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), (70, 70, 70), 1)

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
