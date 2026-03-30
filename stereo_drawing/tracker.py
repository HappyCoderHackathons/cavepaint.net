"""StereoDrawingTracker: captures both cameras, runs hand detection, draws on a shared canvas."""

import asyncio
import math
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from pymongo.errors import PyMongoError

from mongo_whiteboard import MongoWhiteboardReplay
from stroke import Stroke, StrokeStore
from swipe_detect import SwipeDetector
from triangulate import depth_inches_to_str, triangulate

from .camera import CameraReader, ZmqCameraReader, find_cameras, open_camera
from .constants import (
    GESTURE_CONFIDENCE,
    GESTURE_META_PATH,
    GESTURE_MODEL_PATH,
    PALETTE,
    ENABLE_3D_PERSON,
    POSE_MODEL_PATH,
    SWIPE_DISPLAY_FRAMES,
    SWIPE_META_PATH,
    SWIPE_MODEL_PATH,
    _PALM_INDICES,
)
from .gesture import GestureClassifier, classify_rule_gesture
from .landmarker import (detect, draw_hand, make_landmarker,
                         make_pose_landmarker, detect_pose, get_segmentation_mask)
from .mongo import WHITEBOARD_DB_REFRESH_MS, actions_col, drawings_col, erases_col, points_col
from .state_slot import StateSlot


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
        self._active_draw_action_id = None
        self._active_erase_action_id = None
        self._active_erase_batch_id = None
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
        self._swipe_events = []
        self._tracking = {}
        self._was_erasing = False
        self._erase_batches = []
        self._sub_lock = threading.Lock()
        self._slots: list[StateSlot] = []
        self._canvas_version = 0
        self._stone_alpha = 0.60
        self._stone_scroll_per_turn = 3.5
        self._stone_texture = self._load_stone_texture()
        self._board_base_cache_unshifted = None
        self._board_base_cache_size = None
        self._theta = 0.0   # camera mount angle in degrees
        self._live_yaw_deg = 0.0
        self._live_fov_deg = 80.0
        print(f"[session] Drawing session: {self._session_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def set_theta(self, degrees: float):
        with self.lock:
            self._theta = float(degrees)

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            # Return latest immutable frame reference to avoid an extra full-frame copy
            # per WebRTC packet; tracker replaces this ndarray each loop.
            return self.output_frame

    def get_state(self):
        with self.lock:
            return {
                "color_idx": self._color_idx,
                "swipe_events": list(self._swipe_events),
                "tracking": dict(self._tracking),
                "canvas_version": self._canvas_version,
                "live_yaw_deg": self._live_yaw_deg,
            }

    def set_color(self, idx: int):
        with self.lock:
            self._color_idx = idx % len(PALETTE)

    def set_live_view(self, yaw_deg: float | None = None, fov_deg: float | None = None):
        with self.lock:
            if yaw_deg is not None:
                self._live_yaw_deg = float(yaw_deg) % 360.0
            if fov_deg is not None:
                self._live_fov_deg = float(np.clip(fov_deg, 30.0, 150.0))

    def add_mouse_stroke(self, points: list, color_idx: int):
        """Commit a completed mouse-drawn stroke. points is a list of (x, y, z) tuples."""
        if len(points) < 2:
            return
        color = PALETTE[color_idx % len(PALETTE)]
        stroke = Stroke(
            color=color,
            max_radius=8,
            min_radius=2,
            _smooth_alpha=0.0,
            _min_dist=1.0,
        )
        t0 = time.monotonic()
        for i, pt in enumerate(points):
            stroke.pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
            stroke.times.append(t0 + i * 0.016)
        persisted_to_mongo = self._persist_mouse_stroke(stroke)
        snapshot = None
        with self.lock:
            self._strokes._completed.append(stroke)
            self._strokes._cache = None
            if persisted_to_mongo and self._mongo_whiteboard is not None:
                self._mongo_whiteboard.invalidate()
            self._canvas_version += 1
            snapshot = {
                "color_idx": self._color_idx,
                "swipe_events": list(self._swipe_events),
                "tracking": dict(self._tracking),
                "canvas_version": self._canvas_version,
            }
        if snapshot is not None:
            self._push_state(snapshot)

    def _clear_canvas_unlocked(self):
        self._strokes.clear()
        self._erase_batches.clear()
        self._active_erase_batch_id = None
        self._active_erase_action_id = None
        self._was_erasing = False
        self._was_drawing = False
        self._canvas_version += 1

    def clear_canvas(self):
        with self.lock:
            self._clear_canvas_unlocked()

    def undo(self):
        with self.lock:
            # If a stroke is currently active, close it so undo always removes
            # the latest visible stroke.
            if self._was_drawing:
                self._strokes.end()
                self._finish_drawing_doc(status="completed")
                self._was_drawing = False
            if self._was_erasing:
                self._finish_erase_action(status="completed")
                self._was_erasing = False

            kind, ref_id = self._undo_last_action_doc()
            if kind == "erase":
                self._undo_local_erase_batch(ref_id)
            else:
                self._strokes.undo()
            if self._mongo_whiteboard is not None:
                self._mongo_whiteboard.invalidate()
            self._canvas_version += 1

    def subscribe(self, loop: asyncio.AbstractEventLoop) -> StateSlot:
        slot = StateSlot(loop)
        with self._sub_lock:
            self._slots.append(slot)
        return slot

    def unsubscribe(self, slot: StateSlot) -> None:
        with self._sub_lock:
            self._slots = [s for s in self._slots if s is not slot]

    # ------------------------------------------------------------------
    # Whiteboard rendering
    # ------------------------------------------------------------------

    def render_whiteboard(self, yaw_deg=0.0, fov_deg=80.0, width=960, height=260):
        ops = None
        if self._mongo_whiteboard is not None:
            ops = self._mongo_whiteboard.load_state()
        if ops is None or len(ops) == 0:
            with self.lock:
                ops = [{"kind": "stroke", "stroke": s} for s in self._snapshot_strokes()]
        return self.render_ops(ops, yaw_deg=yaw_deg, fov_deg=fov_deg, width=width, height=height)

    def render_ops(self, ops, yaw_deg=0.0, fov_deg=80.0, width=960, height=260):
        """Render an arbitrary ops list (strokes + erases) to a board image."""
        width = int(np.clip(width, 320, 1920))
        height = int(np.clip(height, 120, 1080))
        yaw_deg = float(yaw_deg) % 360.0
        fov_deg = float(np.clip(fov_deg, 30.0, 150.0))
        yaw_rad = float(np.deg2rad(yaw_deg))
        fov_rad = float(np.deg2rad(fov_deg))

        stroke_layer = np.zeros((height, width, 3), dtype=np.uint8)
        for op in ops:
            if op.get("kind") == "stroke":
                stroke = op.get("stroke")
                if stroke is not None:
                    theta_deg = getattr(stroke, "theta", 0.0)
                    def project(px, py, pz, _t=theta_deg):
                        return self._project_whiteboard_point(px, py, pz, _t, yaw_rad, fov_rad, width, height)
                    stroke.render(stroke_layer, project=project)
                continue

            if op.get("kind") == "erase":
                cx = float(op.get("x", 0.0))
                cy = float(op.get("y", 0.0))
                cz = float(op.get("z", 0.0))
                cr = float(op.get("radius", 30.0))
                erase_theta = float(op.get("theta", 0.0))
                for ez in (cz - 4.0, cz, cz + 4.0):
                    center = self._project_whiteboard_point(cx, cy, ez, erase_theta, yaw_rad, fov_rad, width, height)
                    if center is None:
                        continue
                    draw_r = self._project_erase_radius(cx, cy, ez, cr, erase_theta, yaw_rad, fov_rad, width, height)
                    cv2.circle(
                        stroke_layer,
                        (int(round(center[0])), int(round(center[1]))),
                        int(max(1, round(draw_r * 1.08))),
                        (0, 0, 0),
                        -1,
                        cv2.LINE_AA,
                    )

        board = self._get_whiteboard_base(width, height, yaw_deg).copy()
        ink_mask = stroke_layer.any(axis=2)
        board[ink_mask] = stroke_layer[ink_mask]
        return board

    def _load_stone_texture(self):
        root_dir = Path(__file__).resolve().parent.parent
        texture_path = root_dir / "static" / "images" / "stone.jpeg"
        if not texture_path.exists():
            return None
        texture = cv2.imread(str(texture_path), cv2.IMREAD_COLOR)
        return texture if texture is not None else None

    def _get_whiteboard_base(self, width: int, height: int, yaw_deg: float):
        cache_hit = (
            self._board_base_cache_unshifted is not None
            and self._board_base_cache_size == (width, height)
        )
        if cache_hit:
            base_unshifted = self._board_base_cache_unshifted
        else:
            base_unshifted = np.full((height, width, 3), 255, dtype=np.uint8)
            if self._stone_texture is not None:
                texture = cv2.resize(
                    self._stone_texture,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                base_unshifted = cv2.addWeighted(
                    base_unshifted,
                    1.0 - self._stone_alpha,
                    texture,
                    self._stone_alpha,
                    0.0,
                )
            self._board_base_cache_unshifted = base_unshifted
            self._board_base_cache_size = (width, height)

        if width <= 1:
            return base_unshifted

        shift_px = int(round((float(yaw_deg) % 360.0) / 360.0 * width * self._stone_scroll_per_turn))
        if shift_px == 0:
            return base_unshifted
        return np.roll(base_unshifted, shift=shift_px, axis=1)

    # ------------------------------------------------------------------
    # Static overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_swipe_events(frame, events, palette):
        """Overlay recent swipe labels, fading out as frames_remaining drops."""
        if not events:
            return
        h, w = frame.shape[:2]
        y = h // 2 - 60
        for label, color_idx, frames_left in events:
            text   = label.upper().replace("_", " ")
            color  = palette[color_idx]
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

        cv2.rectangle(frame, (x0 - 8, y0 - 18), (x0 + total + 8, y0 + swatch + 8), (30, 30, 30), -1)

        for i, color in enumerate(palette):
            x = x0 + i * (swatch + gap)
            cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), color, -1)
            if i == active_idx:
                cv2.rectangle(frame, (x - 2, y0 - 2), (x + swatch + 2, y0 + swatch + 2), (255, 255, 255), 2)
                cx  = x + swatch // 2
                pts = np.array([[cx, y0 - 5], [cx - 5, y0 - 13], [cx + 5, y0 - 13]], dtype=np.int32)
                cv2.fillPoly(frame, [pts], (255, 255, 255))
            else:
                cv2.rectangle(frame, (x, y0), (x + swatch, y0 + swatch), (70, 70, 70), 1)

    @staticmethod
    def _error_frame(message, width=1280, height=480):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, message, (20, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    def _push_state(self, snapshot: dict) -> None:
        with self._sub_lock:
            for slot in self._slots:
                slot.put_threadsafe(snapshot)

    def _snapshot_strokes(self):
        strokes = []
        for src in self._strokes._completed:
            strokes.append(self._clone_stroke(src))
        active = self._strokes._active
        if active and not active.empty():
            strokes.append(self._clone_stroke(active))
        return strokes

    @staticmethod
    def _clone_stroke(src: Stroke) -> Stroke:
        dst = Stroke(
            color=src.color,
            max_radius=src.max_radius,
            min_radius=src.min_radius,
            _smooth_alpha=src._smooth_alpha,
            _min_dist=src._min_dist,
            _dynamic_strength=src._dynamic_strength,
        )
        dst.pts = list(src.pts)
        dst.times = list(src.times)
        return dst

    # Brio 101 horizontal FOV at 640px width
    _CAMERA_HFOV_DEG = 65.0
    _PX_PER_DEG = 640.0 / _CAMERA_HFOV_DEG  # ≈ 9.85 px per degree of camera rotation

    @staticmethod
    def _project_whiteboard_point(x, y, z, theta_deg, yaw_rad, fov_rad, width, height):
        # Offset x by camera mount angle so strokes at different theta land at the right world position
        xw = (float(x) - 320.0) + theta_deg * StereoDrawingTracker._PX_PER_DEG
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
        return float(u), float(v)

    @staticmethod
    def _project_erase_radius(x, y, z, radius, theta_deg, yaw_rad, fov_rad, width, height):
        c0 = StereoDrawingTracker._project_whiteboard_point(x, y, z, theta_deg, yaw_rad, fov_rad, width, height)
        c1 = StereoDrawingTracker._project_whiteboard_point(x + float(radius), y, z, theta_deg, yaw_rad, fov_rad, width, height)
        if c0 is not None and c1 is not None:
            r = int(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            return max(1, r)
        return max(1, int(float(radius) * (float(width) / 640.0)))

    # ------------------------------------------------------------------
    # MongoDB helpers
    # ------------------------------------------------------------------

    def _persist_mouse_stroke(self, stroke: Stroke) -> bool:
        """Persist a completed mouse stroke without mutating active hand-drawing state."""
        if not self._mongo_enabled:
            return False

        now = self._utcnow()
        brush_radius = int(max(1, round(float(getattr(stroke, "max_radius", 8)))))
        drawing_doc = {
            "createdAt": now,
            "updatedAt": now,
            "endedAt": now,
            "status": "completed",
            "pointCount": len(stroke.pts),
            "color": self._serialize_color(stroke.color),
            "rigRotationDeg": 0,
            "sessionId": self._session_id,
            "brushRadius": brush_radius,
        }

        try:
            drawing_result = drawings_col.insert_one(drawing_doc)
            drawing_id = drawing_result.inserted_id
        except PyMongoError as exc:
            print(f"[mongo] mouse drawing insert failed: {exc}")
            return False

        action_id = None
        if actions_col is not None:
            try:
                action_doc = {
                    "sessionId": self._session_id,
                    "kind": "draw",
                    "drawingId": drawing_id,
                    "createdAt": now,
                    "updatedAt": now,
                    "endedAt": now,
                    "status": "completed",
                }
                action_id = actions_col.insert_one(action_doc).inserted_id
            except PyMongoError as exc:
                print(f"[mongo] mouse draw action insert failed: {exc}")
                action_id = None

        if action_id is not None:
            try:
                drawings_col.update_one(
                    {"_id": drawing_id},
                    {"$set": {"actionId": action_id}},
                )
            except PyMongoError as exc:
                print(f"[mongo] mouse drawing action link failed: {exc}")

        point_docs = []
        base_t = float(now.timestamp())
        serialized_color = self._serialize_color(stroke.color)
        for i, (x, y, z) in enumerate(stroke.pts):
            point_docs.append({
                "drawingId": drawing_id,
                "seq": i,
                "t": base_t + i * 0.016,
                "position": {"x": float(x), "y": float(y), "z": float(z)},
                "rigRotationDeg": 0,
                "color": serialized_color,
                "brushRadius": brush_radius,
                "sessionId": self._session_id,
            })

        written_points = 0
        if point_docs:
            try:
                result = points_col.insert_many(point_docs, ordered=True)
                written_points = len(result.inserted_ids)
            except PyMongoError as exc:
                print(f"[mongo] mouse point batch insert failed: {exc}")

        if written_points != len(point_docs):
            try:
                drawings_col.update_one(
                    {"_id": drawing_id},
                    {"$set": {"pointCount": written_points, "updatedAt": self._utcnow()}},
                )
            except PyMongoError as exc:
                print(f"[mongo] mouse drawing pointCount reconcile failed: {exc}")

        return written_points > 0

    def _start_drawing_doc(self, color="black", brush_radius=None):
        if not self._mongo_enabled:
            return
        now = self._utcnow()
        doc = {
            "createdAt": now, "updatedAt": now, "endedAt": None,
            "status": "active", "pointCount": 0,
            "color": self._serialize_color(color),
            "rigRotationDeg": 0,
            "sessionId": self._session_id,
        }
        if brush_radius is not None:
            doc["brushRadius"] = int(max(1, round(float(brush_radius))))
        try:
            result = drawings_col.insert_one(doc)
            self._active_drawing_id = result.inserted_id
            self._active_draw_action_id = None
            if actions_col is not None:
                try:
                    action_doc = {
                        "sessionId": self._session_id,
                        "kind": "draw",
                        "drawingId": self._active_drawing_id,
                        "createdAt": now,
                        "updatedAt": now,
                        "endedAt": None,
                        "status": "active",
                    }
                    action_result = actions_col.insert_one(action_doc)
                    self._active_draw_action_id = action_result.inserted_id
                    drawings_col.update_one(
                        {"_id": self._active_drawing_id},
                        {"$set": {"actionId": self._active_draw_action_id}},
                    )
                except PyMongoError as exc:
                    print(f"[mongo] draw action insert failed: {exc}")
                    self._active_draw_action_id = None
            self._seq = 0
        except PyMongoError as exc:
            print(f"[mongo] drawing insert failed: {exc}")
            self._active_drawing_id = None
            self._active_draw_action_id = None

    def _insert_point_doc(self, x, y, z, color="black", brush_radius=None):
        if not self._mongo_enabled or self._active_drawing_id is None:
            return
        point_doc = {
            "drawingId": self._active_drawing_id, "seq": self._seq, "t": self._utcnow(),
            "position": {"x": float(x), "y": float(y), "z": float(z)},
            "rigRotationDeg": 0,
            "color": self._serialize_color(color),
            "sessionId": self._session_id,
        }
        if brush_radius is not None:
            point_doc["brushRadius"] = int(max(1, round(float(brush_radius))))
        try:
            points_col.insert_one(point_doc)
            self._seq += 1
        except PyMongoError as exc:
            print(f"[mongo] point insert failed: {exc}")

    def _insert_erase_doc(self, x, y, radius, z=0.0):
        self._record_local_erase_point(x, y, radius)
        if erases_col is None:
            return
        erase_doc = {
            "x": float(x), "y": float(y), "z": float(z),
            "radius": float(radius), "t": self._utcnow(),
            "sessionId": self._session_id,
        }
        if self._active_erase_batch_id is not None:
            erase_doc["eraseBatchId"] = self._active_erase_batch_id
        if self._active_erase_action_id is not None:
            erase_doc["actionId"] = self._active_erase_action_id
        if self._active_drawing_id is not None:
            erase_doc["drawingId"] = self._active_drawing_id
        try:
            erases_col.insert_one(erase_doc)
        except PyMongoError as exc:
            print(f"[mongo] erase insert failed: {exc}")

    def _start_erase_action(self):
        self._active_erase_batch_id = uuid4().hex
        self._erase_batches.append({"batchId": self._active_erase_batch_id, "points": []})
        self._active_erase_action_id = None
        if actions_col is None:
            return
        now = self._utcnow()
        action_doc = {
            "sessionId": self._session_id,
            "kind": "erase",
            "eraseBatchId": self._active_erase_batch_id,
            "createdAt": now,
            "updatedAt": now,
            "endedAt": None,
            "status": "active",
        }
        try:
            result = actions_col.insert_one(action_doc)
            self._active_erase_action_id = result.inserted_id
        except PyMongoError as exc:
            print(f"[mongo] erase action insert failed: {exc}")
            self._active_erase_action_id = None

    def _finish_erase_action(self, status="completed"):
        now = self._utcnow()
        if actions_col is not None and self._active_erase_action_id is not None:
            try:
                actions_col.update_one(
                    {"_id": self._active_erase_action_id},
                    {"$set": {"updatedAt": now, "endedAt": now, "status": status}},
                )
            except PyMongoError as exc:
                print(f"[mongo] erase action finalize failed: {exc}")
        self._active_erase_action_id = None
        self._active_erase_batch_id = None

    def _start_clear_action(self):
        self.clear_canvas()

    def _record_local_erase_point(self, x, y, radius):
        if self._active_erase_batch_id is None:
            self._active_erase_batch_id = uuid4().hex
            self._erase_batches.append({"batchId": self._active_erase_batch_id, "points": []})
        target = None
        for batch in reversed(self._erase_batches):
            if batch.get("batchId") == self._active_erase_batch_id:
                target = batch
                break
        if target is None:
            target = {"batchId": self._active_erase_batch_id, "points": []}
            self._erase_batches.append(target)
        target["points"].append((float(x), float(y), float(radius)))

    def _undo_local_erase_batch(self, erase_batch_id=None):
        if not self._erase_batches:
            return
        removed = False
        if erase_batch_id is None:
            self._erase_batches.pop()
            removed = True
        else:
            for idx in range(len(self._erase_batches) - 1, -1, -1):
                if self._erase_batches[idx].get("batchId") == erase_batch_id:
                    self._erase_batches.pop(idx)
                    removed = True
                    break
        if not removed:
            return
        self._reapply_local_erases()

    def _reapply_local_erases(self):
        shape = self.output_frame.shape if self.output_frame is not None else (self.height, self.width, 3)
        self._strokes._cache = None
        self._strokes._dirty = False
        self._strokes._pixel_edited = False
        self._strokes.render(shape)
        for batch in self._erase_batches:
            for x, y, radius in batch.get("points", []):
                self._strokes.erase_near(x, y, radius=radius)

    def _finish_drawing_doc(self, status="completed"):
        if not self._mongo_enabled or self._active_drawing_id is None:
            return
        drawing_id = self._active_drawing_id
        action_id = self._active_draw_action_id
        now = self._utcnow()
        try:
            drawings_col.update_one(
                {"_id": drawing_id},
                {"$set": {"updatedAt": now, "endedAt": now, "status": status, "pointCount": self._seq}},
            )
        except PyMongoError as exc:
            print(f"[mongo] drawing finalize failed: {exc}")
        if actions_col is not None and action_id is not None:
            try:
                actions_col.update_one(
                    {"_id": action_id},
                    {"$set": {"updatedAt": now, "endedAt": now, "status": status}},
                )
            except PyMongoError as exc:
                print(f"[mongo] draw action finalize failed: {exc}")
        self._active_drawing_id = None
        self._active_draw_action_id = None
        self._seq = 0

    def _undo_drawing_doc_by_id(self, drawing_id):
        if drawing_id is None or drawings_col is None or points_col is None:
            return
        points_col.delete_many({"sessionId": self._session_id, "drawingId": drawing_id})
        if erases_col is not None:
            erases_col.delete_many({"sessionId": self._session_id, "drawingId": drawing_id})
        now = self._utcnow()
        drawings_col.update_one(
            {"_id": drawing_id},
            {"$set": {"updatedAt": now, "endedAt": now, "status": "undone", "pointCount": 0}},
        )

    def _undo_last_drawing_doc(self):
        if drawings_col is None or points_col is None:
            return
        try:
            latest = drawings_col.find_one(
                {"sessionId": self._session_id, "status": {"$in": ["active", "completed"]}},
                projection={"_id": 1},
                sort=[("_id", -1)],
            )
            if latest is None:
                return

            drawing_id = latest.get("_id")
            if drawing_id is None:
                return

            self._undo_drawing_doc_by_id(drawing_id)
        except PyMongoError as exc:
            print(f"[mongo] undo failed: {exc}")

    def _undo_last_action_doc(self):
        if actions_col is None:
            self._undo_last_drawing_doc()
            return "draw", None
        try:
            latest = actions_col.find_one(
                {"sessionId": self._session_id, "status": {"$in": ["active", "completed"]}},
                sort=[("_id", -1)],
            )
            if latest is None:
                self._undo_last_drawing_doc()
                return "draw", None

            now = self._utcnow()
            action_id = latest.get("_id")
            kind = str(latest.get("kind", "draw"))

            if kind == "erase":
                erase_batch_id = latest.get("eraseBatchId")
                if erases_col is not None and erase_batch_id is not None:
                    erases_col.delete_many(
                        {"sessionId": self._session_id, "eraseBatchId": erase_batch_id}
                    )
                actions_col.update_one(
                    {"_id": action_id},
                    {"$set": {"updatedAt": now, "endedAt": now, "status": "undone"}},
                )
                return "erase", erase_batch_id

            drawing_id = latest.get("drawingId")
            if drawing_id is None and drawings_col is not None and action_id is not None:
                linked = drawings_col.find_one(
                    {"sessionId": self._session_id, "actionId": action_id},
                    projection={"_id": 1},
                )
                drawing_id = linked.get("_id") if linked else None
            self._undo_drawing_doc_by_id(drawing_id)
            actions_col.update_one(
                {"_id": action_id},
                {"$set": {"updatedAt": now, "endedAt": now, "status": "undone"}},
            )
            return "draw", drawing_id
        except PyMongoError as exc:
            print(f"[mongo] action undo failed: {exc}")
            self._undo_last_drawing_doc()
            return "draw", None

    def _submit_and_start_new_session(self):
        prev_session = self._session_id

        if self._was_drawing:
            self._strokes.end()
            self._finish_drawing_doc(status="completed")
            self._was_drawing = False
        if self._was_erasing:
            self._finish_erase_action(status="completed")
            self._was_erasing = False

        self._strokes.clear()
        self._erase_batches.clear()
        self._active_drawing_id = None
        self._active_draw_action_id = None
        self._active_erase_action_id = None
        self._active_erase_batch_id = None
        self._seq = 0

        self._session_id = uuid4().hex
        if points_col is not None and erases_col is not None:
            self._mongo_whiteboard = MongoWhiteboardReplay(
                points_col=points_col,
                erases_col=erases_col,
                session_id=self._session_id,
                refresh_interval_ms=WHITEBOARD_DB_REFRESH_MS,
            )
        else:
            self._mongo_whiteboard = None

        print(f"[session] Drawing session: {self._session_id} (submitted {prev_session})")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def _process_loop(self):
        cap0 = None
        cap1 = None
        single_cam = True
        reader0 = None
        reader1 = None
        try:
            cap0, reader0 = self._open_cam0()
            cap1, reader1_or_none, single_cam = self._open_cam1()

            reader0.start()
            if single_cam:
                reader1 = reader0
            else:
                reader1 = reader1_or_none
                reader1.start()

            gesture_clf = self._load_gesture_classifier()
            swipe_det, swipe_events = self._load_swipe_detector()

            try:
                with ExitStack() as stack:
                    lm0 = stack.enter_context(make_landmarker())
                    lm1 = stack.enter_context(make_landmarker())
                    has_pose = POSE_MODEL_PATH.exists()
                    pose_lm0 = stack.enter_context(make_pose_landmarker()) if has_pose else None
                    pose_lm1 = None  # cam1 pose unused; dropping it saves one full inference per frame
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        self._run_loop(
                            lm0, lm1, pool,
                            reader0, reader1, single_cam,
                            gesture_clf, swipe_det, swipe_events,
                            pose_lm0=pose_lm0, pose_lm1=pose_lm1,
                        )
            except Exception as exc:
                print(f"[tracker] processing loop error: {exc}")
                traceback.print_exc()
                with self.lock:
                    self.output_frame = self._error_frame(f"Stereo tracker error: {exc}")
        finally:
            if self._was_drawing:
                self._strokes.end()
                self._finish_drawing_doc(status="interrupted")
                self._was_drawing = False
            if self._was_erasing:
                self._finish_erase_action(status="interrupted")
                self._was_erasing = False
            if reader0 is not None:
                reader0.stop()
            if not single_cam and reader1 is not None:
                reader1.stop()
            if cap0 is not None:
                cap0.release()
            if cap1 is not None:
                cap1.release()

    def _open_cam0(self):
        if isinstance(self.cam0, str) and self.cam0.startswith("zmq://"):
            reader = ZmqCameraReader(self.cam0, upscale_to=(self.width, self.height))
            return None, reader
        try:
            cap = open_camera(self.cam0, self.width, self.height)
            return cap, CameraReader(cap)
        except RuntimeError:
            available = find_cameras()
            if not available:
                with self.lock:
                    self.output_frame = self._error_frame("No cameras found")
                raise
            self.cam0 = available[0]
            cap = open_camera(self.cam0, self.width, self.height)
            return cap, CameraReader(cap)

    def _open_cam1(self):
        if isinstance(self.cam1, str) and self.cam1.startswith("zmq://"):
            reader = ZmqCameraReader(self.cam1, upscale_to=(self.width, self.height))
            return None, reader, False
        try:
            cap1 = open_camera(self.cam1, self.width, self.height)
            return cap1, CameraReader(cap1), False
        except RuntimeError:
            available = [i for i in find_cameras() if i != self.cam0]
            if available:
                self.cam1 = available[0]
                cap1 = open_camera(self.cam1, self.width, self.height)
                return cap1, CameraReader(cap1), False
            return None, None, True

    def _load_gesture_classifier(self):
        if GESTURE_MODEL_PATH.exists() and GESTURE_META_PATH.exists():
            try:
                return GestureClassifier()
            except Exception as exc:
                print(f"[gesture] classifier disabled: {exc}")
        else:
            print(
                f"[gesture] classifier disabled: missing files "
                f"(model={GESTURE_MODEL_PATH.exists()}, meta={GESTURE_META_PATH.exists()})"
            )
        return None

    def _load_swipe_detector(self):
        if SWIPE_MODEL_PATH.exists() and SWIPE_META_PATH.exists():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return SwipeDetector(SWIPE_MODEL_PATH, SWIPE_META_PATH, device=device), []
            except Exception:
                pass
        return None, []

    def _run_loop(self, lm0, lm1, pool, reader0, reader1, single_cam, gesture_clf, swipe_det, swipe_events, pose_lm0=None, pose_lm1=None):
        _fps = 0.0
        _fps_count = 0
        _fps_t0 = time.monotonic()
        _smooth_pinch = 0.5
        _resize_cooldown = 0
        _small_radius_lock_until = 0.0
        _last_person_z = None
        _last_person_mask = None
        _small_radius_lock_px = 16
        _small_radius_lock_sec = 0.8
        _opposite_swipe_lock_sec = 0.70
        _swipe_block_until = {"swipe_left": 0.0, "swipe_right": 0.0}
        _prev_action = "idle"
        _was_gun_gesture = False
        _last_gun_clear_mono = 0.0
        _gun_clear_cooldown_sec = float(os.getenv("GUN_CLEAR_COOLDOWN_SEC", "1.2"))
        _pose_throttle = max(1, int(os.getenv("POSE_THROTTLE", "3")))
        _pose_frame_idx = 0
        _infer_w = int(os.getenv("INFER_W", "320"))
        _infer_h = int(os.getenv("INFER_H", "240"))

        while self.running:
            frame0 = reader0.get()
            frame1 = reader1.get()
            if frame0 is None or frame1 is None:
                time.sleep(0.005)
                continue

            frame0 = cv2.flip(frame0, 1)
            frame1 = cv2.flip(frame1, 1)
            raw_frame = frame0.copy()

            if frame0.shape != frame1.shape:
                frame1 = cv2.resize(frame1, (frame0.shape[1], frame0.shape[0]))

            h, w = frame0.shape[:2]
            ts = int(time.time() * 1000)

            # Downscale frames for faster MediaPipe inference
            if (w, h) != (_infer_w, _infer_h):
                f0s = cv2.resize(frame0, (_infer_w, _infer_h))
                f1s = cv2.resize(frame1, (_infer_w, _infer_h)) if not single_cam else f0s
            else:
                f0s, f1s = frame0, frame1

            run_pose = (_pose_frame_idx % _pose_throttle == 0)
            _pose_frame_idx += 1

            if single_cam:
                res0 = detect(lm0, f0s, ts)
                res1 = res0
                pose_res0 = detect_pose(pose_lm0, f0s, ts) if (pose_lm0 and run_pose) else None
                pose_res1 = None
            else:
                fh0 = pool.submit(detect, lm0, f0s, ts)
                fh1 = pool.submit(detect, lm1, f1s, ts)
                fp0 = pool.submit(detect_pose, pose_lm0, f0s, ts) if (pose_lm0 and run_pose) else None
                res0, res1 = fh0.result(), fh1.result()
                pose_res0 = fp0.result() if fp0 else None
                pose_res1 = None

            tip0 = tip1 = tip8 = tip12 = None
            gesture = None
            conf = None
            gesture_source = None
            if res0.hand_landmarks:
                hand = res0.hand_landmarks[0]
                tip0 = draw_hand(frame0, hand, w, h)
                # `raw_frame` is later used for person-mask compositing. Draw the same
                # hand dots onto it too so the dots are never hidden.
                draw_hand(raw_frame, hand, w, h)
                tip8 = hand[8]
                tip12 = hand[12]
                if gesture_clf:
                    ml_gesture, ml_conf = gesture_clf.classify(hand)
                    conf = float(ml_conf)
                    if ml_conf >= GESTURE_CONFIDENCE:
                        gesture = ml_gesture
                        gesture_source = "ml"
                # Fallback to deterministic geometry when ML confidence is too low
                # or classifier unavailable.
                if gesture is None:
                    rule_gesture, rule_score = classify_rule_gesture(hand)
                    if rule_gesture is not None:
                        gesture = rule_gesture
                        conf = float(rule_score)
                        gesture_source = "rule"
            if res1.hand_landmarks:
                tip1 = draw_hand(frame1, res1.hand_landmarks[0], w, h)

            # Gesture → action flags
            if gesture_clf:
                drawing  = (gesture == "point")
                resizing = (gesture == "peace")
                erasing  = (gesture == "fist")
                clearing = (gesture == "gun")
            else:
                drawing = resizing = erasing = clearing = False

            if resizing:
                action = "resizing"
            elif drawing:
                action = "drawing"
            elif erasing:
                action = "erasing"
            elif clearing:
                action = "clearing"
            else:
                action = "idle"

            # Entering resize mode from drawing should feel immediate.
            #if action == "resizing" and _prev_action == "drawing":
                #_small_radius_lock_until = 0.0

            if resizing:
                _resize_cooldown = 18
            elif _resize_cooldown > 0:
                _resize_cooldown -= 1
                drawing = False

            submit_swipe_up = False

            # Swipe detection
            if res0.hand_landmarks and swipe_det:
                lms = res0.hand_landmarks[0]
                xs = [lms[i].x for i in _PALM_INDICES]
                ys = [lms[i].y for i in _PALM_INDICES]
                palm_x = sum(xs) / len(xs)
                palm_y = sum(ys) / len(ys)
                palm_sc = float(np.hypot(lms[12].x - lms[0].x, lms[12].y - lms[0].y))
                swipe = swipe_det.update(palm_x, palm_y, palm_sc)
                if swipe and gesture == "open_hand":
                    label, _ = swipe
                    now_mono = time.monotonic()
                    accepted_swipe = False
                    if label == "swipe_right":
                        if now_mono < _swipe_block_until["swipe_right"]:
                            label = None
                        else:
                            self._color_idx = (self._color_idx + 1) % len(PALETTE)
                            _swipe_block_until["swipe_left"] = now_mono + _opposite_swipe_lock_sec
                            accepted_swipe = True
                    elif label == "swipe_left":
                        if now_mono < _swipe_block_until["swipe_left"]:
                            label = None
                        else:
                            self._color_idx = (self._color_idx - 1) % len(PALETTE)
                            _swipe_block_until["swipe_right"] = now_mono + _opposite_swipe_lock_sec
                            accepted_swipe = True
                    elif label == "swipe_up":
                        submit_swipe_up = True
                        accepted_swipe = True
                    if accepted_swipe and label in ("swipe_left", "swipe_right", "swipe_up"):
                        swipe_events.append((label, self._color_idx, SWIPE_DISPLAY_FRAMES))

            pos3d = triangulate(tip0, tip1) if (not single_cam and tip0 and tip1) else None

            person_mask = _last_person_mask if ENABLE_3D_PERSON else None
            if ENABLE_3D_PERSON:
                fresh_mask = get_segmentation_mask(pose_res0) if pose_res0 else None
                if fresh_mask is not None:
                    if fresh_mask.shape[:2] != (h, w):
                        fresh_mask = cv2.resize(fresh_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        fresh_mask = np.clip(fresh_mask, 0.0, 1.0)
                    _last_person_mask = fresh_mask
                    person_mask = _last_person_mask

            # Triangulate a stable body point (nose, landmark 0) for person_z
            body_pos3d = None
            if pose_res0 and pose_res0.pose_landmarks and pose_res1 and pose_res1.pose_landmarks:
                nose0 = pose_res0.pose_landmarks[0][0]
                nose1 = pose_res1.pose_landmarks[0][0]
                nose_px0 = (nose0.x * w, nose0.y * h)
                nose_px1 = (nose1.x * w, nose1.y * h)
                body_pos3d = triangulate(nose_px0, nose_px1)

            # Frame labels
            label0 = f"CAM {self.cam0}" + (" (left)" if not single_cam else "")
            label1 = f"CAM {self.cam1 if not single_cam else self.cam0}" + (" (right)" if not single_cam else "")
            cv2.putText(frame0, label0, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame1, label1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if gesture:
                g_color = (0, 255, 0) if gesture == "point" else (0, 165, 255)
                cv2.putText(frame0, gesture, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, g_color, 2, cv2.LINE_AA)

            if resizing and tip8 and tip12:
                px8  = (int(tip8.x  * w), int(tip8.y  * h))
                px12 = (int(tip12.x * w), int(tip12.y * h))
                mid  = ((px8[0] + px12[0]) // 2, (px8[1] + px12[1]) // 2)
                br   = self._strokes.current_radius
                color = PALETTE[self._color_idx]
                cv2.line(frame0, px8, px12, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(frame0, mid, br, color, -1, cv2.LINE_AA)
                cv2.circle(frame0, mid, br, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame0, f"{br}px", (mid[0] + br + 6, mid[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            if not single_cam:
                depth_str = depth_inches_to_str(pos3d)
                depth_color = (0, 255, 0) if pos3d else (100, 100, 100)
                cv2.putText(frame0, depth_str, (20, frame0.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, depth_color, 2, cv2.LINE_AA)

            # Stroke Z relative to shoulder midpoint (in meters).
            # Negative = hand in front of shoulders = stroke in front of person.
            # Positive = hand behind shoulders = stroke behind person.
            # Falls back to stereo depth if pose unavailable.
            hand_world_z = None
            shoulder_world_z = None
            if pose_res0 and pose_res0.pose_world_landmarks:
                wl = pose_res0.pose_world_landmarks[0]
                shoulder_world_z = (wl[11].z + wl[12].z) / 2
                hand_world_z = min(wl[15].z, wl[16].z)  # forward-most wrist
            # In single-camera mode we keep drawing in a flat Z plane.
            # This avoids unstable pseudo-depth from pose-only estimates.
            if single_cam:
                z = 0.0
            else:
                z = hand_world_z if hand_world_z is not None else (pos3d[2] if pos3d else 0.0)

            with self.lock:
                live_yaw_deg = float(self._live_yaw_deg) % 360.0
                live_fov_deg = float(np.clip(self._live_fov_deg, 30.0, 150.0))
                live_yaw_signed = live_yaw_deg if live_yaw_deg <= 180.0 else (live_yaw_deg - 360.0)
                use_projected_live = abs(live_yaw_signed) > 0.05
                live_project = None
                if use_projected_live:
                    yaw_rad = float(np.deg2rad(live_yaw_deg))
                    fov_rad = float(np.deg2rad(live_fov_deg))
                    theta_deg = float(self._theta)

                    def live_project(px, py, pz):
                        return self._project_whiteboard_point(px, py, pz, theta_deg, yaw_rad, fov_rad, w, h)

                if submit_swipe_up:
                    self._submit_and_start_new_session()
                    self._canvas_version += 1

                now_mono = time.monotonic()
                if gesture_clf and gesture == "gun":
                    if (
                        not _was_gun_gesture
                        and (now_mono - _last_gun_clear_mono) >= _gun_clear_cooldown_sec
                    ):
                        self._clear_canvas_unlocked()
                        _last_gun_clear_mono = now_mono
                _was_gun_gesture = bool(gesture_clf and gesture == "gun")

                if erasing and tip0 and not self._was_erasing:
                    self._start_erase_action()
                elif (not erasing or not tip0) and self._was_erasing:
                    self._finish_erase_action(status="completed")
                    self._was_erasing = False

                if erasing and tip0:
                    if self._was_drawing:
                        self._strokes.end()
                        self._finish_drawing_doc(status="completed")
                        # Ensure the just-committed stroke is materialized in cache
                        # before we apply erase marks. Otherwise a draw->erase
                        # transition can leave the new stroke out of the cached canvas.
                        self._strokes.render(frame0.shape)
                    erase_radius = self._strokes.current_radius
                    self._strokes.erase_near(tip0[0], tip0[1], radius=erase_radius, z=z)
                    self._insert_erase_doc(tip0[0], tip0[1], erase_radius, z)
                    self._was_drawing = False
                    self._was_erasing = True
                    self._canvas_version += 1
                elif drawing and tip0:
                    if not self._was_drawing:
                        active_color = PALETTE[self._color_idx]
                        active_radius = int(max(1, round(float(self._strokes.current_radius))))
                        active_min_radius = self._strokes.stroke_min_radius(active_radius)
                        self._strokes.begin(
                            color=active_color,
                            max_radius=active_radius,
                            min_radius=active_min_radius,
                            theta=self._theta,
                        )
                        self._start_drawing_doc(color=active_color, brush_radius=active_radius)
                    active = self._strokes._active
                    before_len = len(active.pts) if active is not None else 0
                    self._strokes.add_point(tip0[0], tip0[1], z)
                    active = self._strokes._active
                    if active is not None and len(active.pts) > before_len:
                        sx, sy, sz = active.pts[-1]
                        point_radius = int(max(1, round(float(getattr(active, "max_radius", self._strokes.current_radius)))))
                        self._insert_point_doc(sx, sy, sz, color=active.color, brush_radius=point_radius)
                        self._canvas_version += 1
                    self._was_drawing = True
                elif resizing and tip0 and tip8 and tip12:
                    PINCH_MAX = 0.25
                    raw_dist = math.hypot(tip12.x - tip8.x, tip12.y - tip8.y)
                    pinch_norm = 1.0 - min(raw_dist / PINCH_MAX, 1.0)
                    _smooth_pinch = 0.85 * _smooth_pinch + 0.15 * pinch_norm
                    now_mono = time.monotonic()
                    next_radius = self._strokes._radius(_smooth_pinch)
                    # Arm lock only on threshold crossing (prevents lock restart each frame).
                    if (
                        next_radius <= _small_radius_lock_px
                        and self._strokes.current_radius > _small_radius_lock_px
                    ):
                        _small_radius_lock_until = now_mono + _small_radius_lock_sec
                    if now_mono < _small_radius_lock_until:
                        next_radius = min(next_radius, _small_radius_lock_px)
                    self._strokes.current_radius = int(max(1, next_radius))
                elif clearing:
                    self._start_clear_action()
                else:
                    if self._was_drawing:
                        self._strokes.end()
                        self._finish_drawing_doc(status="completed")
                    self._was_drawing = False

                # person_z: shoulder midpoint in world coords when pose available,
                # otherwise triangulated body depth from stereo.
                if ENABLE_3D_PERSON:
                    if shoulder_world_z is not None:
                        person_z = shoulder_world_z
                    else:
                        if body_pos3d:
                            _last_person_z = body_pos3d[2]
                        person_z = _last_person_z
                else:
                    person_z = None

                def alpha_composite(base, overlay, alpha):
                    """Blend overlay onto base using float32 H×W alpha (0=base, 1=overlay)."""
                    a = alpha[:, :, np.newaxis]
                    return np.clip(
                        overlay.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a),
                        0, 255,
                    ).astype(np.uint8)

                def apply_strokes(base, canvas):
                    s_m = canvas.any(axis=2)
                    base[s_m] = canvas[s_m]
                    return base

                if ENABLE_3D_PERSON and person_mask is not None and person_z is not None:
                    # Depth-sorted composite: bg strokes → person (alpha) → fg strokes
                    bg_strokes, fg_strokes = self._strokes.render_layered(
                        frame0.shape,
                        person_z,
                        project=live_project,
                    )

                    frame0 = apply_strokes(frame0, bg_strokes)
                    frame0 = alpha_composite(frame0, raw_frame, person_mask)
                    frame0 = apply_strokes(frame0, fg_strokes)
                elif ENABLE_3D_PERSON and person_mask is not None:
                    stroke_canvas = self._strokes.render(frame0.shape, project=live_project)
                    frame0 = apply_strokes(frame0, stroke_canvas)
                    frame0 = alpha_composite(frame0, raw_frame, person_mask)
                else:
                    stroke_canvas = self._strokes.render(frame0.shape, project=live_project)
                    frame0 = apply_strokes(frame0, stroke_canvas)

                if erasing and tip0:
                    cv2.circle(frame0, tip0, self._strokes.current_radius, (0, 0, 255), 2, cv2.LINE_AA)

                self._swipe_events = list(swipe_events)
                self._tracking = {
                    "cam0": self.cam0,
                    "cam1": None if single_cam else self.cam1,
                    "tip0": list(tip0) if tip0 else None,
                    "tip1": list(tip1) if tip1 else None,
                    "pos3d": [round(v, 2) for v in pos3d] if pos3d else None,
                    "gesture": gesture,
                    "gesture_conf": round(conf, 3) if conf is not None else None,
                    "gesture_model_loaded": bool(gesture_clf),
                    "gesture_source": gesture_source,
                    "fps": round(_fps, 1),
                    "brush_radius": self._strokes.current_radius,
                    "live_yaw": round(live_yaw_deg, 1),
                }
                self.output_frame = frame0
                _push = {
                    "color_idx": self._color_idx,
                    "swipe_events": list(self._swipe_events),
                    "tracking": dict(self._tracking),
                    "canvas_version": self._canvas_version,
                }

            self._push_state(_push)

            _fps_count += 1
            _now = time.monotonic()
            if _now - _fps_t0 >= 0.5:
                _fps = _fps_count / (_now - _fps_t0)
                _fps_count = 0
                _fps_t0 = _now

            swipe_events = [(lbl, ci, f - 1) for lbl, ci, f in swipe_events if f > 1]
            _prev_action = action
