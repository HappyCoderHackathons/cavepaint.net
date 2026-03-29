"""StereoDrawingTracker: captures both cameras, runs hand detection, draws on a shared canvas."""

import asyncio
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import cv2
import numpy as np
import torch
from pymongo.errors import PyMongoError

from mongo_whiteboard import MongoWhiteboardReplay
from stroke import Stroke, StrokeStore
from swipe_detect import SwipeDetector
from triangulate import depth_inches_to_str, triangulate

from .camera import CameraReader, find_cameras, open_camera
from .constants import (
    GESTURE_CONFIDENCE,
    GESTURE_META_PATH,
    GESTURE_MODEL_PATH,
    PALETTE,
    SWIPE_DISPLAY_FRAMES,
    SWIPE_META_PATH,
    SWIPE_MODEL_PATH,
    _PALM_INDICES,
)
from .gesture import GestureClassifier
from .landmarker import detect, draw_hand, make_landmarker
from .mongo import WHITEBOARD_DB_REFRESH_MS, drawings_col, erases_col, points_col
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
        self._sub_lock = threading.Lock()
        self._slots: list[StateSlot] = []
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

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None
            return self.output_frame.copy()

    def get_state(self):
        with self.lock:
            return {
                "color_idx": self._color_idx,
                "swipe_events": list(self._swipe_events),
                "tracking": dict(self._tracking),
            }

    def set_color(self, idx: int):
        with self.lock:
            self._color_idx = idx % len(PALETTE)

    def clear_canvas(self):
        with self.lock:
            self._strokes.clear()

    def undo(self):
        with self.lock:
            self._strokes.undo()

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
                center = self._project_whiteboard_point(cx, cy, cz, yaw_rad, fov_rad, width, height)
                if center is None:
                    continue
                draw_r = self._project_erase_radius(cx, cy, cz, cr, yaw_rad, fov_rad, width, height)
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
    def _project_whiteboard_point(x, y, z, yaw_rad, fov_rad, width, height):
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

    @staticmethod
    def _project_erase_radius(x, y, z, radius, yaw_rad, fov_rad, width, height):
        c0 = StereoDrawingTracker._project_whiteboard_point(x, y, z, yaw_rad, fov_rad, width, height)
        c1 = StereoDrawingTracker._project_whiteboard_point(x + float(radius), y, z, yaw_rad, fov_rad, width, height)
        if c0 is not None and c1 is not None:
            r = int(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))
            return max(1, r)
        return max(1, int(float(radius) * (float(width) / 640.0)))

    # ------------------------------------------------------------------
    # MongoDB helpers
    # ------------------------------------------------------------------

    def _start_drawing_doc(self, color="black"):
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
            "drawingId": self._active_drawing_id, "seq": self._seq, "t": self._utcnow(),
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
            "x": float(x), "y": float(y), "z": float(z),
            "radius": float(radius), "t": self._utcnow(),
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

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    def _process_loop(self):
        cap1 = None
        single_cam = True
        reader0 = None
        reader1 = None
        try:
            cap0 = self._open_cam0()
            cap1, single_cam = self._open_cam1()

            reader0 = CameraReader(cap0)
            reader0.start()
            if single_cam:
                reader1 = reader0
            else:
                reader1 = CameraReader(cap1)
                reader1.start()

            gesture_clf = self._load_gesture_classifier()
            swipe_det, swipe_events = self._load_swipe_detector()

            try:
                with make_landmarker() as lm0, make_landmarker() as lm1:
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        self._run_loop(
                            lm0, lm1, pool,
                            reader0, reader1, single_cam,
                            gesture_clf, swipe_det, swipe_events,
                        )
            except Exception as exc:
                with self.lock:
                    self.output_frame = self._error_frame(f"Stereo tracker error: {exc}")
        finally:
            if self._was_drawing:
                self._strokes.end()
                self._finish_drawing_doc(status="interrupted")
                self._was_drawing = False
            if reader0 is not None:
                reader0.stop()
            if not single_cam and reader1 is not None:
                reader1.stop()
            cap0.release()
            if cap1 is not None:
                cap1.release()

    def _open_cam0(self):
        try:
            return open_camera(self.cam0, self.width, self.height)
        except RuntimeError:
            available = find_cameras()
            if not available:
                with self.lock:
                    self.output_frame = self._error_frame("No cameras found")
                raise
            self.cam0 = available[0]
            return open_camera(self.cam0, self.width, self.height)

    def _open_cam1(self):
        try:
            cap1 = open_camera(self.cam1, self.width, self.height)
            return cap1, False
        except RuntimeError:
            available = [i for i in find_cameras() if i != self.cam0]
            if available:
                self.cam1 = available[0]
                return open_camera(self.cam1, self.width, self.height), False
            return None, True

    def _load_gesture_classifier(self):
        if GESTURE_MODEL_PATH.exists() and GESTURE_META_PATH.exists():
            try:
                return GestureClassifier()
            except Exception:
                pass
        return None

    def _load_swipe_detector(self):
        if SWIPE_MODEL_PATH.exists() and SWIPE_META_PATH.exists():
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                return SwipeDetector(SWIPE_MODEL_PATH, SWIPE_META_PATH, device=device), []
            except Exception:
                pass
        return None, []

    def _run_loop(self, lm0, lm1, pool, reader0, reader1, single_cam, gesture_clf, swipe_det, swipe_events):
        _fps = 0.0
        _fps_count = 0
        _fps_t0 = time.monotonic()
        _smooth_pinch = 0.5
        _resize_cooldown = 0

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
                res0 = detect(lm0, frame0, ts)
                res1 = res0
            else:
                f0 = pool.submit(detect, lm0, frame0, ts)
                f1 = pool.submit(detect, lm1, frame1, ts)
                res0, res1 = f0.result(), f1.result()

            tip0 = tip1 = tip8 = tip12 = None
            gesture = None
            if res0.hand_landmarks:
                hand = res0.hand_landmarks[0]
                tip0 = draw_hand(frame0, hand, w, h)
                tip8 = hand[8]
                tip12 = hand[12]
                if gesture_clf:
                    gesture, conf = gesture_clf.classify(hand)
                    if conf < GESTURE_CONFIDENCE:
                        gesture = None
            if res1.hand_landmarks:
                tip1 = draw_hand(frame1, res1.hand_landmarks[0], w, h)

            # Gesture → action flags
            if gesture_clf:
                drawing  = (gesture == "point")
                resizing = (gesture == "peace")
                erasing  = (gesture == "fist")
            else:
                drawing = resizing = erasing = False

            if resizing:
                _resize_cooldown = 18
            elif _resize_cooldown > 0:
                _resize_cooldown -= 1
                drawing = False

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
                    if label == "swipe_right":
                        self._color_idx = (self._color_idx + 1) % len(PALETTE)
                    elif label == "swipe_left":
                        self._color_idx = (self._color_idx - 1) % len(PALETTE)
                    if label in ("swipe_left", "swipe_right"):
                        swipe_events.append((label, self._color_idx, SWIPE_DISPLAY_FRAMES))

            pos3d = triangulate(tip0, tip1) if (not single_cam and tip0 and tip1) else None

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

            combined = cv2.hconcat([frame0, frame1])

            if not single_cam:
                depth_str = depth_inches_to_str(pos3d)
                depth_color = (0, 255, 0) if pos3d else (100, 100, 100)
                cv2.putText(combined, depth_str, (20, combined.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, depth_color, 2, cv2.LINE_AA)

            z = pos3d[2] if pos3d else 0.0

            with self.lock:
                if erasing and tip0:
                    if self._was_drawing:
                        self._strokes.end()
                        self._finish_drawing_doc(status="completed")
                    erase_radius = self._strokes.current_radius
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
                elif resizing and tip0 and tip8 and tip12:
                    PINCH_MAX = 0.25
                    raw_dist = math.hypot(tip12.x - tip8.x, tip12.y - tip8.y)
                    pinch_norm = 1.0 - min(raw_dist / PINCH_MAX, 1.0)
                    _smooth_pinch = 0.85 * _smooth_pinch + 0.15 * pinch_norm
                    self._strokes.current_radius = self._strokes._radius(_smooth_pinch)
                else:
                    if self._was_drawing:
                        self._strokes.end()
                        self._finish_drawing_doc(status="completed")
                    self._was_drawing = False

                stroke_canvas = self._strokes.render(combined.shape)
                mask = stroke_canvas.any(axis=2)
                combined[mask] = stroke_canvas[mask]

                if erasing and tip0:
                    cv2.circle(combined, tip0, self._strokes.current_radius, (0, 0, 255), 2, cv2.LINE_AA)

                self._swipe_events = list(swipe_events)
                self._tracking = {
                    "cam0": self.cam0,
                    "cam1": None if single_cam else self.cam1,
                    "tip0": list(tip0) if tip0 else None,
                    "tip1": list(tip1) if tip1 else None,
                    "pos3d": [round(v, 2) for v in pos3d] if pos3d else None,
                    "gesture": gesture,
                    "fps": round(_fps, 1),
                    "brush_radius": self._strokes.current_radius,
                }
                self.output_frame = combined
                _push = {
                    "color_idx": self._color_idx,
                    "swipe_events": list(self._swipe_events),
                    "tracking": dict(self._tracking),
                }

            self._push_state(_push)

            _fps_count += 1
            _now = time.monotonic()
            if _now - _fps_t0 >= 0.5:
                _fps = _fps_count / (_now - _fps_t0)
                _fps_count = 0
                _fps_t0 = _now

            swipe_events = [(lbl, ci, f - 1) for lbl, ci, f in swipe_events if f > 1]
