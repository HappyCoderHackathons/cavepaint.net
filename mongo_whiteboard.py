"""Mongo-backed whiteboard stroke replay for stream rendering."""

from __future__ import annotations

import threading
import time
from datetime import datetime

from pymongo.errors import PyMongoError

from stroke import Stroke


_COLOR_NAME_TO_BGR = {
    "white": (255, 255, 255),
    "black": (20, 20, 20),
    "red": (0, 0, 220),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 220),
    "green": (0, 200, 60),
    "cyan": (200, 200, 0),
    "blue": (220, 60, 0),
    "magenta": (200, 0, 200),
    "purple": (130, 0, 180),
    "brown": (19, 69, 139),
}


class MongoWhiteboardReplay:
    """Incrementally replays point + erase events from MongoDB for one session."""

    def __init__(
        self,
        points_col,
        erases_col,
        session_id: str | None = None,
        refresh_interval_ms: int = 250,
    ):
        self._points_col = points_col
        self._erases_col = erases_col
        self._session_id = session_id
        self._refresh_interval_s = max(0.01, float(refresh_interval_ms) / 5000.0)

        self._lock = threading.Lock()
        self._next_refresh_at = 0.0

        self._initialized = False
        self._last_point_id = None
        self._last_erase_id = None

        self._active_by_drawing: dict[str, Stroke] = {}
        self._timeline_ops: list[dict] = []
        self._state_snapshot: list[dict] = []

    @staticmethod
    def _coerce_time(value, oid_value) -> float:
        if isinstance(value, datetime):
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(datetime.fromisoformat(value).timestamp())
            except ValueError:
                pass
        if oid_value is not None and hasattr(oid_value, "generation_time"):
            try:
                return float(oid_value.generation_time.timestamp())
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def _coerce_color(value) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(int(max(0, min(255, c))) for c in value)
        if isinstance(value, str):
            return _COLOR_NAME_TO_BGR.get(value.lower(), _COLOR_NAME_TO_BGR["black"])
        return _COLOR_NAME_TO_BGR["black"]

    @staticmethod
    def _coerce_radius(value, default: int = 10) -> int:
        try:
            return max(1, int(round(float(value))))
        except (TypeError, ValueError):
            return max(1, int(default))

    @staticmethod
    def _append_point(stroke: Stroke, x: float, y: float, z: float, event_time: float) -> None:
        before = len(stroke.pts)
        stroke.add(float(x), float(y), float(z))
        if len(stroke.times) > before:
            stroke.times[-1] = float(event_time)

    @staticmethod
    def _event_key(event: dict):
        # Keep point before erase on equal timestamp to match "draw then erase".
        kind_order = 0 if event["kind"] == "point" else 1
        return (
            float(event["t"]),
            kind_order,
            str(event.get("drawing_key", "")),
            int(event.get("seq", 0)),
            str(event.get("oid", "")),
        )

    def _base_query(self) -> dict:
        return {"sessionId": self._session_id} if self._session_id else {}

    def _fetch_new_docs(self, last_point_id, last_erase_id):
        base = self._base_query()
        point_query = dict(base)
        erase_query = dict(base)
        if last_point_id is not None:
            point_query["_id"] = {"$gt": last_point_id}
        if last_erase_id is not None:
            erase_query["_id"] = {"$gt": last_erase_id}

        point_docs = list(
            self._points_col.find(
                point_query,
                {"_id": 1, "drawingId": 1, "seq": 1, "t": 1, "position": 1, "color": 1, "brushRadius": 1},
            ).sort([("_id", 1)])
        )
        erase_docs = list(
            self._erases_col.find(
                erase_query,
                {"_id": 1, "x": 1, "y": 1, "z": 1, "radius": 1, "t": 1},
            ).sort([("_id", 1)])
        )
        return point_docs, erase_docs

    def _docs_to_events(self, point_docs, erase_docs):
        events = []

        for doc in point_docs:
            pos = doc.get("position") or {}
            if not isinstance(pos, dict):
                continue
            x = pos.get("x")
            y = pos.get("y")
            z = pos.get("z", 0.0)
            if x is None or y is None:
                continue
            oid = doc.get("_id")
            events.append(
                {
                    "kind": "point",
                    "oid": oid,
                    "t": self._coerce_time(doc.get("t"), oid),
                    "drawing_key": str(doc.get("drawingId") or ""),
                    "seq": int(doc.get("seq", 0)),
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "color": self._coerce_color(doc.get("color")),
                    "brush_radius": self._coerce_radius(doc.get("brushRadius"), default=10),
                }
            )

        for doc in erase_docs:
            x = doc.get("x")
            y = doc.get("y")
            if x is None or y is None:
                continue
            oid = doc.get("_id")
            events.append(
                {
                    "kind": "erase",
                    "oid": oid,
                    "t": self._coerce_time(doc.get("t"), oid),
                    "x": float(x),
                    "y": float(y),
                    "z": float(doc.get("z", 0.0)),
                    "radius": float(doc.get("radius", 30.0)),
                }
            )

        events.sort(key=self._event_key)
        return events

    def _flush_active(self):
        if not self._active_by_drawing:
            return
        for stroke in self._active_by_drawing.values():
            if not stroke.empty():
                self._timeline_ops.append({"kind": "stroke", "stroke": stroke})
        self._active_by_drawing = {}

    def _apply_events(self, events):
        for event in events:
            if event["kind"] == "point":
                drawing_key = event["drawing_key"]
                stroke = self._active_by_drawing.get(drawing_key)
                if stroke is None:
                    stroke = Stroke(
                        color=event["color"],
                        max_radius=self._coerce_radius(event.get("brush_radius"), default=10),
                    )
                    self._active_by_drawing[drawing_key] = stroke
                self._append_point(stroke, event["x"], event["y"], event["z"], event["t"])
                continue

            self._flush_active()
            self._timeline_ops.append(
                {
                    "kind": "erase",
                    "x": float(event["x"]),
                    "y": float(event["y"]),
                    "z": float(event.get("z", 0.0)),
                    "radius": float(event["radius"]),
                    "t": float(event["t"]),
                }
            )

    def _make_snapshot(self):
        ops = list(self._timeline_ops)
        for stroke in self._active_by_drawing.values():
            if not stroke.empty():
                ops.append({"kind": "stroke", "stroke": stroke})
        return ops

    def load_state(self) -> list[dict] | None:
        if self._points_col is None or self._erases_col is None:
            return None

        now = time.monotonic()
        with self._lock:
            if self._initialized and now < self._next_refresh_at:
                return self._state_snapshot
            last_point_id = self._last_point_id
            last_erase_id = self._last_erase_id

        try:
            point_docs, erase_docs = self._fetch_new_docs(last_point_id, last_erase_id)
        except PyMongoError as exc:
            print(f"[mongo] whiteboard replay failed: {exc}")
            with self._lock:
                return self._state_snapshot

        events = self._docs_to_events(point_docs, erase_docs)

        with self._lock:
            # First refresh loads the full session query.
            if not self._initialized:
                self._active_by_drawing = {}
                self._timeline_ops = []
            if events:
                self._apply_events(events)
            if point_docs:
                self._last_point_id = point_docs[-1].get("_id")
            if erase_docs:
                self._last_erase_id = erase_docs[-1].get("_id")

            self._state_snapshot = self._make_snapshot()
            self._initialized = True
            self._next_refresh_at = now + self._refresh_interval_s
            return self._state_snapshot

    def load_strokes(self) -> list[Stroke] | None:
        ops = self.load_state()
        if ops is None:
            return None
        return [op["stroke"] for op in ops if op.get("kind") == "stroke"]
