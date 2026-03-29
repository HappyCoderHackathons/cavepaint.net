"""Historical session loading for gallery and replay."""
from __future__ import annotations

import threading
from datetime import datetime

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


class SessionStore:
    """Loads session data from MongoDB and builds renderable ops at any point in time.

    Timelines are cached in memory after first load so scrubbing is fast.
    """

    def __init__(self, drawings_col, points_col, erases_col):
        self._drawings = drawings_col
        self._points = points_col
        self._erases = erases_col
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """Return sessions sorted newest-first."""
        pipeline = [
            {"$group": {
                "_id": "$sessionId",
                "started_at": {"$min": "$createdAt"},
                "ended_at": {"$max": {"$ifNull": ["$endedAt", "$updatedAt"]}},
                "stroke_count": {"$sum": "$pointCount"},
                "drawing_count": {"$sum": 1},
            }},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"started_at": -1}},
            {"$limit": limit},
        ]
        out = []
        for r in self._drawings.aggregate(pipeline):
            sid = r.get("_id")
            if not sid:
                continue
            started = r.get("started_at")
            ended = r.get("ended_at")
            out.append({
                "session_id": sid,
                "started_at": self._iso(started),
                "ended_at": self._iso(ended),
                "duration_s": self._diff_s(started, ended),
                "stroke_count": int(r.get("stroke_count") or 0),
                "drawing_count": int(r.get("drawing_count") or 0),
            })
        return out

    def get_session_info(self, session_id: str) -> dict | None:
        tl = self._get_timeline(session_id)
        if tl is None:
            return None
        return {
            "session_id": session_id,
            "started_at": self._iso(tl.get("started_at")),
            "duration_s": tl.get("duration_s", 0.0),
            "stroke_count": tl.get("stroke_count", 0),
        }

    def build_ops_at(self, session_id: str, t_offset_s: float) -> list[dict]:
        """Ops list (strokes + erases) representing the board at t_offset_s seconds in."""
        tl = self._get_timeline(session_id)
        if tl is None:
            return []
        events = [e for e in tl["events"] if e["t_offset"] <= t_offset_s]
        return self._events_to_ops(events)

    # ------------------------------------------------------------------
    # Timeline loading
    # ------------------------------------------------------------------

    def _get_timeline(self, session_id: str) -> dict | None:
        with self._lock:
            if session_id in self._cache:
                return self._cache[session_id]

        point_docs = list(self._points.find(
            {"sessionId": session_id},
            {"_id": 1, "drawingId": 1, "seq": 1, "t": 1, "position": 1, "color": 1, "brushRadius": 1},
        ).sort([("t", 1), ("seq", 1)]))

        erase_docs = list(self._erases.find(
            {"sessionId": session_id},
            {"_id": 1, "x": 1, "y": 1, "z": 1, "radius": 1, "t": 1},
        ).sort([("t", 1)]))

        if not point_docs and not erase_docs:
            return None

        events: list[dict] = []

        for doc in point_docs:
            pos = doc.get("position") or {}
            x, y = pos.get("x"), pos.get("y")
            if x is None or y is None:
                continue
            events.append({
                "kind": "point",
                "t": self._ts(doc.get("t"), doc.get("_id")),
                "drawing_key": str(doc.get("drawingId") or ""),
                "seq": int(doc.get("seq") or 0),
                "x": float(x),
                "y": float(y),
                "z": float(pos.get("z") or 0.0),
                "color": self._coerce_color(doc.get("color")),
                "brush_radius": max(1, int(round(float(doc.get("brushRadius") or 10)))),
            })

        for doc in erase_docs:
            x, y = doc.get("x"), doc.get("y")
            if x is None or y is None:
                continue
            events.append({
                "kind": "erase",
                "t": self._ts(doc.get("t"), doc.get("_id")),
                "x": float(x),
                "y": float(y),
                "z": float(doc.get("z") or 0.0),
                "radius": float(doc.get("radius") or 30.0),
            })

        events.sort(key=lambda e: (e["t"], 0 if e["kind"] == "point" else 1, e.get("seq", 0)))

        start_t = events[0]["t"] if events else 0.0
        for e in events:
            e["t_offset"] = e["t"] - start_t

        duration_s = events[-1]["t_offset"] if events else 0.0

        agg = list(self._drawings.aggregate([
            {"$match": {"sessionId": session_id}},
            {"$group": {
                "_id": None,
                "started_at": {"$min": "$createdAt"},
                "stroke_count": {"$sum": "$pointCount"},
            }},
        ]))
        started_at = agg[0].get("started_at") if agg else None
        stroke_count = int(agg[0].get("stroke_count") or 0) if agg else len(point_docs)

        tl = {
            "session_id": session_id,
            "started_at": started_at,
            "duration_s": duration_s,
            "stroke_count": stroke_count,
            "events": events,
        }
        with self._lock:
            self._cache[session_id] = tl
        return tl

    # ------------------------------------------------------------------
    # Ops builder
    # ------------------------------------------------------------------

    @staticmethod
    def _events_to_ops(events: list[dict]) -> list[dict]:
        active: dict[str, Stroke] = {}
        ops: list[dict] = []

        for event in events:
            if event["kind"] == "point":
                key = event["drawing_key"]
                stroke = active.get(key)
                if stroke is None:
                    max_r = event["brush_radius"]
                    stroke = Stroke(
                        color=event["color"],
                        max_radius=max_r,
                        min_radius=SessionStore._min_r(max_r),
                    )
                    active[key] = stroke
                stroke.add(event["x"], event["y"], event["z"])
            elif event["kind"] == "erase":
                for s in active.values():
                    if not s.empty():
                        ops.append({"kind": "stroke", "stroke": s})
                active = {}
                ops.append({
                    "kind": "erase",
                    "x": event["x"], "y": event["y"], "z": event["z"],
                    "radius": event["radius"],
                })

        for s in active.values():
            if not s.empty():
                ops.append({"kind": "stroke", "stroke": s})

        return ops

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _min_r(max_radius: int) -> int:
        max_r = max(1, int(max_radius))
        target = int(round(max_r * 0.50))
        lo, hi = 3, max(1, max_r - 1)
        if hi < lo:
            return max(1, min(max_r, lo))
        return max(lo, min(hi, target))

    @staticmethod
    def _ts(value, oid=None) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass
        if oid is not None and hasattr(oid, "generation_time"):
            try:
                return oid.generation_time.timestamp()
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _coerce_color(value) -> tuple[int, int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(int(max(0, min(255, c))) for c in value)
        if isinstance(value, str):
            return _COLOR_NAME_TO_BGR.get(value.lower(), _COLOR_NAME_TO_BGR["black"])
        return _COLOR_NAME_TO_BGR["black"]

    @staticmethod
    def _diff_s(a, b) -> float | None:
        if a is None or b is None:
            return None
        try:
            if isinstance(a, datetime) and isinstance(b, datetime):
                return (b - a).total_seconds()
            return float(b) - float(a)
        except Exception:
            return None

    @staticmethod
    def _iso(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
