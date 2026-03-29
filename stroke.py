"""Stroke data model and rendering.

A Stroke is a chain of 3D points captured over time. Rendering projects
them to 2D and draws smooth Catmull-Rom curves with speed-based width
and tapered ends — ink-like behaviour.

StrokeStore manages active + completed strokes and caches the completed
canvas so only the live stroke needs to be redrawn each frame.
"""

import time as _time
from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stroke
# ---------------------------------------------------------------------------

@dataclass
class Stroke:
    pts: list = field(default_factory=list)    # list of (x, y, z) — world or pixel coords
    times: list = field(default_factory=list)  # float timestamp per point
    color: tuple = (255, 80, 20)               # BGR
    max_radius: int = 10
    min_radius: int = 2

    # Input smoothing
    _smooth_alpha: float = 0.25   # EMA blend (0=no smoothing, 1=no follow)
    _min_dist: float = 2.0        # skip points closer than this (pixels)

    
    # --- mutation ----------------------------------------------------------

    def add(self, x: float, y: float, z: float = 0.0):
        if self.pts:
            px, py, pz = self.pts[-1]
            # minimum distance filter
            if (x - px) ** 2 + (y - py) ** 2 < self._min_dist ** 2:
                return
            # exponential moving average smoothing
            a = self._smooth_alpha
            x = px + (1 - a) * (x - px)
            y = py + (1 - a) * (y - py)
            z = pz + (1 - a) * (z - pz)
        self.pts.append((x, y, z))
        self.times.append(_time.monotonic())

    def empty(self) -> bool:
        return len(self.pts) < 2

    # --- rendering ---------------------------------------------------------

    def render(self, canvas: np.ndarray, project=None):
        if self.empty():
            return
        px = self._project(project)
        if len(px) < 2:
            return
        smooth_pts = self._catmull_rom(px)
        self._draw(canvas, smooth_pts, self.max_radius)


    def _project(self, project):
        if project:
            out = [project(x, y, z) for x, y, z in self.pts]
            return [p for p in out if p is not None]
        return [(int(x), int(y)) for x, y, _ in self.pts]

    # def _speeds(self) -> list:
    #     """Speed in coordinate units per second between consecutive points."""
    #     speeds = [0.0]
    #     for i in range(1, len(self.pts)):
    #         x0, y0, z0 = self.pts[i - 1]
    #         x1, y1, z1 = self.pts[i]
    #         dt = max(self.times[i] - self.times[i - 1], 1e-4)
    #         dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    #         speeds.append(dist / dt)
    #     return speeds

    # def _radius(self, speed: float, max_spd: float) -> int:
    #     norm = min(speed / max(max_spd, 1e-6), 1.0)
    #     return max(self.min_radius, int(self.max_radius - norm * (self.max_radius - self.min_radius)))

    # def _radius(self, dist: float) -> int:
    #     dist = max(0.0, min(1.0, dist))
    #     return max(self.min_radius, int(self.max_radius - dist * (self.max_radius - self.min_radius)))


    def _catmull_rom(self, px: list, steps: int = 10):
        pts = [px[0]] + list(px) + [px[-1]]

        out_pts = []
        for i in range(1, len(pts) - 2):
            p0 = np.array(pts[i - 1], dtype=float)
            p1 = np.array(pts[i],     dtype=float)
            p2 = np.array(pts[i + 1], dtype=float)
            p3 = np.array(pts[i + 2], dtype=float)
            for j in range(steps):
                t = j / steps
                q = 0.5 * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
                )
                out_pts.append(tuple(q.astype(int)))

        out_pts.append(px[-1])
        return out_pts

    def _draw(self, canvas: np.ndarray, pts: list, radii: int):
        n = len(pts)
        taper = min(10, n // 4)

        for i in range(n - 1):
            r = radii
            if taper > 0:
                if i < taper:
                    r = max(self.min_radius, r * (i + 1) // taper)
                elif i >= n - 1 - taper:
                    r = max(self.min_radius, r * (n - 1 - i) // taper)


            p1 = np.array(pts[i],     dtype=float)
            p2 = np.array(pts[i + 1], dtype=float)
            d = p2 - p1
            length = float(np.linalg.norm(d))

            if length < 1e-6:
                cv2.circle(canvas, tuple(p1.astype(int)), r, self.color, -1, cv2.LINE_AA)
                continue

            perp = np.array([-d[1], d[0]]) / length
            quad = np.array([
                p1 + perp * r, p1 - perp * r,
                p2 - perp * r, p2 + perp * r,
            ], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(canvas, [quad], self.color)
            cv2.circle(canvas, tuple(p1.astype(int)), r, self.color, -1, cv2.LINE_AA)

        if pts:
            cv2.circle(canvas, pts[-1], radii, self.color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# StrokeStore
# ---------------------------------------------------------------------------

class StrokeStore:
    """Manages all strokes and renders them to a canvas.

    Completed strokes are cached so only the active stroke is redrawn
    each frame, keeping per-frame work O(active stroke length).
    """

    def __init__(self):
        self._completed: list[Stroke] = []
        self._active: Stroke | None = None
        self._cache: np.ndarray | None = None
        self._dirty = False
        self._pixel_edited = False
        self.min_radius = 5
        self.max_radius = 60
        self.current_radius = 10

    # --- stroke lifecycle --------------------------------------------------

    def begin(self, **kwargs):
        """Start a new stroke (ends any active stroke first)."""
        self._commit_active()
        self._active = Stroke(**kwargs)
    
    def _radius(self, dist: float) -> int:
        dist = max(0.0, min(1.0, dist))
        return max(self.min_radius, int(self.max_radius - dist * (self.max_radius - self.min_radius)))

    def add_point(self, x: float, y: float, z: float = 0.0):
        """Append a point to the active stroke, starting one if needed."""
        if self._active is None:
            self._active = Stroke()
        self._active.add(x, y, z)

    def end(self):
        """Finish the active stroke."""
        self._commit_active()

    def undo(self):
        """Remove the most recently completed stroke."""
        if self._completed:
            self._completed.pop()
            self._dirty = True

    # def erase_near(self, x: float, y: float, radius: float = 40.0):
    #     """Remove any completed stroke that has a point within *radius* pixels of (x, y)."""
    #     r2 = radius * radius
    #     before = len(self._completed)
    #     new_completed = []
    #     for s in self._completed:
    # # Split stroke at erased points rather than reconnecting across gaps
    #         current_pts, current_times = [], []
    #         for p, t in zip(s.pts, s.times):
    #             if (p[0] - x) ** 2 + (p[1] - y) ** 2 > r2:
    #                 current_pts.append(p)
    #                 current_times.append(t)
    #             else:
    #                 if len(current_pts) >= 2:
    #                     new_s = Stroke(color=s.color, max_radius=s.max_radius, min_radius=s.min_radius)
    #                     new_s.pts = current_pts
    #                     new_s.times = current_times
    #                     new_completed.append(new_s)
    #                 current_pts, current_times = [], []
    #         if len(current_pts) >= 2:
    #             new_s = Stroke(color=s.color, max_radius=s.max_radius, min_radius=s.min_radius)
    #             new_s.pts = current_pts
    #             new_s.times = current_times
    #             new_completed.append(new_s)
    #     self._completed = new_completed


    #     if len(self._completed) != before:
    #         self._dirty = True

    def erase_near(self, x: float, y: float, radius: float = 40.0):
        if self._cache is None:
            return
        cv2.circle(self._cache, (int(x), int(y)), int(radius), (0, 0, 0), -1, cv2.LINE_AA)
        self._pixel_edited = True # Has been erased over


    def clear(self):
        self._completed.clear()
        self._active = None
        self._cache = None
        self._dirty = False
        self._pixel_edited = False

    # --- rendering ---------------------------------------------------------

    def render(self, shape: tuple, project=None) -> np.ndarray:
        """Return a canvas (same shape) with all strokes drawn.

        project(x, y, z) -> (px, py) | None  for 3D -> 2D projection.
        """
        if self._cache is None or self._cache.shape != shape or (self._dirty and not self._pixel_edited):
            self._cache = np.zeros(shape, dtype=np.uint8)
            for s in self._completed:
                s.render(self._cache, project)
            self._dirty = False

        if self._active and not self._active.empty():
            canvas = self._cache.copy()
            self._active.render(canvas, project)
            return canvas

        return self._cache.copy()

    @property
    def has_content(self) -> bool:
        return bool(self._completed) or (self._active and not self._active.empty())

    # --- internal ----------------------------------------------------------

    def _commit_active(self):
        if self._active and not self._active.empty():
            self._completed.append(self._active)
            if self._pixel_edited and self._cache is not None:
                self._active.render(self._cache)
            else:
                self._dirty = True
        self._active = None
