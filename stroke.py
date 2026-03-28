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
    _smooth_alpha: float = 0.45   # EMA blend (0=no smoothing, 1=no follow)
    _min_dist: float = 4.0        # skip points closer than this (pixels)

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
        """Draw this stroke onto *canvas* in-place.

        project(x, y, z) -> (px, py) | None  — omit for pixel-space strokes.
        """
        if self.empty():
            return

        px = self._project(project)
        if len(px) < 2:
            return

        speeds = self._speeds()
        max_spd = max(speeds) if speeds else 1.0

        smooth_pts, radii = self._catmull_rom(px, speeds, max_spd)
        self._draw(canvas, smooth_pts, radii)

    def _project(self, project):
        if project:
            out = [project(x, y, z) for x, y, z in self.pts]
            return [p for p in out if p is not None]
        return [(int(x), int(y)) for x, y, _ in self.pts]

    def _speeds(self) -> list:
        """Speed in coordinate units per second between consecutive points."""
        speeds = [0.0]
        for i in range(1, len(self.pts)):
            x0, y0, z0 = self.pts[i - 1]
            x1, y1, z1 = self.pts[i]
            dt = max(self.times[i] - self.times[i - 1], 1e-4)
            dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
            speeds.append(dist / dt)
        return speeds

    def _radius(self, speed: float, max_spd: float) -> int:
        norm = min(speed / max(max_spd, 1e-6), 1.0)
        return max(self.min_radius, int(self.max_radius - norm * (self.max_radius - self.min_radius)))

    def _catmull_rom(self, px: list, speeds: list, max_spd: float, steps: int = 20):
        """Catmull-Rom interpolation through 2D pixel points.

        Returns (smooth_pts, radii) with len = (len(px)-1)*steps + 1.
        """
        # Phantom endpoints so the spline passes through first and last points
        pts = [px[0]] + list(px) + [px[-1]]
        spd = [speeds[0]] + list(speeds) + [speeds[-1]]

        out_pts, out_radii = [], []
        for i in range(1, len(pts) - 2):
            p0 = np.array(pts[i - 1], dtype=float)
            p1 = np.array(pts[i],     dtype=float)
            p2 = np.array(pts[i + 1], dtype=float)
            p3 = np.array(pts[i + 2], dtype=float)
            s0, s1 = spd[i], spd[i + 1]
            for j in range(steps):
                t = j / steps
                q = 0.5 * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
                )
                s = s0 + (s1 - s0) * t
                out_pts.append(tuple(q.astype(int)))
                out_radii.append(self._radius(s, max_spd))

        out_pts.append(px[-1])
        out_radii.append(self._radius(speeds[-1], max_spd))
        return out_pts, out_radii

    def _draw(self, canvas: np.ndarray, pts: list, radii: list):
        n = len(pts)
        taper = min(10, n // 4)

        for i in range(n - 1):
            r = radii[i]
            # taper at stroke start and end
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
            cv2.circle(canvas, pts[-1], radii[-1], self.color, -1, cv2.LINE_AA)


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

    # --- stroke lifecycle --------------------------------------------------

    def begin(self, **kwargs):
        """Start a new stroke (ends any active stroke first)."""
        self._commit_active()
        self._active = Stroke(**kwargs)

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

    def erase_near(self, x: float, y: float, radius: float = 40.0):
        """Remove any completed stroke that has a point within *radius* pixels of (x, y)."""
        r2 = radius * radius
        before = len(self._completed)
        self._completed = [
            s for s in self._completed
            if not any((px - x) ** 2 + (py - y) ** 2 <= r2 for px, py, *_ in s.pts)
        ]
        if len(self._completed) != before:
            self._dirty = True

    def clear(self):
        self._completed.clear()
        self._active = None
        self._cache = None
        self._dirty = False

    # --- rendering ---------------------------------------------------------

    def render(self, shape: tuple, project=None) -> np.ndarray:
        """Return a canvas (same shape) with all strokes drawn.

        project(x, y, z) -> (px, py) | None  for 3D -> 2D projection.
        """
        if self._cache is None or self._cache.shape != shape or self._dirty:
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
            self._dirty = True
        self._active = None
