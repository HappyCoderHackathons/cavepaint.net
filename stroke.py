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

    theta: float = 0.0              # camera mount angle (degrees) when stroke was drawn

    # Input smoothing
    _smooth_alpha: float = 0.25   # EMA blend (0=no smoothing, 1=no follow)
    _min_dist: float = 2.0        # skip points closer than this (pixels)
    _dynamic_strength: float = 0.7  # 0..1, lower = less width variation

    
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
        projected = self._project(project)
        if len(projected) < 2:
            return
        px = [p for p, _ in projected]
        ts = [t for _, t in projected]
        base_radii = self._dynamic_radii(px, ts)
        smooth_pts, smooth_radii = self._catmull_rom_with_radii(px, base_radii)
        self._draw(canvas, smooth_pts, smooth_radii)


    def _project(self, project):
        if project:
            out = []
            for (x, y, z), t in zip(self.pts, self.times):
                p = project(x, y, z)
                if p is not None:
                    out.append((p, float(t)))
            return out
        return [((int(x), int(y)), float(t)) for (x, y, _), t in zip(self.pts, self.times)]

    def _dynamic_radii(self, px: list, ts: list) -> list:
        n = len(px)
        if n == 0:
            return []
        if n == 1:
            return [float(self.max_radius)]

        seg_speeds = []
        for i in range(1, n):
            x0, y0 = px[i - 1]
            x1, y1 = px[i]
            dist = float(np.hypot(x1 - x0, y1 - y0))
            dt = max(float(ts[i] - ts[i - 1]), 1e-3)
            seg_speeds.append(dist / dt)

        spd_ref = float(np.percentile(seg_speeds, 90)) if seg_speeds else 1.0
        spd_ref = max(spd_ref, 1e-3)

        raw_radii = []
        for i in range(n):
            if i == 0:
                spd = seg_speeds[0]
            elif i == n - 1:
                spd = seg_speeds[-1]
            else:
                spd = 0.5 * (seg_speeds[i - 1] + seg_speeds[i])
            norm = min(max(spd / spd_ref, 0.0), 1.0)
            strength = min(max(float(self._dynamic_strength), 0.0), 1.0)
            eff_norm = norm * strength
            r = float(self.max_radius - eff_norm * (self.max_radius - self.min_radius))
            raw_radii.append(max(float(self.min_radius), min(float(self.max_radius), r)))

        # Radius smoothing keeps thickness transitions organic.
        smooth_radii = []
        alpha = 0.35
        for r in raw_radii:
            if smooth_radii:
                smooth_radii.append(alpha * smooth_radii[-1] + (1.0 - alpha) * r)
            else:
                smooth_radii.append(r)
        return smooth_radii

    def _catmull_rom_with_radii(self, px: list, radii: list, steps: int = 10):
        if len(px) < 2:
            return list(px), list(radii)

        pts = [px[0]] + list(px) + [px[-1]]
        rs = [radii[0]] + list(radii) + [radii[-1]]

        out_pts = []
        out_radii = []
        for i in range(1, len(pts) - 2):
            p0 = np.array(pts[i - 1], dtype=float)
            p1 = np.array(pts[i],     dtype=float)
            p2 = np.array(pts[i + 1], dtype=float)
            p3 = np.array(pts[i + 2], dtype=float)
            r1 = float(rs[i])
            r2 = float(rs[i + 1])
            for j in range(steps):
                t = j / steps
                q = 0.5 * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
                )
                out_pts.append(tuple(q.astype(int)))
                out_radii.append((1.0 - t) * r1 + t * r2)

        out_pts.append((int(round(px[-1][0])), int(round(px[-1][1]))))
        out_radii.append(float(radii[-1]))
        return out_pts, out_radii

    def _draw(self, canvas: np.ndarray, pts: list, radii):
        n = len(pts)
        if n == 0:
            return

        if isinstance(radii, (int, float)):
            radii_seq = [float(radii)] * n
        else:
            radii_seq = [float(r) for r in radii]
            if len(radii_seq) < n:
                pad = radii_seq[-1] if radii_seq else float(self.max_radius)
                radii_seq.extend([pad] * (n - len(radii_seq)))
            elif len(radii_seq) > n:
                radii_seq = radii_seq[:n]

        taper = min(10, n // 4)

        for i in range(n - 1):
            r = radii_seq[i]
            if taper > 0:
                if i < taper:
                    r = max(self.min_radius, r * (i + 1) / taper)
                elif i >= n - 1 - taper:
                    r = max(self.min_radius, r * (n - 1 - i) / taper)
            r = int(max(self.min_radius, min(self.max_radius, round(float(r)))))


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
            end_r = int(max(self.min_radius, min(self.max_radius, round(float(radii_seq[-1])))))
            end_pt = (int(round(float(pts[-1][0]))), int(round(float(pts[-1][1]))))
            cv2.circle(canvas, end_pt, end_r, self.color, -1, cv2.LINE_AA)


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
        self._erase_canvas: np.ndarray | None = None
        self._erase_ops: list[tuple[float, float, float, float]] = []
        self._dirty = False
        self._pixel_edited = False
        self.min_radius = 5
        self.max_radius = 60
        self.current_radius = 10
        # decrease to allow more dynamic lines
        self.dynamic_min_ratio = 0.50

    # --- stroke lifecycle --------------------------------------------------

    def begin(self, **kwargs):
        """Start a new stroke (ends any active stroke first)."""
        self._commit_active()
        self._active = Stroke(**kwargs)

    def stroke_min_radius(self, max_radius: int) -> int:
        max_r = max(1, int(max_radius))
        target = int(round(max_r * float(self.dynamic_min_ratio)))
        lo = max(1, int(self.min_radius))
        hi = max(1, max_r - 1)
        if hi < lo:
            return max(1, min(max_r, lo))
        return max(lo, min(hi, target))
    
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

    def erase_near(self, x: float, y: float, radius: float = 40.0, z: float = 0.0):
        pt = (int(x), int(y))
        r  = int(radius)
        # Record op so render_layered can replay it without needing _cache
        self._erase_ops.append((float(x), float(y), float(radius), float(z)))
        if self._cache is not None:
            cv2.circle(self._cache, pt, r, (0, 0, 0), -1, cv2.LINE_AA)
            self._pixel_edited = True
            if self._erase_canvas is None:
                self._erase_canvas = np.full(self._cache.shape, 255, dtype=np.uint8)
            cv2.circle(self._erase_canvas, pt, r, (0, 0, 0), -1, cv2.LINE_AA)


    def clear(self):
        self._completed.clear()
        self._active = None
        self._cache = None
        self._erase_canvas = None
        self._erase_ops.clear()
        self._dirty = False
        self._pixel_edited = False

    # --- rendering ---------------------------------------------------------

    def render_layered(self, shape: tuple, person_z: float, project=None):
        """Return (behind_canvas, infront_canvas) splitting strokes by Z vs person_z.

        Strokes whose mean Z > person_z are "behind" the person (drawn first);
        strokes whose mean Z <= person_z are "in front" (drawn after).
        """
        behind_strokes, infront_strokes = [], []
        for s in self._completed:
            if s.pts:
                avg_z = sum(p[2] for p in s.pts) / len(s.pts)
                (infront_strokes if avg_z > person_z else behind_strokes).append(s)
            else:
                infront_strokes.append(s)

        active_layer = None
        if self._active and not self._active.empty():
            avg_z = sum(p[2] for p in self._active.pts) / len(self._active.pts)
            active_layer = "infront" if avg_z > person_z else "behind"

        behind = np.zeros(shape, dtype=np.uint8)
        for s in behind_strokes:
            s.render(behind, project)
        if active_layer == "behind":
            self._active.render(behind, project)

        infront = np.zeros(shape, dtype=np.uint8)
        for s in infront_strokes:
            s.render(infront, project)
        if active_layer == "infront":
            self._active.render(infront, project)

        # Replay erase ops onto both layers.
        self._apply_erase_ops(behind, project=project)
        self._apply_erase_ops(infront, project=project)

        return behind, infront

    def render(self, shape: tuple, project=None) -> np.ndarray:
        """Return a canvas (same shape) with all strokes drawn.

        project(x, y, z) -> (px, py) | None  for 3D -> 2D projection.
        """
        if project is not None:
            # Projection depends on camera/view parameters; bypass 2D cache.
            canvas = np.zeros(shape, dtype=np.uint8)
            for s in self._completed:
                s.render(canvas, project)
            if self._active and not self._active.empty():
                self._active.render(canvas, project)
            self._apply_erase_ops(canvas, project=project)
            return canvas

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

    def _apply_erase_ops(self, canvas: np.ndarray, project=None):
        for op in self._erase_ops:
            if len(op) == 4:
                ex, ey, er, ez = op
            else:
                ex, ey, er = op
                ez = 0.0

            if project is None:
                center = (int(ex), int(ey))
                draw_r = int(er)
            else:
                c0 = project(ex, ey, ez)
                if c0 is None:
                    continue
                c1 = project(ex + float(er), ey, ez)
                if c1 is None:
                    draw_r = int(max(1, round(float(er))))
                else:
                    draw_r = int(max(1, round(np.hypot(c1[0] - c0[0], c1[1] - c0[1]))))
                center = (int(round(c0[0])), int(round(c0[1])))

            cv2.circle(canvas, center, int(max(1, draw_r)), (0, 0, 0), -1, cv2.LINE_AA)
