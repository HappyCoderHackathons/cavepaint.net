"""Stereo triangulation for fingertip 3D position.

Corrects for asymmetric toe-in by de-rotating each camera's pixel coordinates
individually before computing disparity.

Camera assignment (matches stereo_preview.py):
    left  = cap0 = cam index 2
    right = cap1 = cam index 1

Coordinate system (origin at bar center):
    X: horizontal, positive right
    Y: vertical, positive down
    Z: depth, positive away from cameras

Units: inches.
"""

import json
import math
from pathlib import Path

FRAME_W = 640
FRAME_H = 480
CX = FRAME_W / 2
CY = FRAME_H / 2
BASELINE_INCHES = 12.0
Z_MIN_INCHES = -5.0
Z_MAX_INCHES = 220.0
MIN_DISPARITY_PX = 2.0

_cal_path = Path(__file__).parent / "calibration.json"
if _cal_path.exists():
    _cal = json.loads(_cal_path.read_text())
    # left=cap0=cam_index_2, right=cap1=cam_index_1
    FOCAL_LEFT      = _cal.get("cam2", {}).get("focal_length_px", 650.0)
    FOCAL_RIGHT     = _cal.get("cam1", {}).get("focal_length_px", 650.0)
    FOCAL_LENGTH_PX = (FOCAL_LEFT + FOCAL_RIGHT) / 2
    ANGLE_LEFT_DEG  = _cal.get("cam2", {}).get("toein_angle_deg", 0.0)
    ANGLE_RIGHT_DEG = _cal.get("cam1", {}).get("toein_angle_deg", 0.0)
    print(f"[triangulate] left(cam2)  focal={FOCAL_LEFT:.1f}px  angle={ANGLE_LEFT_DEG:+.2f}°")
    print(f"[triangulate] right(cam1) focal={FOCAL_RIGHT:.1f}px  angle={ANGLE_RIGHT_DEG:+.2f}°")
else:
    FOCAL_LEFT = FOCAL_RIGHT = FOCAL_LENGTH_PX = 650.0
    ANGLE_LEFT_DEG = ANGLE_RIGHT_DEG = 0.0
    print("[triangulate] no calibration.json — using defaults")


def _derotate_x(x: float, focal: float, angle_deg: float) -> float:
    """Convert pixel x in a toed-in camera to the equivalent in a parallel camera."""
    return focal * math.tan(math.atan((x - CX) / focal) - math.radians(angle_deg)) + CX


def triangulate(
    left_pt: tuple[float, float],
    right_pt: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Return (X, Y, Z) in inches, or None if unreliable."""
    x1 = _derotate_x(left_pt[0],  FOCAL_LEFT,  ANGLE_LEFT_DEG)
    x2 = _derotate_x(right_pt[0], FOCAL_RIGHT, ANGLE_RIGHT_DEG)

    disparity = x1 - x2

    print(f"[tri] raw=({left_pt[0]:.0f},{right_pt[0]:.0f}) rect=({x1:.1f},{x2:.1f}) disp={disparity:.1f}")

    if abs(disparity) < MIN_DISPARITY_PX:
        print(f"[tri] rejected: disparity too small")
        return None

    Z = (FOCAL_LENGTH_PX * BASELINE_INCHES) / disparity

    if not (Z_MIN_INCHES < Z < Z_MAX_INCHES):
        print(f"[tri] rejected: Z={Z:.1f} out of range")
        return None

    X = (left_pt[0] - CX) * Z / FOCAL_LENGTH_PX
    Y = (left_pt[1] - CY) * Z / FOCAL_LENGTH_PX

    return X, Y, Z


def depth_inches_to_str(pos: tuple[float, float, float] | None) -> str:
    if pos is None:
        return "depth: --"
    x, y, z = pos
    return f"X:{x:+.1f}\"  Y:{y:+.1f}\"  Z:{z:.1f}\""
