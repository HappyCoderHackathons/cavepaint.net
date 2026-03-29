"""Stereo drawing package.

Public surface (matches the old stereo_drawing.py exports used by app.py):
    PALETTE, StereoDrawingTracker, find_cameras
"""

from .camera import find_cameras
from .constants import PALETTE
from .tracker import StereoDrawingTracker

__all__ = ["PALETTE", "StereoDrawingTracker", "find_cameras"]
