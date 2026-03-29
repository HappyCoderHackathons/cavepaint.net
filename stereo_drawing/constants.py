"""Shared constants for the stereo drawing system."""

from pathlib import Path

HAND_MODEL_PATH    = Path(__file__).parent.parent / "hand_landmarker.task"
GESTURE_MODEL_PATH = Path(__file__).parent.parent / "gesture_model.pth"
GESTURE_META_PATH  = Path(__file__).parent.parent / "gesture_meta.json"
SWIPE_MODEL_PATH   = Path(__file__).parent.parent / "swipe_model.pth"
SWIPE_META_PATH    = Path(__file__).parent.parent / "swipe_meta.json"

INDEX_FINGERTIP    = 8
GESTURE_CONFIDENCE = 0.5
SWIPE_DISPLAY_FRAMES = 35  # how long a swipe label stays on screen (~1 s)

# 10-color palette (BGR) cycled by swipe left/right when palm is open
PALETTE = [
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

_PALM_INDICES     = [0, 5, 9, 13, 17]
_FINGERTIP_INDICES = [4, 8, 12, 16, 20]
_INPUT_DIM        = 63 + 15  # 21 landmarks × 3 coords + 15 distance features
