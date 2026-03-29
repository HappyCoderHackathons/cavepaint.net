"""Shared constants for the stereo drawing system."""

import os
from pathlib import Path

HAND_MODEL_PATH    = Path(__file__).parent.parent / "hand_landmarker.task"
POSE_MODEL_PATH    = Path(__file__).parent.parent / "pose_landmarker.task"
GESTURE_MODEL_PATH = Path(__file__).parent.parent / "gesture_model.pth"
GESTURE_META_PATH  = Path(__file__).parent.parent / "gesture_meta.json"
SWIPE_MODEL_PATH   = Path(__file__).parent.parent / "swipe_model.pth"
SWIPE_META_PATH    = Path(__file__).parent.parent / "swipe_meta.json"

INDEX_FINGERTIP    = 8
GESTURE_CONFIDENCE = 0.5
SWIPE_DISPLAY_FRAMES = 35  # how long a swipe label stays on screen (~1 s)

# When false, disable "3D person" silhouette alpha compositing and
# behind/in-front splitting. Strokes still project in 3D.
ENABLE_3D_PERSON = os.getenv("ENABLE_3D_PERSON", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)

# 10-color palette (BGR) cycled by swipe left/right when palm is open
# PALETTE = [
#     ( 20,  20,  20),  # near-black
#     (  0,   0, 220),  # red
#     (  0, 140, 255),  # orange
#     (  0, 220, 220),  # yellow
#     (  0, 200,  60),  # green
#     (200, 200,   0),  # cyan
#     (220,  60,   0),  # blue
#     (200,   0, 200),  # magenta
#     (130,   0, 180),  # purple
#     ( 19,  69, 139),  # brown
# ]

#CAVEMAN PALETTE
PALETTE = [
    (23, 26, 27), #Charcoal Black
    (41, 58, 91), #Burnt Umber
    (23, 160, 212), #Ochre Yellow
    (46, 59, 142), #Iron Oxide Red
    (227, 203, 175), #Pale Cave Blue
    (60, 94, 182), #Terracotta
    (123, 169, 201), #Sandstone
    (199, 221, 232), #Bone White
    (104, 116, 126), #Ash Gray
    (79, 107, 95) #Moss Patina
]

_PALM_INDICES     = [0, 5, 9, 13, 17]
_FINGERTIP_INDICES = [4, 8, 12, 16, 20]
_INPUT_DIM        = 63 + 15  # 21 landmarks × 3 coords + 15 distance features
