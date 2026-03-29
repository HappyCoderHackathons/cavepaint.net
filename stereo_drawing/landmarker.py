"""MediaPipe hand landmarker: creation, detection, and frame overlay."""

import cv2
import mediapipe as mp

from .constants import HAND_MODEL_PATH, INDEX_FINGERTIP

HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions           = mp.tasks.BaseOptions
RunningMode           = mp.tasks.vision.RunningMode


def make_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def detect(landmarker, frame, ts_ms):
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def draw_hand(frame, landmarks, w, h):
    conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for conn in conns:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    fp = pts[INDEX_FINGERTIP]
    cv2.circle(frame, fp, 10, (0, 255, 255), -1)
    return pts[INDEX_FINGERTIP]
