"""MediaPipe hand and pose landmarkers: creation, detection, and frame overlay."""

import cv2
import mediapipe as mp
import numpy as np

from .constants import HAND_MODEL_PATH, INDEX_FINGERTIP, POSE_MODEL_PATH

HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
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


def make_pose_landmarker():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
    )
    return PoseLandmarker.create_from_options(options)


def detect_pose(landmarker, frame, ts_ms):
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def get_segmentation_mask(result, threshold: float = 0.6, blur_ksize: int = 21) -> np.ndarray | None:
    """Return a float32 H×W alpha mask (0=background, 1=person) with soft edges.

    Steps:
      1. Threshold the raw confidence map.
      2. Morphological close to fill holes (e.g. loose clothing, gaps).
      3. Gaussian blur for feathered edges.
    """
    if not result.segmentation_masks:
        return None
    raw = result.segmentation_masks[0].numpy_view().squeeze()  # float32 H×W

    # Binary mask at threshold
    binary = (raw > threshold).astype(np.uint8)

    # Close small holes inside the body silhouette
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Feather edges with Gaussian blur → smooth float alpha
    alpha = cv2.GaussianBlur(closed.astype(np.float32), (blur_ksize, blur_ksize), 0)
    return np.clip(alpha, 0.0, 1.0)


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
