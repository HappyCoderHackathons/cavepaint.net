"""Gesture classifier: MLP model, feature extraction, and inference."""

import json

import numpy as np
import torch
import torch.nn as nn

from .constants import GESTURE_META_PATH, GESTURE_MODEL_PATH, _FINGERTIP_INDICES, _INPUT_DIM


class _GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32),   nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords -= coords[0].copy()
    ft = coords[_FINGERTIP_INDICES]
    scale = max(np.linalg.norm(ft, axis=1).max(), 1e-6)
    coords /= scale
    ft = coords[_FINGERTIP_INDICES]
    pair_dists = [np.linalg.norm(ft[i] - ft[j]) for i in range(5) for j in range(i + 1, 5)]
    tip_dists = np.linalg.norm(ft, axis=1).tolist()
    return np.concatenate([coords.flatten(), pair_dists, tip_dists]).astype(np.float32)


class GestureClassifier:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        meta = json.loads(GESTURE_META_PATH.read_text())
        gestures_from_meta = meta["gestures"]

        # Some deployments may have a gesture_meta.json that doesn't match the
        # currently present gesture_model.pth class-count (e.g. meta includes
        # "gun" but the checkpoint is still 4-class). If that happens, we
        # infer the checkpoint's output dimension and truncate gestures so
        # load_state_dict succeeds and point drawing keeps working.
        state_dict = torch.load(
            GESTURE_MODEL_PATH,
            map_location=device,
            weights_only=True,
        )

        num_classes_ckpt = None
        for _, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == 32:
                # Last Linear in _GestureMLP is Linear(32, num_classes), so its
                # weight has shape (num_classes, 32).
                num_classes_ckpt = int(v.shape[0])
                break

        if num_classes_ckpt is not None and num_classes_ckpt != len(gestures_from_meta):
            self.gestures = gestures_from_meta[:num_classes_ckpt]
        else:
            self.gestures = gestures_from_meta

        model = _GestureMLP(num_classes=len(self.gestures)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def classify(self, landmarks) -> tuple[str, float]:
        """Return (gesture_name, confidence)."""
        features = compute_features(landmarks)
        with torch.no_grad():
            x = torch.from_numpy(features).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(x), dim=1)
            conf, idx = probs.max(1)
        return self.gestures[idx.item()], conf.item()


def classify_rule_gesture(landmarks) -> tuple[str | None, float]:
    """Lightweight geometric fallback when ML confidence is too low.

    Returns (gesture_name, confidence_like_score in [0,1]).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    if coords.shape != (21, 3):
        return None, 0.0

    wrist = coords[0]

    def is_extended(tip_idx: int, pip_idx: int, ratio: float) -> bool:
        tip_d = float(np.linalg.norm(coords[tip_idx] - wrist))
        pip_d = float(np.linalg.norm(coords[pip_idx] - wrist))
        return tip_d > max(1e-6, pip_d * ratio)

    thumb = is_extended(4, 3, 1.05)
    index = is_extended(8, 6, 1.12)
    middle = is_extended(12, 10, 1.12)
    ring = is_extended(16, 14, 1.12)
    pinky = is_extended(20, 18, 1.12)

    def score(parts: list[bool]) -> float:
        return float(sum(1 for p in parts if p)) / float(len(parts))

    # Keep ordering aligned with interactive use priorities.
    # point: index only
    point_s = score([index, not middle, not ring, not pinky])
    if point_s >= 0.85 and index:
        return "point", point_s

    # peace: index + middle
    peace_s = score([index, middle, not ring, not pinky])
    if peace_s >= 0.85 and index and middle:
        return "peace", peace_s

    # gun: thumb + index
    gun_s = score([thumb, index, not middle, not ring, not pinky])
    if gun_s >= 0.90 and thumb and index:
        return "gun", gun_s

    # fist: no fingers extended
    fist_s = score([not index, not middle, not ring, not pinky])
    if fist_s >= 0.90:
        return "fist", fist_s

    # open hand: most fingers extended
    ext_count = int(thumb) + int(index) + int(middle) + int(ring) + int(pinky)
    if ext_count >= 4:
        return "open_hand", ext_count / 5.0

    return None, 0.0
