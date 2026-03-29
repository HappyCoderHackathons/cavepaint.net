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
        meta = json.loads(GESTURE_META_PATH.read_text())
        self.gestures = meta["gestures"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        model = _GestureMLP(num_classes=len(self.gestures)).to(device)
        model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=device, weights_only=True))
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
