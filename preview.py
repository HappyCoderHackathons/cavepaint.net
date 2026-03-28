"""Live gesture preview — shows what gesture you're holding up.

Usage:
    uv run preview.py
"""

import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

HAND_MODEL_PATH = Path("hand_landmarker.task")
GESTURE_MODEL_PATH = Path("gesture_model.pth")
META_PATH = Path("gesture_meta.json")

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
LANDMARK_DIM = 63
EXTRA_FEATURES = 15
INPUT_DIM = LANDMARK_DIM + EXTRA_FEATURES
HIDDEN1, HIDDEN2, HIDDEN3 = 128, 64, 32
CONFIDENCE_THRESHOLD = 0.5

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode


class GestureMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.BatchNorm1d(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.BatchNorm1d(HIDDEN2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN2, HIDDEN3),
            nn.BatchNorm1d(HIDDEN3),
            nn.ReLU(),
            nn.Linear(HIDDEN3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def compute_features(landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)
    coords -= coords[0].copy()

    fingertips = coords[FINGERTIP_INDICES]
    scale = max(np.linalg.norm(fingertips, axis=1).max(), 1e-6)
    coords /= scale

    ft = coords[FINGERTIP_INDICES]
    pair_dists = [np.linalg.norm(ft[i] - ft[j]) for i in range(5) for j in range(i + 1, 5)]
    tip_dists = np.linalg.norm(ft, axis=1)

    return np.concatenate([coords.flatten(), pair_dists, tip_dists]).astype(np.float32)


def draw_hand(frame, landmarks, w, h):
    conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for conn in conns:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def main():
    if not GESTURE_MODEL_PATH.exists():
        print("No trained model found. Run collect.py then train.py first.")
        return

    meta = json.loads(META_PATH.read_text())
    gestures = meta["gestures"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureMLP(num_classes=len(gestures)).to(device)
    model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    frame_ts_ms = 0

    print("Gesture preview running. Press 'q' to quit.")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            label = ""
            color = (200, 200, 200)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_hand(frame, landmarks, w, h)

                features = compute_features(landmarks)
                with torch.no_grad():
                    x = torch.from_numpy(features).unsqueeze(0).to(device)
                    probs = torch.softmax(model(x), dim=1)
                    conf, idx = probs.max(1)
                    conf = conf.item()

                if conf >= CONFIDENCE_THRESHOLD:
                    label = f"{gestures[idx.item()]}  {conf:.0%}"
                    color = (0, 255, 0)
                else:
                    label = f"?  {conf:.0%}"
                    color = (0, 165, 255)

            if label:
                cv2.putText(frame, label, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)

            cv2.imshow("Gesture Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
