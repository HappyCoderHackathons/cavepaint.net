"""Live swipe motion detection using the trained CNN.

Usage:
    uv run swipe_detect.py
"""

import json
import math
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = Path("swipe_model.pth")
META_PATH  = Path("swipe_meta.json")
HAND_MODEL_PATH = Path("hand_landmarker.task")

PALM_INDICES = [0, 5, 9, 13, 17]
CONFIDENCE_THRESHOLD = 0.7
COOLDOWN_FRAMES      = 20
DISPLAY_FRAMES       = 30

COLORS = {
    "swipe_left":  (255, 200,   0),
    "swipe_right": (  0, 200, 255),
    "swipe_up":    (  0, 255, 200),
    "swipe_down":  (200, 100, 255),
}


class SwipeCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))


class SwipeDetector:
    def __init__(self, model_path, meta_path, device=None):
        self.device = device or torch.device("cpu")
        meta = json.loads(Path(meta_path).read_text())
        self.motions     = meta["motions"]
        self.window_size = meta["window_size"]
        self.none_idx    = self.motions.index("none")

        self.model = SwipeCNN(len(self.motions)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        self.buf      = deque(maxlen=self.window_size)
        self.cooldown = 0

    def update(self, px, py, scale):
        """Returns (label, confidence) or None."""
        self.buf.append((px, py, scale))
        if self.cooldown > 0:
            self.cooldown -= 1
            return None
        if len(self.buf) < self.window_size:
            return None

        x = np.array(self.buf, dtype=np.float32)  # (30, 3)
        x[:, 0] -= x[:, 0].mean()
        x[:, 1] -= x[:, 1].mean()
        mean_sc = x[:, 2].mean()
        if mean_sc > 1e-6:
            x[:, 2] /= mean_sc
        x = torch.from_numpy(x.T).unsqueeze(0).to(self.device)  # (1, 3, 30)

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
            conf, pred = probs.max(1)
            conf, pred = conf.item(), pred.item()

        if pred == self.none_idx or conf < CONFIDENCE_THRESHOLD:
            return None

        self.cooldown = COOLDOWN_FRAMES
        self.buf.clear()
        return self.motions[pred], conf


def main():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("No trained model found. Run swipe_collect.py then swipe_train.py first.")
        return

    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    BaseOptions           = mp.tasks.BaseOptions
    RunningMode           = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    detector = SwipeDetector(MODEL_PATH, META_PATH)
    print("Swipe detection running. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    frame_ts_ms   = 0
    active_events = []  # [(label, frames_remaining)]

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_ts_ms)

            if result.hand_landmarks:
                lms   = result.hand_landmarks[0]
                conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                pts   = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(lms)}
                for conn in conns:
                    cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 0), 2)
                for pt in pts.values():
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

                xs = [lms[i].x for i in PALM_INDICES]
                ys = [lms[i].y for i in PALM_INDICES]
                px = sum(xs) / len(xs)
                py = sum(ys) / len(ys)
                sc = math.hypot(lms[12].x - lms[0].x, lms[12].y - lms[0].y)

                hit = detector.update(px, py, sc)
                if hit:
                    label, conf = hit
                    active_events.append((label, DISPLAY_FRAMES))
                    print(f"{label}  ({conf:.0%})")

            # Draw active events
            y_off = 70
            for label, _ in active_events:
                color = COLORS.get(label, (255, 255, 255))
                cv2.putText(frame, label.upper().replace("_", " "),
                            (w // 2 - 120, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
                y_off += 50

            active_events = [(n, f - 1) for n, f in active_events if f > 1]

            cv2.putText(frame, "Swipe Detection", (w - 190, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)

            cv2.imshow("Swipe Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
