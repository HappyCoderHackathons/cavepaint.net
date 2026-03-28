"""Train a gesture classifier on collected landmark data.

Usage:
    uv run train.py
    uv run train.py --epochs 80
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

GESTURES = ["open_hand", "point", "fist"]
NUM_CLASSES = len(GESTURES)

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
LANDMARK_DIM = 63       # 21 landmarks * 3
EXTRA_FEATURES = 15     # 10 pairwise fingertip distances + 5 fingertip-to-wrist
INPUT_DIM = LANDMARK_DIM + EXTRA_FEATURES

HIDDEN1, HIDDEN2, HIDDEN3 = 128, 64, 32
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

MODEL_PATH = Path("gesture_model.pth")
META_PATH = Path("gesture_meta.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to a specific session dir (default: latest)")
    return parser.parse_args()


def find_csv(dataset_arg):
    if dataset_arg:
        return dataset_arg / "landmarks.csv"
    sessions = sorted(Path("dataset").glob("session_*"))
    if not sessions:
        raise FileNotFoundError("No dataset found. Run collect.py first.")
    return sessions[-1] / "landmarks.csv"


def compute_features(raw_coords: np.ndarray) -> np.ndarray:
    n = raw_coords.shape[0]
    coords = raw_coords.reshape(n, 21, 3)

    fingertips = coords[:, FINGERTIP_INDICES, :]
    scale = np.linalg.norm(fingertips, axis=2).max(axis=1, keepdims=True)
    scale = np.clip(scale, 1e-6, None)
    coords = coords / scale[:, :, np.newaxis]

    ft = coords[:, FINGERTIP_INDICES, :]
    pair_dists = []
    for i in range(5):
        for j in range(i + 1, 5):
            pair_dists.append(np.linalg.norm(ft[:, i] - ft[:, j], axis=1))
    pair_dists = np.stack(pair_dists, axis=1)

    tip_dists = np.linalg.norm(ft, axis=2)

    return np.concatenate([coords.reshape(n, LANDMARK_DIM), pair_dists, tip_dists], axis=1).astype(np.float32)


def load_splits(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["gesture"].isin(GESTURES)].copy()
    g2i = {g: i for i, g in enumerate(GESTURES)}
    df["label"] = df["gesture"].map(g2i)

    norm_cols = [f"norm_{i}_{ax}" for i in range(21) for ax in ("x", "y", "z")]

    def extract(subset):
        X = compute_features(subset[norm_cols].values.astype(np.float32))
        y = subset["label"].values.astype(np.int64)
        return X, y

    train = df[df["cycle"].isin([1, 2, 3])]
    val   = df[df["cycle"] == 4]
    test  = df[df["cycle"] == 5]

    return extract(train), extract(val), extract(test)


class GestureMLP(nn.Module):
    def __init__(self):
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
            nn.Linear(HIDDEN3, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def main():
    args = parse_args()
    csv_path = find_csv(args.dataset)
    print(f"Dataset: {csv_path}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(csv_path)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = torch.from_numpy(1.0 / np.clip(class_counts / class_counts.sum(), 1e-6, None))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GestureMLP().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    train_loader = make_loader(X_train, y_train)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                val_correct += (out.argmax(1) == yb).sum().item()
                val_total += len(xb)

        val_acc = val_correct / val_total
        scheduler.step(val_loss / val_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train={correct/total:.4f}  val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.4f}  ->  {MODEL_PATH}")

    # Test evaluation
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    print("\n--- Test Set ---")
    print(classification_report(labels, preds, target_names=GESTURES))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix:")
    print("             " + "  ".join(f"{g[:7]:>7}" for g in GESTURES))
    for i, row in enumerate(cm):
        print(f"{GESTURES[i]:>12}  " + "  ".join(f"{v:7d}" for v in row))

    META_PATH.write_text(json.dumps({
        "gestures": GESTURES,
        "input_dim": INPUT_DIM,
        "fingertip_indices": FINGERTIP_INDICES,
    }))
    print(f"\nMetadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
