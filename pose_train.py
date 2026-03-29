"""Train a pose classifier on collected body landmark data.

Usage:
    uv run pose_train.py
    uv run pose_train.py --epochs 80
    uv run pose_train.py --dataset pose_dataset/session_20260329_120000
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

POSES = ["idle", "arms_up", "arms_wide", "point_left", "point_right"]
NUM_CLASSES = len(POSES)

NUM_LANDMARKS = 33
INPUT_DIM = NUM_LANDMARKS * 3   # normalized x, y, z per landmark

HIDDEN1, HIDDEN2, HIDDEN3 = 128, 64, 32
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3

MODEL_PATH = Path("pose_model.pth")
META_PATH = Path("pose_meta.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Path to a specific session dir (default: all sessions)")
    return parser.parse_args()


def find_csvs(dataset_arg):
    if dataset_arg:
        return [dataset_arg / "landmarks.csv"]
    csvs = sorted(Path("pose_dataset").glob("session_*/landmarks.csv"))
    if not csvs:
        raise FileNotFoundError("No pose dataset found. Run pose_collect.py first.")
    return csvs


def load_splits(csv_paths):
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    df = df[df["pose"].isin(POSES)].copy()
    p2i = {p: i for i, p in enumerate(POSES)}
    df["label"] = df["pose"].map(p2i)

    norm_cols = [f"norm_{i}_{ax}" for i in range(NUM_LANDMARKS) for ax in ("x", "y", "z")]

    def extract(subset):
        X = subset[norm_cols].values.astype(np.float32)
        y = subset["label"].values.astype(np.int64)
        return X, y

    train = df[df["cycle"].isin([1, 2, 3])]
    val   = df[df["cycle"] == 4]
    test  = df[df["cycle"] == 5]

    return extract(train), extract(val), extract(test)


class PoseMLP(nn.Module):
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
    csv_paths = find_csvs(args.dataset)
    for p in csv_paths:
        print(f"Dataset: {p}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_splits(csv_paths)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = torch.from_numpy(1.0 / np.clip(class_counts / class_counts.sum(), 1e-6, None))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PoseMLP().to(device)
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

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    print("\n--- Test Set ---")
    print(classification_report(labels, preds, target_names=POSES))

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix:")
    print("              " + "  ".join(f"{p[:10]:>10}" for p in POSES))
    for i, row in enumerate(cm):
        print(f"{POSES[i]:>13}  " + "  ".join(f"{v:10d}" for v in row))

    META_PATH.write_text(json.dumps({
        "poses": POSES,
        "input_dim": INPUT_DIM,
        "num_landmarks": NUM_LANDMARKS,
    }))
    print(f"\nMetadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
