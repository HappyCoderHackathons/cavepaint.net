"""Train a 1D CNN on swipe motion sequences.

Usage:
    uv run swipe_train.py swipe_dataset/session_*/sequences.csv
    uv run swipe_train.py swipe_dataset/session_20260328_*/sequences.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

MOTIONS = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "none"]
NUM_CLASSES = len(MOTIONS)

WINDOW_SIZE        = 30
FEATURES_PER_FRAME = 3   # px, py, scale
EPOCHS             = 80
BATCH_SIZE         = 64
LR                 = 1e-3

MODEL_PATH = Path("swipe_model.pth")
META_PATH  = Path("swipe_meta.json")


class SwipeCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 3, 30) -> (B, 32, 28) -> pool -> (B, 32, 14)
            nn.Conv1d(FEATURES_PER_FRAME, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # (B, 32, 14) -> (B, 64, 12) -> pool -> (B, 64, 6)
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


def load_csv_files(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "motion" not in df.columns and "gesture" in df.columns:
            df = df.rename(columns={"gesture": "motion"})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_features(df):
    feat_cols = [f"f{i}_{ax}" for i in range(WINDOW_SIZE) for ax in ("px", "py", "scale")]
    # Fall back to khe-project column naming if needed
    if feat_cols[0] not in df.columns:
        feat_cols = [f"frame_{i}_{ax}" for i in range(WINDOW_SIZE)
                     for ax in ("px", "py", "scale")]
    X = df[feat_cols].values.astype(np.float32)
    label_map = {m: i for i, m in enumerate(MOTIONS)}
    y = df["motion"].map(label_map).values.astype(np.int64)
    cycles = df["cycle"].values
    return X, y, cycles


def normalize(X):
    """Per-sequence: center x/y, normalize scale to mean=1."""
    X = X.reshape(-1, WINDOW_SIZE, FEATURES_PER_FRAME).copy()
    X[:, :, 0] -= X[:, :, 0].mean(axis=1, keepdims=True)
    X[:, :, 1] -= X[:, :, 1].mean(axis=1, keepdims=True)
    mean_sc = X[:, :, 2].mean(axis=1, keepdims=True)
    X[:, :, 2] /= np.clip(mean_sc, 1e-6, None)
    return X.transpose(0, 2, 1).astype(np.float32)  # (N, 3, 30)


def split_by_cycle(X, y, cycles):
    """Adaptive split: last cycle=test, second-to-last=val, rest=train.
    Falls back to random splits when there are too few cycles."""
    max_c = cycles.max()

    if max_c == 1:
        n = len(y)
        idx = np.random.permutation(n)
        t1, t2 = int(n * 0.8), int(n * 0.9)
        train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
        train[idx[:t1]] = True; val[idx[t1:t2]] = True; test[idx[t2:]] = True
    elif max_c == 2:
        c1_idx = np.where(cycles == 1)[0]
        np.random.shuffle(c1_idx)
        split = int(len(c1_idx) * 0.8)
        train = np.zeros(len(y), bool); val = np.zeros(len(y), bool)
        train[c1_idx[:split]] = True; val[c1_idx[split:]] = True
        test = cycles == 2
    else:
        test  = cycles == max_c
        val   = cycles == max_c - 1
        train = cycles < max_c - 1

    return (X[train], y[train]), (X[val], y[val]), (X[test], y[test])


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train(csv_paths):
    print(f"Loading {len(csv_paths)} CSV file(s)...")
    df = load_csv_files(csv_paths)
    X_raw, y, cycles = extract_features(df)

    unknown = np.isnan(y)
    if unknown.any():
        bad = df["motion"][unknown].unique()
        print(f"Warning: skipping {unknown.sum()} rows with unknown labels: {bad}")
        X_raw = X_raw[~unknown]
        y = y[~unknown].astype(np.int64)
        cycles = cycles[~unknown]

    X = normalize(X_raw)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_by_cycle(X, y, cycles)

    print(f"Train: {len(X_tr)}  Val: {len(X_va)}  Test: {len(X_te)}")
    for i, m in enumerate(MOTIONS):
        print(f"  {m:>12s}: train={( y_tr==i).sum():4d}  "
              f"val={(y_va==i).sum():4d}  test={(y_te==i).sum():4d}")

    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / np.clip(counts, 1, None)
    weights = weights / weights.sum() * NUM_CLASSES

    train_loader = make_loader(X_tr, y_tr)
    val_loader   = make_loader(X_va, y_va, shuffle=False)
    test_loader  = make_loader(X_te, y_te, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = SwipeCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += xb.size(0)

        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                loss = criterion(out, yb)
                va_loss    += loss.item() * xb.size(0)
                va_correct += (out.argmax(1) == yb).sum().item()
                va_total   += xb.size(0)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / max(va_total, 1)
        scheduler.step(va_loss / max(va_total, 1))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  train={tr_acc:.4f}  val={va_acc:.4f}  "
                  f"loss={tr_loss/tr_total:.4f}/{va_loss/max(va_total,1):.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {MODEL_PATH}")

    # Test
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    preds  = np.array(preds)
    labels = np.array(labels)

    print("\n--- Test Set ---")
    print(classification_report(labels, preds, labels=list(range(NUM_CLASSES)),
                                 target_names=MOTIONS, zero_division=0))

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("Confusion matrix:")
    print("             " + "".join(f"{m[:5]:>7s}" for m in MOTIONS))
    for i, row in enumerate(cm):
        print(f"{MOTIONS[i]:>12s}" + "".join(f"{v:7d}" for v in row))

    META_PATH.write_text(json.dumps({
        "motions": MOTIONS,
        "num_classes": NUM_CLASSES,
        "window_size": WINDOW_SIZE,
        "features_per_frame": FEATURES_PER_FRAME,
    }))
    print(f"\nMetadata saved to {META_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python swipe_train.py <sequences.csv> [more.csv ...]")
        print("Example: python swipe_train.py swipe_dataset/session_*/sequences.csv")
        sys.exit(1)

    import glob
    paths = []
    for pattern in sys.argv[1:]:
        matched = glob.glob(pattern)
        paths += matched if matched else [pattern]

    train([Path(p) for p in paths])
