import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from load_data import (
    build_metadata,
    build_cache,
    load_audio,
    augment_audio,
    extract_logmel_3ch_from_waveform,
)
from model_cnn import SER_CNN_Attention


# =========================
# Config
# =========================
DATA_PATH = "data/crema"
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 90
LR = 3e-4
WEIGHT_DECAY = 1e-4

SR = 16000
N_MELS = 128
MAX_LEN = 360
CROP_LEN = 300

NUM_CLASSES = 6
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]


# =========================
# Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Feature-space SpecAugment
# =========================
def spec_augment_3ch(x, p=0.4, F_mask=8, T_mask=12):
    if random.random() > p:
        return x

    x = x.clone()
    _, n_mels, n_steps = x.shape

    if n_mels > F_mask:
        f0 = random.randint(0, n_mels - F_mask)
        x[:, f0:f0 + F_mask, :] = 0

    if n_steps > T_mask:
        t0 = random.randint(0, n_steps - T_mask)
        x[:, :, t0:t0 + T_mask] = 0

    return x


# =========================
# Dataset
# =========================
class TrainWaveformDataset(Dataset):
    def __init__(self, paths, labels, mean, std):
        self.paths = paths
        self.labels = labels
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y = load_audio(self.paths[idx], sr=SR)
        y = augment_audio(y, sr=SR)

        x = extract_logmel_3ch_from_waveform(
            y,
            sr=SR,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            max_len=MAX_LEN,
        )

        x = (x - self.mean) / (self.std + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)

        x = spec_augment_3ch(x, p=0.4, F_mask=8, T_mask=12)

        return x, torch.tensor(self.labels[idx], dtype=torch.long)


class FeatureDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / (self.std + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# =========================
# Multi-crop eval
# =========================
def get_eval_crops(xb, crop_len):
    """
    xb: (B, 3, 128, T)
    returns 3 crops: left, center, right
    """
    T = xb.shape[-1]

    if T <= crop_len:
        if T < crop_len:
            xb = F.pad(xb, (0, crop_len - T))
        return [xb]

    left = xb[..., :crop_len]
    center_start = (T - crop_len) // 2
    center = xb[..., center_start:center_start + crop_len]
    right = xb[..., -crop_len:]

    return [left, center, right]


# =========================
# Eval
# =========================
def evaluate_single(model, loader, criterion, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_true, all_preds),
        f1_score(all_true, all_preds, average="macro"),
        np.array(all_true),
        np.array(all_preds),
    )


def evaluate_multicrop(model, loader, criterion, device, crop_len):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            crops = get_eval_crops(xb, crop_len)
            logits_sum = 0.0
            loss_sum = 0.0

            for crop in crops:
                logits = model(crop)
                logits_sum = logits_sum + logits
                loss_sum = loss_sum + criterion(logits, yb)

            logits_avg = logits_sum / len(crops)
            loss_avg = loss_sum / len(crops)
            total_loss += loss_avg.item()

            preds = torch.argmax(logits_avg, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.cpu().numpy())

    return (
        total_loss / len(loader),
        accuracy_score(all_true, all_preds),
        f1_score(all_true, all_preds, average="macro"),
        np.array(all_true),
        np.array(all_preds),
    )


# =========================
# Plot functions
# =========================
def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_spectrograms(X, y, save_path):
    plt.figure(figsize=(10, 6))

    for i in range(NUM_CLASSES):
        idx = np.where(y == i)[0][0]
        plt.subplot(2, 3, i + 1)
        plt.imshow(X[idx, 0], aspect="auto", origin="lower")
        plt.title(CLASS_NAMES[i])
        plt.xlabel("Time")
        plt.ylabel("Mel Bin")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # clean cached features for analysis / val / test / train-eval
    X_all, y_all, _, _ = build_cache(DATA_PATH, cache_dir=CACHE_DIR, max_len=MAX_LEN)
    print("Loaded X:", X_all.shape)
    print("Loaded y:", y_all.shape)

    # metadata with waveform paths for train-time on-the-fly augmentation
    paths_all, labels_all, _, _ = build_metadata(DATA_PATH)

    idx_all = np.arange(len(labels_all))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, labels_all, test_size=0.2, stratify=labels_all, random_state=SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    X_train_clean = X_all[idx_train]
    X_val = X_all[idx_val]
    X_test = X_all[idx_test]

    train_paths = paths_all[idx_train]
    train_labels = labels_all[idx_train]

    print("Train:", X_train_clean.shape, train_labels.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    mean = np.mean(X_train_clean, axis=(0, 2, 3)).reshape(3, 1, 1)
    std = np.std(X_train_clean, axis=(0, 2, 3)).reshape(3, 1, 1)

    train_ds = TrainWaveformDataset(train_paths, train_labels, mean, std)
    train_eval_ds = FeatureDataset(X_train_clean, train_labels, mean, std)
    val_ds = FeatureDataset(X_val, y_val, mean, std)
    test_ds = FeatureDataset(X_test, y_test, mean, std)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = SER_CNN_Attention(NUM_CLASSES).to(device)

    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=4
    )

    best_f1 = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        _, train_acc, train_f1, _, _ = evaluate_single(model, train_eval_loader, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_multicrop(model, val_loader, criterion, device, CROP_LEN)

        scheduler.step(val_f1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))

    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best.pt"), map_location=device, weights_only=True)
    )

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_multicrop(
        model, test_loader, criterion, device, CROP_LEN
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest Acc: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

    with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro F1: {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_f1s, label="Train Macro F1")
    plt.plot(epochs, val_f1s, label="Validation Macro F1")
    plt.legend()
    plt.title("F1 Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_curve.png"), dpi=150)
    plt.close()

    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_sample_spectrograms(X_train_clean, train_labels, os.path.join(OUTPUT_DIR, "sample_spectrograms.png"))


if __name__ == "__main__":
    main()