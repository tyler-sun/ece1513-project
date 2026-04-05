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

from load_data import build_cache
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
LR = 4e-4
WEIGHT_DECAY = 1e-4

SR = 16000
N_MELS = 128
MAX_LEN = 360
CROP_LEN = 300

NUM_CLASSES = 6
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

MIXUP_ALPHA = 0.10
MIXUP_PROB = 0.35


# =========================
# Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Augmentations
# =========================
def spec_augment_3ch(x):
    x = x.clone()
    _, n_mels, n_steps = x.shape

    for _ in range(2):
        f = random.randint(0, 10)
        if f > 0:
            f0 = random.randint(0, max(0, n_mels - f))
            x[:, f0:f0 + f, :] = 0

    for _ in range(2):
        t = random.randint(0, 20)
        if t > 0:
            t0 = random.randint(0, max(0, n_steps - t))
            x[:, :, t0:t0 + t] = 0

    return x


def random_time_shift(x):
    shift = random.randint(-10, 10)
    return torch.roll(x, shifts=shift, dims=-1)


def random_time_resample(x, p=0.4, min_scale=0.9, max_scale=1.1):
    """
    Safe time-axis speed perturbation in feature space using interpolation.
    x: (3, 128, T)
    """
    if random.random() > p:
        return x

    c, m, t = x.shape
    scale = random.uniform(min_scale, max_scale)
    new_t = max(16, int(round(t * scale)))

    x_in = x.unsqueeze(0)  # (1,3,128,T)
    x_rs = F.interpolate(
        x_in,
        size=(m, new_t),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if new_t < t:
        pad = t - new_t
        x_rs = F.pad(x_rs, (0, pad))
    elif new_t > t:
        start = random.randint(0, new_t - t)
        x_rs = x_rs[:, :, start:start + t]

    return x_rs


def mixup_data(x, y, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_loss(logits, y_a, y_b, lam, criterion):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# =========================
# Dataset
# =========================
class SERDataset(Dataset):
    def __init__(self, X, y, mean, std, crop_len, train=False):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std
        self.crop_len = crop_len
        self.train = train

    def __len__(self):
        return len(self.X)

    def _crop(self, x):
        T = x.shape[-1]

        if T <= self.crop_len:
            pad = self.crop_len - T
            return np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="constant")

        if self.train:
            start = np.random.randint(0, T - self.crop_len + 1)
        else:
            start = (T - self.crop_len) // 2

        return x[:, :, start:start + self.crop_len]

    def __getitem__(self, idx):
        x = self._crop(self.X[idx])
        y = self.y[idx]

        x = (x - self.mean) / (self.std + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)

        if self.train:
            if random.random() < 0.75:
                x = spec_augment_3ch(x)
            if random.random() < 0.5:
                x = random_time_shift(x)
            x = random_time_resample(x, p=0.4)

        return x, torch.tensor(y, dtype=torch.long)


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

    X, y, _, _ = build_cache(DATA_PATH, cache_dir=CACHE_DIR, max_len=MAX_LEN)
    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    mean = np.mean(X_train, axis=(0, 2, 3)).reshape(3, 1, 1)
    std = np.std(X_train, axis=(0, 2, 3)).reshape(3, 1, 1)

    train_ds = SERDataset(X_train, y_train, mean, std, CROP_LEN, train=True)
    train_eval_ds = SERDataset(X_train, y_train, mean, std, CROP_LEN, train=False)
    val_ds = SERDataset(X_val, y_val, mean, std, MAX_LEN, train=False)
    test_ds = SERDataset(X_test, y_test, mean, std, MAX_LEN, train=False)

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

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if random.random() < MIXUP_PROB:
                xb, y_a, y_b, lam = mixup_data(xb, yb, alpha=MIXUP_ALPHA)
                logits = model(xb)
                loss = mixup_loss(logits, y_a, y_b, lam, criterion)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        _, train_acc, train_f1, _, _ = evaluate_single(model, train_eval_loader, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_multicrop(model, val_loader, criterion, device, CROP_LEN)

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
    plot_sample_spectrograms(X_train, y_train, os.path.join(OUTPUT_DIR, "sample_spectrograms.png"))


if __name__ == "__main__":
    main()