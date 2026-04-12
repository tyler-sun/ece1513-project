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

# Fixed data split seed for fair comparison across all runs
SPLIT_SEED = 42

# 5 independent training seeds
RUN_SEEDS = [42, 52, 62, 72, 82]

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
# Seed control
# =========================
def set_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Feature-space SpecAugment
# =========================
def spec_augment_3ch(x, p=0.4, F_mask=8, T_mask=12):
    """
    Light SpecAugment:
    randomly mask one frequency region and one time region.
    """
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
# Dataset classes
# =========================
class TrainWaveformDataset(Dataset):
    """
    Training dataset:
    - load waveform on the fly
    - apply waveform augmentation
    - convert to log-mel + delta + delta-delta
    - apply light SpecAugment
    """
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

        # Normalize using training-set statistics
        x = (x - self.mean) / (self.std + 1e-8)
        x = torch.tensor(x, dtype=torch.float32)

        # Feature-space masking
        x = spec_augment_3ch(x, p=0.4, F_mask=8, T_mask=12)

        return x, torch.tensor(self.labels[idx], dtype=torch.long)


class FeatureDataset(Dataset):
    """
    Clean feature dataset for:
    - train-set evaluation
    - validation
    - test
    """
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
# Multi-crop evaluation
# =========================
def get_eval_crops(xb, crop_len):
    """
    Generate left / center / right crops for more stable evaluation.
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
# Evaluation helpers
# =========================
def evaluate_single(model, loader, criterion, device):
    """
    Evaluate using one clean crop.
    Used mainly for train-set tracking.
    """
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
    """
    Evaluate by averaging predictions over multiple temporal crops.
    This makes validation/test predictions more robust.
    """
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
# Plot helpers
# =========================
def plot_curve_with_std(x, mean_train, std_train, mean_val, std_val, title, ylabel, save_path):
    """
    Plot mean curve with standard deviation shading across runs.
    """
    plt.figure(figsize=(8, 5))

    plt.plot(x, mean_train, label=f"Train {ylabel}")
    plt.fill_between(x, mean_train - std_train, mean_train + std_train, alpha=0.2)

    plt.plot(x, mean_val, label=f"Validation {ylabel}")
    plt.fill_between(x, mean_val - std_val, mean_val + std_val, alpha=0.2)

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Force full epoch range to appear on plot
    plt.xlim(1, EPOCHS)
    ticks = list(np.arange(1, EPOCHS + 1, 10))
    if EPOCHS not in ticks:
        ticks.append(EPOCHS)
    plt.xticks(ticks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, save_path):
    """
    Plot average confusion matrix across runs.
    """
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Average Across Runs)")
    plt.colorbar()

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center")

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_spectrograms(X, y, save_path):
    """
    Save one clean example spectrogram per class for the report.
    """
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
# One complete experiment run
# =========================
def run_one_experiment(run_idx, seed, X_all, y_all, paths_all, labels_all, device):
    """
    Run one full training / validation / test cycle.

    Returns:
        per-run curves, final test metrics, and confusion matrix
    """
    print(f"\n{'=' * 30}")
    print(f"Run {run_idx + 1}/{len(RUN_SEEDS)} | Seed = {seed}")
    print(f"{'=' * 30}")

    # Reproducibility for this run
    set_seed(seed)

    # Fixed split structure, only training randomness changes by seed
    idx_all = np.arange(len(labels_all))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, labels_all, test_size=0.2, stratify=labels_all, random_state=SPLIT_SEED
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SPLIT_SEED
    )

    X_train_clean = X_all[idx_train]
    X_val = X_all[idx_val]
    X_test = X_all[idx_test]

    train_paths = paths_all[idx_train]
    train_labels = labels_all[idx_train]

    # Compute normalization stats from clean training features only
    mean = np.mean(X_train_clean, axis=(0, 2, 3)).reshape(3, 1, 1)
    std = np.std(X_train_clean, axis=(0, 2, 3)).reshape(3, 1, 1)

    # Build datasets
    train_ds = TrainWaveformDataset(train_paths, train_labels, mean, std)
    train_eval_ds = FeatureDataset(X_train_clean, train_labels, mean, std)
    val_ds = FeatureDataset(X_val, y_val, mean, std)
    test_ds = FeatureDataset(X_test, y_test, mean, std)

    # Build dataloaders
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

    # Initialize model fresh for each run
    model = SER_CNN_Attention(NUM_CLASSES).to(device)

    # Balanced class weights for cross-entropy
    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    # Track best validation F1 for checkpointing
    best_f1 = 0.0
    best_path = os.path.join(OUTPUT_DIR, f"best_run_{run_idx + 1}.pt")

    # Curves stored per epoch
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    # =========================
    # Training loop
    # =========================
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        # Evaluate on clean train features and validation set
        train_loss = total_loss / len(train_loader)
        _, train_acc, train_f1, _, _ = evaluate_single(model, train_eval_loader, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate_multicrop(model, val_loader, criterion, device, CROP_LEN)

        # Scheduler reacts to validation F1
        scheduler.step(val_f1)

        # Save curves
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

        # Save best checkpoint according to validation F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    # =========================
    # Final test evaluation
    # =========================
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_multicrop(
        model, test_loader, criterion, device, CROP_LEN
    )
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)

    print(f"\nRun {run_idx + 1} Test Acc: {test_acc:.4f}")
    print(f"Run {run_idx + 1} Macro F1: {test_f1:.4f}")

    # Save per-run report
    with open(os.path.join(OUTPUT_DIR, f"test_results_run_{run_idx + 1}.txt"), "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Best Validation Macro F1: {best_f1:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Macro F1: {test_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    return {
        "train_losses": np.array(train_losses),
        "val_losses": np.array(val_losses),
        "train_accs": np.array(train_accs),
        "val_accs": np.array(val_accs),
        "train_f1s": np.array(train_f1s),
        "val_f1s": np.array(val_f1s),
        "test_acc": test_acc,
        "test_f1": test_f1,
        "cm": cm,
    }


# =========================
# Main entry point
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load cached clean features once
    X_all, y_all, _, _ = build_cache(DATA_PATH, cache_dir=CACHE_DIR, max_len=MAX_LEN)
    print("Loaded X:", X_all.shape)
    print("Loaded y:", y_all.shape)

    # Load metadata once
    paths_all, labels_all, _, _ = build_metadata(DATA_PATH)

    # Save one spectrogram figure for report use
    idx_all = np.arange(len(labels_all))
    idx_train, _, _, _ = train_test_split(
        idx_all, labels_all, test_size=0.2, stratify=labels_all, random_state=SPLIT_SEED
    )
    plot_sample_spectrograms(X_all[idx_train], labels_all[idx_train],
                             os.path.join(OUTPUT_DIR, "sample_spectrograms.png"))

    # Storage for all run outputs
    all_results = []
    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)

    # =========================
    # Run experiment 5 times
    # =========================
    for run_idx, seed in enumerate(RUN_SEEDS):
        result = run_one_experiment(
            run_idx, seed,
            X_all, y_all,
            paths_all, labels_all,
            device
        )
        all_results.append(result)
        cm_sum += result["cm"]

    # =========================
    # Aggregate metrics across runs
    # =========================
    train_losses_all = np.stack([r["train_losses"] for r in all_results], axis=0)
    val_losses_all = np.stack([r["val_losses"] for r in all_results], axis=0)
    train_accs_all = np.stack([r["train_accs"] for r in all_results], axis=0)
    val_accs_all = np.stack([r["val_accs"] for r in all_results], axis=0)
    train_f1s_all = np.stack([r["train_f1s"] for r in all_results], axis=0)
    val_f1s_all = np.stack([r["val_f1s"] for r in all_results], axis=0)

    test_accs = np.array([r["test_acc"] for r in all_results])
    test_f1s = np.array([r["test_f1"] for r in all_results])

    mean_train_losses = train_losses_all.mean(axis=0)
    std_train_losses = train_losses_all.std(axis=0)
    mean_val_losses = val_losses_all.mean(axis=0)
    std_val_losses = val_losses_all.std(axis=0)

    mean_train_accs = train_accs_all.mean(axis=0)
    std_train_accs = train_accs_all.std(axis=0)
    mean_val_accs = val_accs_all.mean(axis=0)
    std_val_accs = val_accs_all.std(axis=0)

    mean_train_f1s = train_f1s_all.mean(axis=0)
    std_train_f1s = train_f1s_all.std(axis=0)
    mean_val_f1s = val_f1s_all.mean(axis=0)
    std_val_f1s = val_f1s_all.std(axis=0)

    avg_cm = cm_sum / len(RUN_SEEDS)

    # Print final summary
    print("\n" + "=" * 35)
    print("FINAL AVERAGED RESULTS OVER RUNS")
    print("=" * 35)
    print(f"Test Accuracy: {test_accs.mean():.4f} ± {test_accs.std():.4f}")
    print(f"Macro F1:      {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")

    # Save aggregate summary for report table use
    with open(os.path.join(OUTPUT_DIR, "aggregate_results.txt"), "w") as f:
        f.write(f"Run seeds: {RUN_SEEDS}\n")
        f.write(f"Fixed split seed: {SPLIT_SEED}\n\n")

        for i, r in enumerate(all_results):
            f.write(
                f"Run {i+1} | Seed {RUN_SEEDS[i]} | "
                f"Test Accuracy: {r['test_acc']:.4f} | "
                f"Macro F1: {r['test_f1']:.4f}\n"
            )

        f.write("\n")
        f.write(f"Average Test Accuracy: {test_accs.mean():.4f}\n")
        f.write(f"Std Test Accuracy: {test_accs.std():.4f}\n")
        f.write(f"Average Macro F1: {test_f1s.mean():.4f}\n")
        f.write(f"Std Macro F1: {test_f1s.std():.4f}\n")

    # =========================
    # Plot averaged curves
    # =========================
    epochs = np.arange(1, EPOCHS + 1)

    plot_curve_with_std(
        epochs,
        mean_train_losses, std_train_losses,
        mean_val_losses, std_val_losses,
        "Loss Curve (Mean ± Std Across Runs)",
        "Loss",
        os.path.join(OUTPUT_DIR, "loss_curve.png"),
    )

    plot_curve_with_std(
        epochs,
        mean_train_accs, std_train_accs,
        mean_val_accs, std_val_accs,
        "Accuracy Curve (Mean ± Std Across Runs)",
        "Accuracy",
        os.path.join(OUTPUT_DIR, "accuracy_curve.png"),
    )

    plot_curve_with_std(
        epochs,
        mean_train_f1s, std_train_f1s,
        mean_val_f1s, std_val_f1s,
        "F1 Curve (Mean ± Std Across Runs)",
        "Macro F1",
        os.path.join(OUTPUT_DIR, "f1_curve.png"),
    )

    # Average confusion matrix across runs
    plot_confusion_matrix(avg_cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()