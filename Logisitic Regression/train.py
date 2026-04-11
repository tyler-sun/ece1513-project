import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, log_loss
from sklearn.preprocessing import StandardScaler

from load_data import build_cache
from model_baseline import prepare_features, get_model


# =========================
# Config
# =========================
DATA_PATH = "data/crema"
CACHE_DIR = "cache"
OUTPUT_DIR = "outputs"

SPLIT_SEED = 42
RUN_SEEDS = [42, 52, 62, 72, 82]

BATCH_SIZE = 32   # kept only for same style; not used by logistic regression
EPOCHS = 90
LR = 4e-4         # kept only for same style; not used
WEIGHT_DECAY = 1e-4  # kept only for same style; not used

SR = 16000
N_MELS = 128
MAX_LEN = 360
CROP_LEN = 300    # kept only for same style; not used

NUM_CLASSES = 6
CLASS_NAMES = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

MIXUP_ALPHA = 0.10   # kept only for same style; not used
MIXUP_PROB = 0.35    # kept only for same style; not used


# =========================
# Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# =========================
# Eval
# =========================
def evaluate(model, X, y):
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)

    loss = log_loss(y, probs, labels=np.arange(NUM_CLASSES))
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    return (
        loss,
        acc,
        f1,
        np.array(y),
        np.array(preds),
    )


# =========================
# Plot functions
# =========================
def plot_confusion_matrix(cm, save_path):
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


def plot_curve_with_std(x, mean_train, std_train, mean_val, std_val, title, ylabel, save_path):
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

    plt.xlim(1, EPOCHS)
    ticks = list(np.arange(1, EPOCHS + 1, 10))
    if EPOCHS not in ticks:
        ticks.append(EPOCHS)
    plt.xticks(ticks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# =========================
# One run
# =========================
def run_one_experiment(run_idx, seed, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n{'=' * 25}")
    print(f"Run {run_idx + 1}/{len(RUN_SEEDS)} | Seed = {seed}")
    print(f"{'=' * 25}")

    set_seed(seed)

    X_train_feat = prepare_features(X_train)
    X_val_feat = prepare_features(X_val)
    X_test_feat = prepare_features(X_test)

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_val_feat = scaler.transform(X_val_feat)
    X_test_feat = scaler.transform(X_test_feat)

    X_train_feat = np.nan_to_num(X_train_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_feat = np.nan_to_num(X_val_feat, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_feat = np.nan_to_num(X_test_feat, nan=0.0, posinf=0.0, neginf=0.0)

    model = get_model(seed=seed)

    best_f1 = 0.0
    best_path = os.path.join(OUTPUT_DIR, f"best_run_{run_idx + 1}.pkl")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(EPOCHS):
        model.fit(X_train_feat, y_train)

        train_loss, train_acc, train_f1, _, _ = evaluate(model, X_train_feat, y_train)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, X_val_feat, y_val)

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
            with open(best_path, "wb") as f:
                pickle.dump((model, scaler), f)

    with open(best_path, "rb") as f:
        model, scaler = pickle.load(f)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, X_test_feat, y_test)
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)

    print(f"\nRun {run_idx + 1} Test Acc: {test_acc:.4f}")
    print(f"Run {run_idx + 1} Macro F1: {test_f1:.4f}")

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
# Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, _, _ = build_cache(DATA_PATH, cache_dir=CACHE_DIR, max_len=MAX_LEN)
    print("Loaded X:", X.shape)
    print("Loaded y:", y.shape)

    # fixed split for all runs
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SPLIT_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SPLIT_SEED
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val  :", X_val.shape, y_val.shape)
    print("Test :", X_test.shape, y_test.shape)

    plot_sample_spectrograms(X_train, y_train, os.path.join(OUTPUT_DIR, "sample_spectrograms.png"))

    all_results = []
    cm_sum = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)

    for run_idx, seed in enumerate(RUN_SEEDS):
        result = run_one_experiment(
            run_idx, seed,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test
        )
        all_results.append(result)
        cm_sum += result["cm"]

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

    print("\n" + "=" * 35)
    print("FINAL AVERAGED RESULTS OVER RUNS")
    print("=" * 35)
    print(f"Test Accuracy: {test_accs.mean():.4f} ± {test_accs.std():.4f}")
    print(f"Macro F1:      {test_f1s.mean():.4f} ± {test_f1s.std():.4f}")

    with open(os.path.join(OUTPUT_DIR, "aggregate_results.txt"), "w") as f:
        f.write(f"Run seeds: {RUN_SEEDS}\n")
        f.write(f"Fixed split seed: {SPLIT_SEED}\n\n")

        for i, r in enumerate(all_results):
            f.write(
                f"Run {i+1} | Seed {RUN_SEEDS[i]} | "
                f"Test Accuracy: {r['test_acc']:.4f} | Macro F1: {r['test_f1']:.4f}\n"
            )

        f.write("\n")
        f.write(f"Average Test Accuracy: {test_accs.mean():.4f}\n")
        f.write(f"Std Test Accuracy: {test_accs.std():.4f}\n")
        f.write(f"Average Macro F1: {test_f1s.mean():.4f}\n")
        f.write(f"Std Macro F1: {test_f1s.std():.4f}\n")

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

    plot_confusion_matrix(avg_cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()