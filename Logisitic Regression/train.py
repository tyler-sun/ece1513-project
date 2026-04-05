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

SEED = 42
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

    # same sample figure style as deep model
    plot_sample_spectrograms(X_train, y_train, os.path.join(OUTPUT_DIR, "sample_spectrograms.png"))

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

    model = get_model(seed=SEED)

    best_f1 = 0.0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    best_path = os.path.join(OUTPUT_DIR, "best.pkl")

    for epoch in range(EPOCHS):
        # one more optimization pass each epoch
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


if __name__ == "__main__":
    main()