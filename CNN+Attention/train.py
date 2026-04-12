# final model's training in src/model_cnn.py
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from load_data import load_dataset
from model_cnn import CNNEmotionRevised
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# replace with relative path to dataset if necessary
DATASET_PATH = "../AudioWAV"
ITERATIONS = 5
OUTPUT_DIR = "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y = load_dataset(DATASET_PATH)
print("Shape of features:", X.shape)
print("Shape of labels:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

model = CNNEmotionRevised(num_classes=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Validation function
def validate(model, device, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    return 100 * correct / total

# Training loop
epochs = 80
print(f"Training for {epochs} epochs on device: {device}")

train_accs_by_iter = np.zeros((ITERATIONS, epochs), dtype=np.float32)
val_accs_by_iter = np.zeros((ITERATIONS, epochs), dtype=np.float32)

test_accs = []
f1_scores = []
for iteration in range(ITERATIONS):
    print("Starting iteration:", iteration + 1, "/", ITERATIONS)
    print("=" * 50)

    model = CNNEmotionRevised(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        train_accs_by_iter[iteration, epoch] = 100.0 * correct / total

        val_acc = validate(model, device, val_loader)
        val_accs_by_iter[iteration, epoch] = val_acc

        print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {train_accs_by_iter[iteration, epoch]:.2f}%, Validation Accuracy: {val_acc:.2f}%")
        scheduler.step(val_acc)

    # Testing batch
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_targets.append(y_batch.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    test_acc = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average="macro")

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 score: {f1:.4f}")

    test_accs.append(test_acc)
    f1_scores.append(f1)

# plot average training and validation curves with standard deviation shading over epochs
epoch_range = np.arange(1, epochs + 1)
train_mean = train_accs_by_iter.mean(axis=0)
train_std = train_accs_by_iter.std(axis=0)
val_mean = val_accs_by_iter.mean(axis=0)
val_std = val_accs_by_iter.std(axis=0)

plt.figure(figsize=(10, 6))
plt.plot(epoch_range, train_mean, label="Training Mean", color="tab:blue")
plt.fill_between(epoch_range,
                 train_mean - train_std,
                 train_mean + train_std,
                 color="tab:blue",
                 alpha=0.2,
                 label="Training Std")
plt.plot(epoch_range, val_mean, label="Validation Mean", color="tab:orange")
plt.fill_between(epoch_range,
                 val_mean - val_std,
                 val_mean + val_std,
                 color="tab:orange",
                 alpha=0.2,
                 label="Validation Std")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Mean Accuracy with Std Dev")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "cnn_1_curves.png"))

# plot test accuracy on all iterations
plt.figure(figsize=(8, 5))
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracies Across Iterations")
plt.plot(range(1, ITERATIONS + 1), test_accs, marker="o", label="Test Accuracies")
plt.legend(loc="lower right")
plt.savefig(os.path.join(OUTPUT_DIR, "test_accs.png"))

mean_acc = np.mean(test_accs)
std_acc = np.std(test_accs)
mean_f1 = np.mean(f1_scores)

print("Test accuracies:", test_accs)
print("Average test accuracy with std dev:", mean_acc, "+/-", std_acc)
print("F1 scores:", f1_scores)
print("Average test F1 score:", mean_f1)

with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
    f.write("Test accuracies: " + str(test_accs) + "\n")
    f.write("Average test accuracy with std dev: " + str(mean_acc) + " +/- " + str(std_acc) + "\n")
    f.write("F1 scores: " + str(f1_scores) + "\n")
    f.write("Average test F1 score: " + str(mean_f1) + "\n")