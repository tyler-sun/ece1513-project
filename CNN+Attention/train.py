# final model's training in src/model_cnn.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from load_data import load_dataset
from model_cnn import CNNEmotionRevised
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Replace with relative path to dataset
DATASET_PATH = "../../AudioWAV"

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
train_accs = []
val_accs = []
epochs = 50
print(f"Training for {epochs} epochs on device: {device}")
for epoch in range(epochs):
    model.train()
    running_loss = 0
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
        running_loss += loss.item()
    
    train_accs.append(100 * correct/total)

    val_acc = validate(model, device, val_loader)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{epochs}, Training Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {val_acc:.2f}%")
    scheduler.step(val_acc)

plt.figure()
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Curves")
plt.plot(train_accs, label="Training")
plt.plot(val_accs, label="Validation")
plt.legend(loc="lower right")
plt.savefig("cnn_1_curves.png")

# Testing batch
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

f1 = f1_score(y_batch.cpu().numpy(), predicted.cpu().numpy(), average="macro")
print(f"Test Accuracy: {100 * correct / total:.2f}%")
print(f"Test F1 score: {f1: .4f}")