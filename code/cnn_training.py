import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from dataset_class import ExerciseDataset, exercise_transform
from model_class import PowerliftingCNN, accuracy_fn


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------
# DATASET LOADING
# -----------------------
print("\nLoading powerlifting dataset...\n")

num_classes = 3
batch_size = 32

dataset = ExerciseDataset("data/augmented_dataset/", transform=exercise_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print("Dataset loaded successfully!\n")

# -----------------------
# MODEL CREATION
# -----------------------
print("Creating model...\n")
Powerlifting_CNN = PowerliftingCNN(num_classes).to(device)
print(Powerlifting_CNN)
print("\nModel created successfully!\n")

# -----------------------
# TRAINING SETUP
# -----------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Powerlifting_CNN.parameters(), lr=0.001)
epochs = 31

train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

print("Training...\n")

# -----------------------
# TRAINING LOOP
# -----------------------
for epoch in range(epochs):
    Powerlifting_CNN.train()
    losses, accs = [], []

    for train_input, train_labels in train_loader:
        train_input, train_labels = train_input.to(device), train_labels.to(device)

        outputs = Powerlifting_CNN(train_input)
        loss = loss_fn(outputs, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy_fn(outputs, train_labels))

    # Evaluation
    Powerlifting_CNN.eval()
    test_losses, test_accs = [], []

    with torch.inference_mode():
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = Powerlifting_CNN(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)

            test_losses.append(test_loss.item())
            test_accs.append(accuracy_fn(test_outputs, test_labels))

    # Save metrics per epoch
    train_loss_history.append(np.mean(losses))
    test_loss_history.append(np.mean(test_losses))
    train_acc_history.append(np.mean(accs))
    test_acc_history.append(np.mean(test_accs))

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss_history[-1]:.4f} | Test Loss: {test_loss_history[-1]:.4f} | "
          f"Train Acc: {train_acc_history[-1]:.4f} | Test Acc: {test_acc_history[-1]:.4f}")

print("\nTraining completed.\n")

# -----------------------
# PLOTTING
# -----------------------
Path("plots").mkdir(exist_ok=True)

# Accuracy curve
plt.figure()
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.savefig("plots/accuracy_curve.pdf", dpi=300)

# Loss curve
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(test_loss_history, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.savefig("plots/loss_curve.pdf", dpi=300)

# -----------------------
# CONFUSION MATRIX
# -----------------------
Powerlifting_CNN.eval()
all_preds, all_labels = [], []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_outputs = Powerlifting_CNN(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))

# -----------------------
# SAVE MODEL
# -----------------------
# model_path = Path("Models")
# model_path.mkdir(exist_ok=True)
# save_path = model_path / "Powerlifting_CNN_Classification_Augmented.pth"
# torch.save(Powerlifting_CNN.state_dict(), save_path)
# print(f"\nModel saved to: {save_path}")
