# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:09:11 2025

@author: rouxm
"""

from SqueezeNet import MiniSqueezeNet # import your model class

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
torch.manual_seed(42)  # for reproducibility

# Parameters
print("Charging Parameters")

DATA_DIR = r"C://Users//rouxm//Desktop//2025-2026 - ARIA//Image//SqueezeNet"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

LR_START=1e-3
LR_END=1e-6
DEVICE = torch.device("cpu")

print("Setting Datasets") #OK
print(os.path.isdir(f"{DATA_DIR}/train")) #OK
# Datasets
transform = {
    'train': transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.5],[0.5])
    ]),
    'valid': transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.5],[0.5])
    ])
}

print("Dataset successfully created") #OK

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform['train'])
print("train_ds created") #Tourne, mais après si il le considère vide ...
valid_ds = datasets.ImageFolder(f"{DATA_DIR}/valid", transform=transform['valid'])


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
print("Train_loader Successfully created") 
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)


print("Classes:", train_ds.classes)

# Model
model = MiniSqueezeNet(num_classes=len(train_ds.classes)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_lambda = lambda epoch: (LR_END/LR_START) + (1 - epoch/(EPOCHS-1)) * (1 - LR_END/LR_START)
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Training loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} ",f"lr={optimizer.param_groups[0]['lr']:.6f}")
    # ---- Train ----
    model.train()
    running_loss, correct, total = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Train loss: {running_loss/total:.4f}, acc: {correct/total:.4f}")

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Valid acc: {correct/total:.4f}")
    

# Save the model
torch.save(model.state_dict(), "MiniSqueeze_1.pth")
print("Model saved!")
