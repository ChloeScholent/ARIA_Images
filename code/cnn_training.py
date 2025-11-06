import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dataset_class import ExerciseDataset, exercise_transform
from model_class import PowerliftingCNN
import matplotlib.pyplot as plt

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)
print('\n')
#DATASET

print('Loading powerlifting dataset...')
print('\n')

input_size = 224*224
num_classes = 3
batch_size = 32

dataset = ExerciseDataset("data/augmented_dataset/", transform=exercise_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)


print(f'train_loader: {train_loader} \ntest_loader: {test_loader}')
print('\n')
print('Dataset loaded successfully !')


#MODEL
print('Creation of the model...')

Powerlifting_CNN = PowerliftingCNN(num_classes).to(device)

print(Powerlifting_CNN)
print('\nModel created successfully !')
print('\n')

#TRAINING

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=Powerlifting_CNN.parameters(), lr=0.001)

def accuracy_fn(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    acc = (torch.sum(preds == labels).item()/len(preds))*100
    return acc

epochs = 31

print('Training...')
print('\n')

for epoch in range(epochs):
    losses = []
    test_losses = []
    accs = []
    test_accs = []
    for train_input, train_labels in train_loader:
        train_input = train_input.to(device)
        train_labels = train_labels.to(device)

        Powerlifting_CNN.train()

        outputs = Powerlifting_CNN(train_input)
        loss = loss_fn(outputs, train_labels) 
        losses.append(loss.item())
        acc = accuracy_fn(outputs, train_labels)
        accs.append(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses = sum(losses)/len(losses)
    train_acc = sum(accs)/len(accs)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Loss/Train', losses, epoch)
#EVALUATION
    Powerlifting_CNN.eval()
    with torch.inference_mode():
        for test_inputs, test_labels in test_loader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)

            test_outputs = Powerlifting_CNN(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            test_losses.append(test_loss.item())
            test_acc = accuracy_fn(test_outputs, test_labels)
            test_accs.append(test_acc)
        test_accs = sum(test_accs)/len(test_accs)
        test_loss = sum(test_losses)/len(test_losses)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', test_accs, epoch)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch} | Loss: {losses:.5f}, Accuracy: {sum(accs)/len(accs):.2f}% | Test loss: {(sum(test_losses)/len(test_losses)):.5f}, Test acc: {test_accs:.2f}%')



print('\n')
print('Training completed')

writer.flush()
writer.close()

#Confusion matrix and classification report

Powerlifting_CNN.eval()

all_preds = []
all_labels = []

with torch.inference_mode():
    for test_inputs, test_labels in test_loader:
        test_inputs = test_inputs.to(device)
        test_labels = test_labels.to(device)

        test_outputs = Powerlifting_CNN(test_inputs)
        preds = torch.argmax(test_outputs, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(test_labels.cpu().numpy())

# Concatenate all predictions
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Print confusion matrix & classification report
print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))


# cm = confusion_matrix(all_labels, all_preds)

# # Plot confusion matrix heatmap with matplotlib
# plt.figure(figsize=(8, 6))
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title("Confusion Matrix Heatmap")
# plt.colorbar()

# # Add labels, ticks, and numbers
# num_classes = cm.shape[0]
# tick_marks = np.arange(num_classes)
# plt.xticks(tick_marks, tick_marks)
# plt.yticks(tick_marks, tick_marks)

# # Add text annotations
# for i in range(num_classes):
#     for j in range(num_classes):
#         plt.text(j, i, str(cm[i, j]),
#                  ha='center', va='center',
#                  color='white' if cm[i, j] > cm.max() / 2 else 'black')

# plt.ylabel("True Label")
# plt.xlabel("Predicted Label")
# plt.tight_layout()

# # Save the heatmap
# plt.savefig(f"visual/confusion_matrix_heatmap_augmented.pdf", format="pdf", dpi=300)
# plt.close()

#Saving the model
MODEL_PATH = Path("Models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Powerlifting_CNN_Classification_Augmented.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#save the model state dictionary
print(f'Saving model to {MODEL_SAVE_PATH}')
torch.save(obj=Powerlifting_CNN.state_dict(), f=MODEL_SAVE_PATH)