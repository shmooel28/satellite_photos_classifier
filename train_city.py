import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from CustomImageDataset import CustomImageDataset
from Classifier import Classifier

from sklearn.model_selection import train_test_split
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

class_labels = ['Ashdod', 'Hebron', 'Jerusalem', 'Tel-Aviv']

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
batch_size = 64
num_epochs = 20

model = Classifier().to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


transformations = [
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomVerticalFlip(p=0.001),
    transforms.RandomHorizontalFlip(p=0.001),
    transforms.ToTensor(),
]
transform = transforms.Compose(transformations)
dataset = CustomImageDataset(main_dir="D:\Capital City\Israel\City",transform=transform)
print(len(dataset))
# Split the dataset into training and validation sets
train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

class_counts = [0] * 4
for _, label in train_data:
    class_counts[label] += 1

count = 0
for class_count in class_counts:
    print(class_labels[count], " = ", class_count)
    count += 1
class_weights = []
total_samples = len(train_data)
for class_count in class_counts:
    class_weight = 1 / (class_count / total_samples)
    class_weights.append(class_weight)
class_weights = torch.tensor(class_weights)
class_weights = class_weights.to(device)


def tran_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        weights = class_weights[y]
        weighted_loss = torch.mean(loss * weights)

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    predictions = []
    targets = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    f1 = calculate_f1_score(predictions, targets)
    print(f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}, F1 Score: {f1:.4f}\n")


def calculate_f1_score(predictions, targets):
    predictions = np.argmax(predictions, axis=1)
    return f1_score(targets, predictions, average='weighted')


def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    predictions = []
    targets = []
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    test_loss /= size
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    return test_loss, predictions, targets


train_losses = []
train_f1_scores = []
valid_losses = []
valid_f1_scores = []

for t in range(num_epochs):
    print(f"Epoch {t + 1}\n-----------------------")
    tran_loop(train_loader, model, loss_fn, optimizer)
    train_loss, train_predictions, train_targets = evaluate(train_loader, model, loss_fn)
    valid_loss, valid_predictions, valid_targets = evaluate(valid_loader, model, loss_fn)

    train_f1 = calculate_f1_score(train_predictions, train_targets)
    valid_f1 = calculate_f1_score(valid_predictions, valid_targets)

    train_losses.append(train_loss)
    train_f1_scores.append(train_f1)
    valid_losses.append(valid_loss)
    valid_f1_scores.append(valid_f1)

    print(f"Train Loss: {train_loss:.4f} | Train F1 Score: {train_f1:.4f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid F1 Score: {valid_f1:.4f}")

# Plot the training results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_f1_scores, label='Train F1 Score')
plt.plot(valid_f1_scores, label='Valid F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.show()

print("Done!")

# Save the model
model_path = 'city_model.pth'
torch.save(model.state_dict(), model_path)
