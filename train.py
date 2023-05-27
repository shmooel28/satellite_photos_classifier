import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CustomImageDataset import CustomImageDataset
from Classifier import Classifier

from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
batch_size = 64
num_epochs = 10

model = Classifier().to(device)

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

dataset = CustomImageDataset(main_dir="D:\Capital City")
print(len(dataset))
# Split the dataset into training and validation sets
train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)


def tran_loop(dataloader, model, loss_fn,optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correcr = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correcr += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correcr /= size
    print(f"Tets Error: \n Accuracy: {(100*correcr):>0.1f}%, Avg loss: {test_loss:>8f} \n")



for t in range(num_epochs):
    print(f"Epoch {t+1}\n-----------------------")
    tran_loop(train_loader,model,loss_fn,optimizer)
    test_loop(valid_loader,model,loss_fn)
print("Done!")
# Training loop

model_path = 'model.pth'

# Save the model
torch.save(model.state_dict(), model_path)