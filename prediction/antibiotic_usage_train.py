import torch
import itertools

import pandas as pd

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.set_default_device("cuda")

WORLDBANK = "./data/worldbank/2022-2000_worldbank_normalized.csv"
ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_normalized.csv"


N_INDICATOR = 1683


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train
        self.load_db()

    def load_db(self):
        x = "" if self.train else "_test"
        self.db_x = torch.load(f"./prediction/db_x{x}.pt")
        self.db_y = torch.load(f"./prediction/db_y{x}.pt")

    def __len__(self):
        return len(self.db_x)

    def __getitem__(self, idx):
        return (self.db_x[idx], self.db_y[idx])


class AntibioticPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=N_INDICATOR, out_features=N_INDICATOR),
            nn.ReLU(),
            nn.Linear(in_features=N_INDICATOR, out_features=5),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100

model = AntibioticPredictor()

class_weights = torch.tensor([0.37748772, 1.3284197, 3.1629505, 5.4296036, 5.8649216])
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

train_dataloader = DataLoader(AntibioticDataset(), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(AntibioticDataset(train=False), batch_size=BATCH_SIZE)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 8 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
test_loop(test_dataloader, model, loss_fn)

...
