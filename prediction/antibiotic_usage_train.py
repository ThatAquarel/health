import torch
import itertools

import pandas as pd

from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter


torch.set_default_device("cuda")

WORLDBANK = "./data/worldbank/2022-2000_worldbank_normalized.csv"
ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_normalized.csv"

RUNS = "./runs/"
MODEL = "./model/2024_02_28_AntibioticPredictor.pt"


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
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=N_INDICATOR, out_features=844),
            nn.ReLU(),
            nn.Linear(in_features=844, out_features=5),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


BATCH_SIZE = 50
LEARNING_RATE = 1e-4
EPOCHS = 128


def train():
    model = AntibioticPredictor()

    # np.unique(
    # pd.cut(
    # self._antibiotics["Antibiotic consumption (DDD/1,000/day)"], 5, labels=[0,1,2,3,4]
    # ), return_counts=True)
    # a = array([2658, 1027,  164,   17,   11], dtype=int64)
    # 3877/a

    class_weights = torch.tensor([1.4586155, 3.775073, 23.640244, 228.05882, 352.45456])
    class_weights = torch.sqrt(class_weights)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataloader = DataLoader(AntibioticDataset(), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(AntibioticDataset(train=False), batch_size=BATCH_SIZE)

    writer = SummaryWriter(RUNS)

    def train_loop(dataloader, model, loss_fn, optimizer, epoch):
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

            writer.add_scalar("Loss/train", loss, epoch)

            if batch % 8 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn, epoch=None):
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
                correct += (
                    (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                )

        if epoch:
            writer.add_scalar("Loss/test", test_loss, epoch)

        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        test_loop(test_dataloader, model, loss_fn, t)
    test_loop(test_dataloader, model, loss_fn)

    dataiter = iter(train_dataloader)
    inputs, labels = next(dataiter)

    writer.add_graph(model, inputs)
    writer.close()

    writer.flush()
    writer.close()

    torch.save(model.state_dict(), MODEL)


if __name__ == "__main__":
    train()
