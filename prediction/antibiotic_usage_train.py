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

        self._worldbank = pd.read_csv(WORLDBANK)
        self._prep_worldbank(2003, 2019)
        self._test_year = 2018

        self._antibiotics = pd.read_csv(ANTIBIOTICS)
        self._prep_antiobiotics("J01C-Penicillins")

        self._cut_bins(5)

        self._build_db_idx()
        self._build_db()
        # print(np.unique(self._antibiotics[self.CONSUMPTION], return_counts=True))
        # print(len(self))

        self._worldbank[["Indicator"]] = self._worldbank[["Indicator"]].astype(
            "float32"
        )
        ...

    def _prep_worldbank(self, start_year, end_year):
        lower = self._worldbank["Year"] >= start_year
        upper = self._worldbank["Year"] < end_year

        self._worldbank = self._worldbank[lower & upper]
        self._worldbank = self._worldbank[
            ["Country Name", "Year", "Series Name", "Indicator"]
        ]

        self._worldbank = self._worldbank.fillna(0)

    CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"

    def _prep_antiobiotics(self, atc_3_class):
        self._antibiotics = self._antibiotics[
            self._antibiotics["ATC level 3 class"] == atc_3_class
        ]
        self._antibiotics = self._antibiotics[
            ["Country Name", "Year", self.CONSUMPTION]
        ]

    def _cut_bins(self, n_bins):
        self.n_bins = n_bins

        bins = list(range(n_bins))

        self._antibiotics.loc[:, self.CONSUMPTION] = pd.cut(
            self._antibiotics[self.CONSUMPTION], n_bins, labels=bins
        )

        self._one_hot = torch.zeros((n_bins, n_bins)).float()
        self._one_hot[bins, bins] = 1

    def _build_db_idx(self):
        a = self._antibiotics[["Country Name"]].drop_duplicates()
        b = self._worldbank[["Country Name"]].drop_duplicates()
        ax1 = a.merge(b)["Country Name"]

        a = self._antibiotics[["Year"]].drop_duplicates()
        b = self._worldbank[["Year"]].drop_duplicates()
        ax2 = a.merge(b)["Year"] if self.train else [self._test_year]

        self._db_idx = list(itertools.product(ax1, ax2))

    def _build_db(self):
        self.db_x = torch.zeros(len(self), N_INDICATOR)
        self.db_y = torch.zeros(len(self), self.n_bins)

        for idx in tqdm(range(len(self))):
            ax1, ax2 = self._db_idx[idx]

            a_country = self._antibiotics["Country Name"] == ax1
            a_year = self._antibiotics["Year"] == ax2
            (c,) = self._antibiotics.loc[a_country & a_year, self.CONSUMPTION].values
            y = self._one_hot[c]

            b_country = self._worldbank["Country Name"] == ax1
            b_year = self._worldbank["Year"] == ax2
            i = self._worldbank.loc[b_country & b_year, "Indicator"].values
            x = torch.tensor(i).float()

            self.db_x[idx] = x
            self.db_y[idx] = y

    def __len__(self):
        return len(self._db_idx)

    def __getitem__(self, idx):
        return (self.db_x[idx], self.db_y[idx])


class AntibioticPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=N_INDICATOR, out_features=N_INDICATOR),
            nn.ReLU(),
            nn.Linear(in_features=N_INDICATOR, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=5),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100

model = AntibioticPredictor()

loss_fn = nn.CrossEntropyLoss()
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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, optimizer)
test_loop(test_dataloader, model, loss_fn, optimizer)

...
