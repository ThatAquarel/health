import torch

import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader

from hyperopt import hp
from hyperopt import fmin, tpe, space_eval


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


BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50

class_weights = torch.tensor([1.4586155, 3.775073, 23.640244, 228.05882, 352.45456])
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

train_dataloader = DataLoader(AntibioticDataset(), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(AntibioticDataset(train=False), batch_size=BATCH_SIZE)


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

        if batch % 8 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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

    return correct
    # print(
    #     f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    # )


def objective(params):
    n_features, n_layers = params["n_features"], params["n_layers"]

    class AntibioticPredictor(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear_relu_stack = nn.Sequential(
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(in_features=N_INDICATOR, out_features=n_features),
                *[
                    nn.Linear(in_features=n_features, out_features=n_features)
                    for n in range(n_layers)
                ],
                nn.Linear(in_features=n_features, out_features=5),
            )

        def forward(self, x):
            return self.linear_relu_stack(x)

    model = AntibioticPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for t in range(EPOCHS):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        test_loop(test_dataloader, model, loss_fn)

    return 1 / test_loop(test_dataloader, model, loss_fn)


search_space = {
    "n_features": hp.randint("n_features", 500, 3000),
    "n_layers": hp.randint("n_layers", 0, 5),
}

best = fmin(objective, space=search_space, algo=tpe.suggest, max_evals=1000)

print(best)
...

# 100%|█████████████████████| 1000/1000 [2:46:15<00:00,  9.98s/trial, best loss: 1.108695652173913]
# {'n_features': 788, 'n_layers': 0}
