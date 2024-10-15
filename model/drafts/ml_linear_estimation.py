import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = "cuda"


class LinearDataset(Dataset):
    def __init__(self, train=True):
        self.train = train

    def __len__(self):
        return 2048 if self.train else 16

    def __getitem__(self, idx):
        rand = (
            torch.rand(len(self))
            if self.train
            else [
                0.0244,
                0.7002,
                0.5293,
                0.8554,
                0.1242,
                0.9431,
                0.8787,
                0.7753,
                0.9811,
                0.9194,
                0.0589,
                0.7592,
                0.9974,
                0.9651,
                0.7171,
                0.0100,
            ]
        )
        lin = torch.rand(len(self)) if self.train else torch.linspace(0, 1, len(self))

        return (torch.tensor([rand[idx], lin[idx]]), torch.tensor([lin[idx]]))


class LinearEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


model = LinearEstimationModel()

learning_rate = 1e-3
batch_size = 64
epochs = 100


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 4 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y).item()

            ...


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(LinearDataset(), batch_size=batch_size)
test_dataloader = DataLoader(LinearDataset(train=False), batch_size=16)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn, optimizer)
test_loop(test_dataloader, model, loss_fn, optimizer)
print("Done!")
