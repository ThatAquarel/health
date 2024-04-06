import os

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


torch.set_default_device("cpu")
CWD = os.getcwd()


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train
        self.load_db()

    def _load(self, rel_path):
        full_path = os.path.join(CWD, rel_path)
        return torch.load(full_path, map_location="cpu")

    def load_db(self):
        if self.train:
            self.x = self._load(f"./prediction/x_2003-2017_train.pt")
            self.y = self._load(f"./prediction/y_2003-2017_train.pt")
            return

        self.x = self._load(f"./prediction/x_2018_test.pt")
        self.y = self._load(f"./prediction/y_2018_test.pt")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer=None):
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

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)


def test_loop(dataloader, model, loss_fn, epoch=None, writer=None):
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

    if epoch:
        if writer:
            writer.add_scalar("Loss/test", test_loss, epoch)

    test_loss /= num_batches
    correct /= size
    return correct


def objective(config):
    class AntibioticPredictor(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear_relu_stack = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=804, out_features=config["hidden_n"]),
                nn.Tanh(),
                nn.Linear(in_features=config["hidden_n"], out_features=5),
            )

        def forward(self, x):
            return self.linear_relu_stack(x)

    train_dataloader = DataLoader(AntibioticDataset(), batch_size=config["batch_size"])
    test_dataloader = DataLoader(
        AntibioticDataset(train=False), batch_size=config["batch_size"]
    )

    model = AntibioticPredictor()

    optimizer = torch.optim.SGD(  # Tune the optimizer
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    class_weights = torch.tensor(
        [1.98095238, 2.7752809, 8.82142857, 61.75, 188.19047619]
    )
    class_weights = class_weights ** config["class_weights_scale"]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epoch = 0
    while True:
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch)
        acc = test_loop(test_dataloader, model, loss_fn, epoch)
        train.report({"mean_accuracy": acc})

        epoch += 1


search_space = {
    "batch_size": tune.grid_search([2, 4, 8, 16, 32, 64, 128]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "momentum": tune.uniform(0.1, 0.9),
    "class_weights_scale": tune.uniform(0.0, 1.0),
    "hidden_n": tune.randint(5, 1608),
}

results = tune.run(
    objective,
    num_samples=16,
    scheduler=ASHAScheduler(metric="mean_accuracy", mode="max", grace_period=1),
    config=search_space,
)

# tuner = tune.Tuner(
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="mean_accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     run_config=train.RunConfig(stop={"training_iteration": 100, "mean_accuracy": 0.95}),
#     param_space=search_space,
# )
# results = tuner.fit()
print("Best config is:", results.best_result().config)
