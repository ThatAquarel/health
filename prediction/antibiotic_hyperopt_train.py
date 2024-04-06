from ray import train, tune
from ray.tune.search.bayesopt import BayesOptSearch

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


torch.set_default_device("cpu")


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train
        self.load_db()

    def load_db(self):
        if train:
            self.x = torch.load(
                f"./prediction/x_2003-2017_train.pt", map_location="cpu"
            )
            self.y = torch.load(
                f"./prediction/y_2003-2017_train.pt", map_location="cpu"
            )
            return

        self.x = torch.load(f"./prediction/x_2018_test.pt", map_location="cpu")
        self.y = torch.load(f"./prediction/y_2018_test.pt", map_location="cpu")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


class AntibioticPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=804, out_features=405),
            nn.Tanh(),
            nn.Linear(in_features=405, out_features=5),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
    
def train_loop(dataloader, model, loss_fn, optimizer, epoch, writer=None, verbose=True):
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

        if batch % 8 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            if verbose:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, epoch=None, writer=None, verbose=True):
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
    if verbose:
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
    return correct


model = AntibioticPredictor()

BATCH_SIZE = 32
train_dataloader = DataLoader(AntibioticDataset(), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(AntibioticDataset(train=False), batch_size=BATCH_SIZE)


def objective(config):
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
        train_loop(train_dataloader, model, loss_fn, optimizer, epoch, verbose=False)
        acc = test_loop(test_dataloader, model, loss_fn, epoch, verbose=False)
        train.report({"mean_accuracy": acc})

        epoch += 1


search_space = {
    # "batch_size": tune.grid_search([2, 4, 8, 16, 32, 64]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "momentum": tune.uniform(0.1, 0.9),
    "class_weights_scale": tune.uniform(0.0, 1.0),
}
algo = BayesOptSearch()

tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
        search_alg=algo,
    ),
    run_config=train.RunConfig(
        stop={"training_iteration": 50},
    ),
    param_space=search_space,
)
results = tuner.fit()
print("Best config is:", results.get_best_result().config)
# Best config is: {'lr': 0.00951207163345817, 'momentum': 0.685595153449124, 'class_weights_scale': 0.3745401188473625}
