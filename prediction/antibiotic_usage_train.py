import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter


torch.set_default_device("cuda:0")

RUNS = "./runs/"
MODEL = "./model/AntibioticPredictor.pt"


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train
        self.load_db()

    def load_db(self):
        if train:
            self.x = torch.load(f"./prediction/x_2003-2017_train.pt")
            self.y = torch.load(f"./prediction/y_2003-2017_train.pt")
            return

        self.x = torch.load(f"./prediction/x_2018_test.pt")
        self.y = torch.load(f"./prediction/y_2018_test.pt")

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


BATCH_SIZE = 50
LEARNING_RATE = 1e-4
EPOCHS = 100


def train():
    model = AntibioticPredictor()

    class_weights = torch.tensor(
        [1.98095238, 2.7752809, 8.82142857, 61.75, 188.19047619]
    )
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
        return correct

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, t)
        a = test_loop(test_dataloader, model, loss_fn, t)
        # if a >= 0.956:
        #     break
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
