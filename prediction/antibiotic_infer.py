import torch
from torch.utils.data import DataLoader
from antibiotic_usage_train import (
    AntibioticPredictor,
    AntibioticDataset,
    MODEL,
    BATCH_SIZE,
)


def infer():
    db_x_test = torch.load("./prediction/db_x_test.pt")
    db_y_test = torch.load("./prediction/db_y_test.pt")

    db_x = torch.load("./prediction/db_x.pt")
    db_y = torch.load("./prediction/db_y.pt")

    model = AntibioticPredictor()
    model.load_state_dict(torch.load(MODEL))

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        pred_y_test = model(db_x_test)
        class_pred = pred_y_test.argmax(1)
        class_real = db_y_test.argmax(1)

        mark = (class_pred == class_real).type(torch.float)

        print(
            f"2018 Prediction accuracy {torch.sum(mark).item()}/{len(class_pred)}, {torch.mean(mark).item()}%"
        )

        # 2018 Prediction accuracy 195.0/204, 0.9558823704719543%

        pred_y = model(db_x)
        class_pred = pred_y.argmax(1)
        class_real = db_y.argmax(1)

        mark = (class_pred == class_real).type(torch.float)

        print(
            f"2002-2017 Prediction accuracy {torch.sum(mark).item()}/{len(class_pred)}, {torch.mean(mark).item()}%"
        )

        # 2002-2017 Prediction accuracy 3039.0/3060, 0.9931373000144958%


if __name__ == "__main__":
    infer()
