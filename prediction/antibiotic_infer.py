import torch
import numpy as np
import pandas as pd
from antibiotic_usage_train import (
    AntibioticPredictor,
    MODEL,
)

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


def infer():
    db_x_test = torch.load("./prediction/x_2018_test.pt")
    db_y_test = torch.load("./prediction/y_2018_test.pt")

    db_x = torch.load("./prediction/x_2003-2017_train.pt")
    db_y = torch.load("./prediction/y_2003-2017_train.pt")

    model = AntibioticPredictor()
    model.load_state_dict(torch.load(MODEL))

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        pred_y_test = model(db_x_test)
        class_pred_test = pred_y_test.argmax(1)
        class_real_test = db_y_test.argmax(1)

        mark_test = (class_pred_test == class_real_test).type(torch.float)

        pred_y = model(db_x)
        class_pred_train = pred_y.argmax(1)
        class_real_train = db_y.argmax(1)

        mark_train = (class_pred_train == class_real_train).type(torch.float)

    print(
        f"2018 Prediction accuracy {torch.sum(mark_test).item()}/{len(class_pred_test)}"
    )
    print(f"{torch.mean(mark_test).item()*100}%")
    print(
        f"2002-2017 Prediction accuracy {torch.sum(mark_train).item()}/{len(class_pred_train)}"
    )
    print(f"{torch.mean(mark_train).item()*100}%")

    # 2018 Prediction accuracy 195.0/204
    # 95.58823704719543%
    # 2002-2017 Prediction accuracy 3039.0/3060
    # 99.31373000144958%

    plt.rcParams["figure.figsize"] = (8, 8)
    for title, (pred, real) in zip(
        [
            "2018 Penicillin Antibiotic Usage and Resistance Risk Prediction Confusion matrix",
            "2003-2017 Penicillin Antibiotic Usage and Resistance Risk Prediction Confusion matrix",
        ],
        [(class_pred_test, class_real_test), (class_pred_train, class_real_train)],
    ):
        cm_display = ConfusionMatrixDisplay.from_predictions(
            real.cpu().numpy(),
            pred.cpu().numpy(),
            display_labels=["Low", "Medium-low", "Medium", "Medium-high", "High"],
            cmap=plt.cm.Blues,
            normalize="pred",
        )
        disp = cm_display.plot()
        disp.ax_.set_title(title)

        plt.show()

    db_x_infer = torch.load("./prediction/x_2003-2022_infer.pt")
    db_x_infer_cases = pd.read_csv("./prediction/x_2003-2022_infer_cases.csv")
    db_x_infer_cases = db_x_infer_cases[["Country Name", "Year"]]

    with torch.no_grad():
        pred_y_infer = model(db_x_infer)
        class_pred = pred_y_infer.argmax(1)

    db_x_infer_cases.insert(
        2,
        "Predicted Category",
        class_pred.cpu().numpy().tolist(),
        allow_duplicates=True,
    )
    db_x_infer_cases.to_csv("./prediction/results/2003-2022_predicted_categories.csv")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    infer()
