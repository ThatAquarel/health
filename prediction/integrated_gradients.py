import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from antibiotic_usage_train import AntibioticPredictor, MODEL


model = AntibioticPredictor()
model.load_state_dict(torch.load(MODEL))
model.eval()

# use ./prediction/db_x_test.pt to generate
# ordered_factors_2018_high.csv
# ordered_factors_2018_low.csv

# use ./prediction/db_x.pt to generate
# ordered_factors_2003_2018_high.csv
# ordered_factors_2003_2018_low.csv

db_x_test = torch.load("./prediction/db_x.pt")
baseline = torch.zeros(db_x_test.shape)

...

ig = IntegratedGradients(model)

for level, label in {0: "low", 4: "high"}.items():
    attributions, approximation_error = ig.attribute(
        db_x_test,
        baselines=baseline,
        method="gausslegendre",
        return_convergence_delta=True,
        target=level,
    )

    importance = np.mean(attributions.cpu().numpy(), axis=0)

    indices = np.argsort(importance)
    factors = pd.read_csv("./data/worldbank/links/Series_Name_Series_Code.csv")
    factors = factors[["Series Name", "Series Code"]]

    factors.insert(2, "Attribution", list(importance), True)

    ordered = factors.reindex(indices)
    ordered.to_csv(f"./prediction/results/ordered_factors_2003_2018_{label}.csv")

# def visualize_importances(
#     feature_names,
#     importances,
#     title="Average Feature Importances",
#     plot=True,
#     axis_title="Features",
# ):
#     print(title)
#     for i in range(len(feature_names)):
#         print(feature_names[i], ": ", "%.3f" % (importances[i]))
#     x_pos = np.arange(len(feature_names))
#     if plot:
#         plt.figure(figsize=(12, 6))
#         plt.bar(x_pos, importances, align="center")
#         plt.xticks(x_pos, feature_names, wrap=True)
#         plt.xlabel(axis_title)
#         plt.title(title)
# visualize_importances(np.arange(1683), importance, axis=0)
# plt.show()
