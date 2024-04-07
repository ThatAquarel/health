import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

cases = pd.read_csv("prediction/x_2003-2022_infer_cases.csv")
cases = cases[["Country Name", "Year"]]
cases = cases[cases["Year"] == 2021]

indicators = pd.read_csv("prediction/x_2003-2022_infer_factors.csv")
indicators = indicators[["Series Name"]]
top_indicators = pd.read_csv("prediction/results/top100_factors.csv")
indicators = indicators.merge(
    top_indicators[["Series Name"]], how="inner", on="Series Name"
)
indicators_idx = list(indicators.index)
del indicators_idx[25:80]

predicted_categories = pd.read_csv("prediction/results/predicted_categories.csv")
assert list(cases["Country Name"]) == list(predicted_categories["Country Name"])

categories_idx = np.array(predicted_categories["Predicted Category"]).argsort()
pal = sns.color_palette("Reds", 5)
country_colors = [
    pal[int(i)] for i in predicted_categories["Predicted Category"][categories_idx]
]

worldbank = torch.load("prediction/x_2003-2022_infer.pt")
worldbank = worldbank[cases.index]
worldbank = worldbank[:, indicators_idx]
worldbank = worldbank[categories_idx]

# countries =

values = worldbank.cpu().numpy().astype(np.float32)
values = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0))
values = np.nan_to_num(values)

sns.clustermap(
    values.T,
    col_colors={"Antibiotic risk": country_colors},
    col_cluster=False,
    cmap="mako",
)
plt.show()
