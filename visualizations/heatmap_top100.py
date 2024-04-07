import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

cases = pd.read_csv("prediction/x_2003-2022_infer_cases.csv")
cases = cases[["Country Name", "Year"]]
cases = cases[cases["Year"] == 2021]

# indicators
all_indicators = pd.read_csv("prediction/x_2003-2022_infer_factors.csv")
all_indicators = all_indicators[["Series Name"]]
top_indicators = pd.read_csv("prediction/results/top100_factors.csv")
indicator_filter = all_indicators[
    all_indicators["Series Name"].isin(top_indicators["Series Name"])
]

# sort predicted categories
predicted_categories = pd.read_csv("prediction/results/predicted_categories.csv")
categories_idx = np.array(predicted_categories["Predicted Category"]).argsort()
pal = sns.color_palette("Reds", 5)
# pal_lut = {i: pal[i] for i in range(5)}
# country_colors = predicted_categories["Predicted Category"][categories_idx].map(pal_lut)
country_colors = [
    pal[i] for i in predicted_categories["Predicted Category"][categories_idx]
]

worldbank = torch.load("prediction/x_2003-2022_infer.pt")
matrix = pd.DataFrame(worldbank.cpu().T, index=all_indicators["Series Name"])
matrix = matrix[matrix.columns[cases.index]]
matrix = matrix.rename(
    columns={a: b for a, b in zip(cases.index, cases["Country Name"])}
)

# filter matrix
matrix = matrix[matrix.columns[categories_idx]]
matrix = matrix[matrix.index.isin(indicator_filter["Series Name"])]
country_colors = pd.Series(
    country_colors,
    index=matrix.columns,
    name="Total antibiotic consumption (DDD/1,000/day)",
)

amin = matrix.min(axis=1)
amax = matrix.max(axis=1)

matrix = matrix.sub(amin, axis=0).div(amax - amin, axis=0)
matrix = matrix.fillna(0)

# show heatmap
sns.clustermap(
    matrix,
    col_colors=country_colors,
    col_cluster=False,
    dendrogram_ratio=(0.1, 0.1),
    cbar_pos=(0.02, 0.05, 0.028, 0.04),
    figsize=((40, 40)),
)
plt.savefig("visualizations/heatmap_top100.png")
