import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
g = sns.clustermap(
    matrix,
    col_colors=country_colors,
    col_cluster=False,
    dendrogram_ratio=(0.1, 0.1),
    cbar_pos=(0.01, 0.05, 0.02, 0.2),
    figsize=((45, 40)),
)

intervals = [
    "≤ 11.42",
    "(11.42, 20.04]",
    "(20.04, 28.66]",
    "(28.66, 37.28]",
    "> 37.28",
]

levels = ["Low", "Medium-low", "Medium", "Medium-high", "High"]

fig = g.figure
fig.legend(
    handles=[
        *[mpatches.Patch(color=color, label=levels[i]) for i, color in enumerate(pal)],
        *[mpatches.Patch(color="white", label=interval) for interval in intervals],
    ],
    ncols=2,
    title="Total antibiotic consumption (DDD/1,000/day)",
    loc="lower left",
)

fig.suptitle(
    "Comprehensive correspondence between top indicators (n=100)\n and total antibiotic usage worldwide (n=145 regions), 2022",
    fontsize=48,
)

plt.tight_layout()
plt.savefig("visualizations/heatmap_top100.png")
