import torch

import seaborn as sns
import matplotlib.pyplot as plt

x = torch.load("./prediction/x_2018_test.pt")
x = x.cpu()
y = torch.load("./prediction/y_2018_test.pt")
y = y.argmax(1).cpu()

y, idx = y.sort()
x = x[idx]

pal = sns.husl_palette(5, s=0.45)
y_colors = [pal[int(i)] for i in y]

sns.set_theme()

g = sns.clustermap(
    x,
    center=0,
    cmap="vlag",
    row_colors=y_colors,
    row_cluster=False,
    cbar_pos=(0.02, 0.32, 0.03, 0.2),
    figsize=(12, 13),
)

plt.show()
