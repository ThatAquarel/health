import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
species = iris.pop("species")
iris.index = ["a"] * 150

sns.clustermap(iris)
plt.show()


import pandas as pd
import seaborn as sns

sns.set_theme()

# Load the brain networks example dataset
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# Select a subset of the networks
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = df.columns.get_level_values("network").astype(int).isin(used_networks)
df = df.loc[:, used_columns]

# Create a categorical palette to identify the networks
network_pal = sns.husl_palette(8, s=0.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# Convert the palette to vectors that will be drawn on the side of the matrix
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

# Draw the full plot
g = sns.clustermap(
    df.corr(),
    center=0,
    cmap="vlag",
    row_colors=network_colors,
    col_colors=network_colors,
    dendrogram_ratio=(0.1, 0.2),
    cbar_pos=(0.02, 0.32, 0.03, 0.2),
    linewidths=0.75,
    figsize=(12, 13),
)

g.ax_row_dendrogram.remove()


colors = list(df["Color"].unique())
for i in range(0, len(colors)):
    data = df.loc[df["Color"] == colors[i]]
    plt.scatter("A", "B", data=data, color="Color", label=colors[i])
plt.legend()
plt.show()
