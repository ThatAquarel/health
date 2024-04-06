import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/worldbank/2022-2000_worldbank_data.csv")
filtered = pd.read_csv("./data/worldbank/2022-2003_worldbank_filtered.csv")

YEARS = [f"{i} [YR{i}]" for i in range(2003, 2023)]

# calculate completness
unavailable = pd.isna(data[YEARS]).sum().sum()
total = np.multiply.reduce(data[YEARS].shape)
print(f"N/A percentage: {unavailable/total * 100}")

unavailable = pd.isna(filtered[YEARS]).sum().sum()
total = np.multiply.reduce(filtered[YEARS].shape)
print(f"N/A percentage: {unavailable/total * 100}")


a = data.copy()
a = a[["Country Name", "Series Name", *YEARS]]
a.loc[:, YEARS] = pd.isna(a[YEARS])
a = a.groupby(by=["Series Name"]).sum()
b = pd.melt(a.reset_index(), id_vars=["Series Name"], value_vars=YEARS, var_name="Year")
counts = b.groupby(by=["Series Name"]).sum()

plt.hist(counts.reset_index()["value"])
plt.show()

...

for YEAR in YEARS:
    columns = ["Country Name", "Series Name", YEAR]

    current = data[columns]
    current.loc[:, YEAR] = pd.isna(current[YEAR]) * 1

    completeness = current.pivot_table(YEAR, "Country Name", "Series Name")

    print(completeness.shape)
    print(YEAR)

    plt.imshow(completeness.to_numpy())
    plt.show()


data = pd.read_csv("./data/worldbank/2022-2000_worldbank_data.csv")
YEARS = [f"{i} [YR{i}]" for i in range(2003, 2023)]
data = data[["Country Name", "Series Name", *YEARS]]

a = data.copy()
a.loc[:, YEARS] = pd.isna(a[YEARS])
a = a.groupby(by=["Series Name"]).sum()
b = pd.melt(a.reset_index(), id_vars=["Series Name"], value_vars=YEARS, var_name="Year")

counts = b.groupby(by=["Series Name"]).sum()

plt.hist(counts.reset_index()["value"])
plt.show()
# visualize counts (availability) of each factor
...

# one year
# year = "2018 [YR2018]"

# latest_year = data[["Country Name", "Series Name", year]].copy()
# latest_year.loc[:, year] = pd.isna(latest_year[year])
# pivoted = pd.pivot_table(
#     latest_year, values=[year], index=["Series Name"], columns=["Country Name"]
# )

# sns.clustermap(pivoted)
# plt.show()

# 20 years availability heatmap
latest_year = data[["Country Name", "Series Name", *YEARS]].copy()
latest_year.loc[:, YEARS] = pd.isna(latest_year[YEARS])
latest_year["Counts"] = latest_year[YEARS].sum(axis=1)
pivoted = pd.pivot_table(
    latest_year, values=["Counts"], index=["Series Name"], columns=["Country Name"]
)

sns.set_theme(rc={"figure.figsize": (24, 16)})
sns.clustermap(pivoted.to_numpy())
plt.show()
