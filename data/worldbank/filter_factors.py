import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./data/worldbank/2022-2000_worldbank_data.csv")
YEARS = [f"{i} [YR{i}]" for i in range(2003, 2023)]

data = data.copy()
data = data[["Country Name", "Series Name", *YEARS]]
data.loc[:, YEARS] = pd.isna(data[YEARS])

a = data.copy()
a = a.groupby(by=["Series Name"]).sum()
a = pd.melt(a.reset_index(), id_vars=["Series Name"], value_vars=YEARS, var_name="Year")
counts = a.groupby(by=["Series Name"]).sum()
counts.to_csv("./data/worldbank/filtering/counts_groupby_series_name.csv")

filtered = counts.loc[counts["value"] <= 2000]
filtered = filtered.reset_index()

b = data.copy()
b = b.groupby(by=["Country Name"])

...

# latest_year = data[["Country Name", "Series Name", *YEARS]].copy()
# latest_year.loc[:, YEARS] = pd.isna(latest_year[YEARS])
# latest_year["Counts"] = latest_year[YEARS].sum(axis=1)

# first filtered stage
#
# filtered1 = latest_year.merge(
#     filtered[["Series Name"]], how="inner", on=["Series Name"]
# )
# pivoted1 = pd.pivot_table(
#     filtered1, values=["Counts"], index=["Series Name"], columns=["Country Name"]
# )

# sns.set_theme(rc={"figure.figsize": (24, 16)})
# sns.clustermap(pivoted1.to_numpy())
# plt.show()


...
