import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


# determine filter merge tables

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

filtered_series = counts.loc[counts["value"] <= 2000].reset_index()

b = data.copy()
b = b.merge(filtered_series[["Series Name"]], how="inner", on=["Series Name"])
b = b.groupby(by=["Country Name"]).sum()
b = pd.melt(
    b.reset_index(), id_vars=["Country Name"], value_vars=YEARS, var_name="Year"
)
counts = b.groupby(by=["Country Name"]).sum()
counts.to_csv("./data/worldbank/filtering/counts_groupby_country_name.csv")

filtered_countries = counts.loc[counts["value"] <= 2000].reset_index()


# display filtered data


def display_available(subset):
    pivoted = pd.pivot_table(
        subset, values=["Counts"], index=["Series Name"], columns=["Country Name"]
    )

    sns.set_theme(rc={"figure.figsize": (24, 16)})
    sns.clustermap(pivoted.to_numpy())
    plt.show()


all_years = data.copy()
all_years["Counts"] = all_years[YEARS].sum(axis=1)

display_available(all_years)

filtered = all_years.merge(
    filtered_series[["Series Name"]], how="inner", on=["Series Name"]
)
filtered = filtered.merge(
    filtered_countries[["Country Name"]], how="inner", on=["Country Name"]
)

display_available(filtered)


# filter data and export

data = pd.read_csv("./data/worldbank/2022-2000_worldbank_data.csv")
keys = ["Country Name", "Country Code", "Series Name", "Series Code", *YEARS]
data = data[keys]

data = data.merge(filtered_series[["Series Name"]], how="inner", on=["Series Name"])
data = data.merge(
    filtered_countries[["Country Name"]], how="inner", on=["Country Name"]
)

data.to_csv("./data/worldbank/2022-2003_worldbank_filtered.csv")
