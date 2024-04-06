import torch

import numpy as np
import pandas as pd

# worldbank dataset
worldbank = pd.read_csv("./data/worldbank/2022-2003_worldbank_normalized.csv")
worldbank = worldbank[["Country Name", "Series Name", "Year", "Indicator"]]
worldbank.loc[:, ["Indicator"]] = worldbank["Indicator"].astype("float32")
worldbank = worldbank.fillna(0)

# antibiotics dataset
antibiotics = pd.read_csv("./data/antibiotics/2018-2000_antibiotic_normalized.csv")
CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"
antibiotics = antibiotics[["Country Name", "Year", CONSUMPTION]]

# equal width binning into N_BINS categories
N_BINS = 5
bins = list(range(N_BINS))
antibiotics.loc[:, CONSUMPTION] = pd.cut(antibiotics[CONSUMPTION], N_BINS, labels=bins)
_, counts = np.unique(antibiotics[CONSUMPTION], return_counts=True)
print(f"class_weights: {counts.sum() / counts}")

# merge worldbank and antibiotics (2003-2018)
merged = antibiotics.merge(worldbank, how="inner", on=["Country Name", "Year"])
y_cases = merged[["Country Name", "Year"]].drop_duplicates()
y_cases = y_cases.reset_index(drop=True)
N_CASES = len(y_cases)
N_INDICATORS = len(merged[["Series Name"]].drop_duplicates())

# one hot encoding for categories
ONE_HOT_LUT = torch.zeros((N_BINS, N_BINS)).float()
ONE_HOT_LUT[bins, bins] = 1.0

# output var (2003-2018)
y = torch.zeros((N_CASES, N_BINS))
y_idx = antibiotics.merge(y_cases, how="inner", on=["Country Name", "Year"])[
    CONSUMPTION
]
y[range(N_CASES), y_idx] = 1.0
assert (y.argmax(axis=1).numpy() == y_idx.to_numpy()).all()


def get_idx(cases, start_year, end_year):
    condition = (cases["Year"] >= start_year) & (cases["Year"] <= end_year)
    return cases[condition].index.values


# y_2003-2017_train.pt
torch.save(y[get_idx(y_cases, 2003, 2017), :], "./prediction/y_2003-2017_train.pt")

# y_2018_test.pt
torch.save(y[get_idx(y_cases, 2018, 2018), :], "./prediction/y_2018_test.pt")

# input var (2003-2022)
x_cases = worldbank[["Country Name", "Year"]].drop_duplicates()
x_cases = x_cases.reset_index(drop=True)
N_CASES = len(x_cases)

pivoted_worldbank = pd.pivot_table(
    worldbank,
    values="Indicator",
    index=["Country Name", "Year"],
    columns=["Series Name"],
)
pivoted_worldbank = pivoted_worldbank.merge(
    x_cases, how="inner", on=["Country Name", "Year"]
)
x = torch.tensor(
    pivoted_worldbank[pivoted_worldbank.columns.values[2:]].to_numpy()
).float()

# x_2003-2017_train.pt
torch.save(x[get_idx(x_cases, 2003, 2017), :], "./prediction/x_2003-2017_train.pt")

# x_2018_test.pt
torch.save(x[get_idx(x_cases, 2018, 2018), :], "./prediction/x_2018_test.pt")

# x_2003-2022_infer.pt and x_2003-2022_infer_cases.csv
torch.save(x[get_idx(x_cases, 2003, 2022), :], "./prediction/x_2003-2022_infer.pt")
x_cases.to_csv("./prediction/x_2003-2022_infer_cases.csv")
