import torch

import numpy as np
import pandas as pd

torch.set_default_device("cuda:0")

CASES_IDENTIFIERS = ["Country Name", "Year"]

# worldbank dataset
worldbank = pd.read_csv("./data/worldbank/2022-2003_worldbank_normalized.csv")
worldbank = worldbank[[*CASES_IDENTIFIERS, "Series Name", "Indicator"]]
worldbank.loc[:, ["Indicator"]] = worldbank["Indicator"].astype("float32")
worldbank = worldbank.fillna(0)

# antibiotics dataset
antibiotics = pd.read_csv("./data/antibiotics/2018-2000_antibiotic_normalized.csv")
CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"
antibiotics = antibiotics[[*CASES_IDENTIFIERS, CONSUMPTION]]

# equal width binning into N_BINS categories
N_BINS = 5
bins = list(range(N_BINS))
antibiotics.loc[:, "Category"] = pd.cut(antibiotics[CONSUMPTION], N_BINS, labels=bins)
antibiotics = antibiotics[[*CASES_IDENTIFIERS, "Category"]]
_, counts = np.unique(antibiotics["Category"], return_counts=True)
print(f"class_weights: {counts.sum() / counts}")

# merge antibiotics and worldbank
pivoted_worldbank = pd.pivot_table(
    worldbank,
    values="Indicator",
    index=CASES_IDENTIFIERS,
    columns=["Series Name"],
)
merged = pivoted_worldbank.merge(antibiotics, how="left", on=CASES_IDENTIFIERS)
merged = merged.fillna(0)

# determine cases and axes
unique_cases = merged[CASES_IDENTIFIERS].drop_duplicates()
N_CASES = len(unique_cases)

unique_indicators = merged.columns.values[2:-1]
N_INDICATORS = len(unique_indicators)


# build matrices
def get_idx(cases, start_year, end_year):
    condition = (cases["Year"] >= start_year) & (cases["Year"] <= end_year)
    return cases[condition].index.values


# build output variable
ONE_HOT_LUT = torch.zeros((N_BINS, N_BINS)).float()
ONE_HOT_LUT[bins, bins] = 1.0

y = torch.zeros((N_CASES, N_BINS))
y_idx = merged["Category"]
y[range(N_CASES), y_idx] = 1.0
assert (y.argmax(axis=1).cpu().numpy() == y_idx.to_numpy()).all()

# y_2003-2017_train.pt
torch.save(y[get_idx(unique_cases, 2003, 2017), :], "./prediction/y_2003-2017_train.pt")

# y_2018_test.pt
torch.save(y[get_idx(unique_cases, 2018, 2018), :], "./prediction/y_2018_test.pt")


# build input variable
x = torch.tensor(merged[merged.columns.values[2:-1]].to_numpy()).float()

# x_2003-2017_train.pt
torch.save(x[get_idx(unique_cases, 2003, 2017), :], "./prediction/x_2003-2017_train.pt")

# x_2018_test.pt
torch.save(x[get_idx(unique_cases, 2018, 2018), :], "./prediction/x_2018_test.pt")

# x_2003-2022_infer.pt and x_2003-2022_infer_cases.csv
torch.save(x[get_idx(unique_cases, 2003, 2022), :], "./prediction/x_2003-2022_infer.pt")
unique_cases.to_csv("./prediction/x_2003-2022_infer_cases.csv")
factors = pd.DataFrame({"Series Name": merged.columns.values[2:-1]})
factors.to_csv("./prediction/x_2003-2022_infer_factors.csv")
