import torch
import itertools

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
index = merged[["Country Name", "Year"]].drop_duplicates()
index = index.reset_index(drop=True)
N_CASES = len(index)
N_INDICATORS = len(merged[["Series Name"]].drop_duplicates())

# one hot encoding for categories
ONE_HOT_LUT = torch.zeros((N_BINS,N_BINS)).float()
ONE_HOT_LUT[bins, bins] = 1.0

# output var (2003-2018)
y = torch.zeros((N_CASES, N_BINS))
y_idx = antibiotics.merge(index, how="inner", on=["Country Name", "Year"])[CONSUMPTION]
y[range(N_CASES), y_idx] = 1.0
assert (y.argmax(axis=1).numpy() == y_idx.to_numpy()).all()

# save output var
def get_y_idx(start_year, end_year):
    condition = (index["Year"] >= start_year) & (index["Year"] <= end_year)
    return index[condition].index.values

# y_2003-2017_train.pt
torch.save(y[get_y_idx(2003, 2017), :], "./prediction/y_2003-2017_train.pt")

# y_2018_test.pt
torch.save(y[get_y_idx(2018, 2018), :], "./prediction/y_2018_test.pt")

...

class AntibioticDataset(Dataset):
    def __init__(self, train=True, end_year=2019) -> None:
        super().__init__()

        self.train = train

        self._worldbank = pd.read_csv(WORLDBANK)
        self._prep_worldbank(2003, end_year)
        self._test_year = 2018

        self._antibiotics = pd.read_csv(ANTIBIOTICS)
        self._prep_antiobiotics("J01C-Penicillins")

        self._cut_bins(5)

        self._build_db_idx()
        self._build_db()
        # print(np.unique(self._antibiotics[self.CONSUMPTION], return_counts=True))
        # print(len(self))

        self._worldbank[["Indicator"]] = self._worldbank[["Indicator"]].astype(
            "float32"
        )
        ...

    def _prep_worldbank(self, start_year, end_year):
        lower = self._worldbank["Year"] >= start_year
        upper = self._worldbank["Year"] < end_year

        self._worldbank = self._worldbank[lower & upper]
        self._worldbank = self._worldbank[
            ["Country Name", "Year", "Series Name", "Indicator"]
        ]

        self._worldbank = self._worldbank.fillna(0)

    CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"

    def _prep_antiobiotics(self, atc_3_class):
        self._antibiotics = self._antibiotics[
            self._antibiotics["ATC level 3 class"] == atc_3_class
        ]
        self._antibiotics = self._antibiotics[
            ["Country Name", "Year", self.CONSUMPTION]
        ]

    def _cut_bins(self, n_bins):
        self.n_bins = n_bins

        bins = list(range(n_bins))

        self._antibiotics.loc[:, self.CONSUMPTION] = pd.cut(
            self._antibiotics[self.CONSUMPTION], n_bins, labels=bins
        )
        # to print z-score intervals
        # print(pd.cut(self._antibiotics[self.CONSUMPTION], n_bins))
        # convert to normal
        #
        # a = np.array([[-1.274, 0.31],
        # [0.31, 1.885],
        # [1.885, 3.46],
        # [3.46, 5.036],
        # [5.036, 6.611]])
        #
        # a * std + mean
        # a * 3.2099593064292544 + 4.762091049780758

        self._one_hot = torch.zeros((n_bins, n_bins)).float()
        self._one_hot[bins, bins] = 1

    def _build_db_idx(self):
        a = self._antibiotics[["Country Name"]].drop_duplicates()
        b = self._worldbank[["Country Name"]].drop_duplicates()
        self.ax1 = a.merge(b)["Country Name"]

        a = self._antibiotics[["Year"]].drop_duplicates()
        b = self._worldbank[["Year"]].drop_duplicates()
        self.ax2 = a.merge(b)["Year"] if self.train else [self._test_year]

        self._db_idx = list(itertools.product(self.ax1, self.ax2))

    def _build_db(self):
        self._antibiotics = self._antibiotics[
            self._antibiotics["Country Name"].isin(self.ax1)
            & self._antibiotics["Year"].isin(self.ax2)
        ]
        self._worldbank = self._worldbank[
            self._worldbank["Country Name"].isin(self.ax1)
            & self._worldbank["Year"].isin(self.ax2)
        ]

        self.db_x = torch.zeros(len(self), N_INDICATOR)
        self.db_y = torch.zeros(len(self), self.n_bins)

        pivoted_worldbank = pd.pivot_table(
            self._worldbank,
            values="Indicator",
            index=["Country Name", "Year"],
            columns=["Series Name"],
        )

        combined = pivoted_worldbank.merge(
            self._antibiotics.drop_duplicates(subset=["Country Name", "Year"]),
            on=["Country Name", "Year"],
        )

        self.db_x = torch.tensor(
            combined[combined.columns.values[2:-1]].to_numpy()
        ).float()
        self.db_y[
            list(range(len(self))), combined[combined.columns.values[-1]].to_numpy()
        ] = 1.0
        self.db_y = self.db_y.float()

    def save(self):
        torch.save(self.db_x, f"./prediction/db_x{"" if self.train else "_test"}.pt")
        torch.save(self.db_y, f"./prediction/db_y{"" if self.train else "_test"}.pt")

    def __len__(self):
        return len(self._db_idx)

    def __getitem__(self, idx):
        return (self.db_x[idx], self.db_y[idx])


a = AntibioticDataset(train=True, end_year=2018)
a.save()

a = AntibioticDataset(train=False, end_year=2019)
a.save()
