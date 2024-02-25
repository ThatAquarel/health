import torch
import itertools

import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader

WORLDBANK = "./data/worldbank/2022-2000_worldbank_normalized.csv"
ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_normalized.csv"


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train

        self._worldbank = pd.read_csv(WORLDBANK)
        self._prep_worldbank(2003, 2019)
        self._test_year = 2018

        self._antibiotics = pd.read_csv(ANTIBIOTICS)
        self._prep_antiobiotics("J01C-Penicillins")

        self._cut_bins(5)

        self._build_db()
        # print(np.unique(self._antibiotics[self.CONSUMPTION], return_counts=True))
        # print(len(self))
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
        bins = list(range(n_bins))

        self._antibiotics.loc[:, self.CONSUMPTION] = pd.cut(
            self._antibiotics[self.CONSUMPTION], n_bins, labels=bins
        )

        self._one_hot = torch.zeros((n_bins, n_bins))
        self._one_hot[bins, bins] = 1

    def _build_db(self):
        a = self._antibiotics[["Country Name"]].drop_duplicates()
        b = self._worldbank[["Country Name"]].drop_duplicates()
        ax1 = a.merge(b)["Country Name"]

        a = self._antibiotics[["Year"]].drop_duplicates()
        b = self._worldbank[["Year"]].drop_duplicates()
        ax2 = a.merge(b)["Year"] if self.train else [self._test_year]

        self._db_idx = list(itertools.product(ax1, ax2))

    def __len__(self):
        return len(self._db_idx)

    def __getitem__(self, idx):
        ax1, ax2 = self._db_idx[idx]

        a_country = self._antibiotics["Country Name"] == ax1
        a_year = self._antibiotics["Year"] == ax2
        (c,) = self._antibiotics.loc[a_country & a_year, self.CONSUMPTION].values
        y = self._one_hot[c]

        b_country = self._worldbank["Country Name"] == ax1
        b_year = self._worldbank["Year"] == ax2
        i = self._worldbank.loc[b_country & b_year, "Indicator"].values
        x = torch.tensor(i)

        return (x, y)


train_dataloader = DataLoader(AntibioticDataset(), batch_size=64)
