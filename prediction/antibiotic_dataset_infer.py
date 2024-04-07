import torch
import itertools

import pandas as pd

from torch.utils.data import Dataset

torch.set_default_device("cuda")

WORLDBANK = "./data/worldbank/2022-2000_worldbank_normalized.csv"
ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_normalized.csv"


N_INDICATOR = 1683


class AntibioticDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._worldbank = pd.read_csv(WORLDBANK)
        self._prep_worldbank(2003, 2023)

        self._antibiotics = pd.read_csv(ANTIBIOTICS)
        self._prep_antiobiotics("J01C-Penicillins")

        self._build_db_idx()
        self._build_db()

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

    def _build_db_idx(self):
        a = self._antibiotics[["Country Name"]].drop_duplicates()
        b = self._worldbank[["Country Name"]].drop_duplicates()
        self.ax1 = a.merge(b)["Country Name"]

        self.ax2 = self._worldbank[["Year"]].drop_duplicates()["Year"]

        self._db_idx = list(itertools.product(self.ax1, self.ax2))

    def _build_db(self):
        self._worldbank = self._worldbank[
            self._worldbank["Country Name"].isin(self.ax1)
            & self._worldbank["Year"].isin(self.ax2)
        ]

        self.db_x = torch.zeros(len(self), N_INDICATOR)

        combined = pd.pivot_table(
            self._worldbank,
            values="Indicator",
            index=["Country Name", "Year"],
            columns=["Series Name"],
        ).reset_index()

        cases = combined[["Country Name", "Year"]]
        cases.to_csv("./prediction/db_x_infer_cases.csv")

        self.db_x = torch.tensor(
            combined[combined.columns.values[2:]].to_numpy()
        ).float()

    def save(self):
        torch.save(self.db_x, f"./prediction/db_x_infer.pt")

    def __len__(self):
        return len(self._db_idx)


a = AntibioticDataset()
a.save()
