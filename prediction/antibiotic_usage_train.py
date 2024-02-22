import torch
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
        self._prep_worldbank(2000, 2019)

        self._antibiotics = pd.read_csv(ANTIBIOTICS)
        self._prep_antiobiotics("J01C-Penicillins")

        ...

    def _prep_worldbank(self, start_year, end_year):
        lower = self._worldbank["Year"] >= start_year
        upper = self._worldbank["Year"] < end_year

        self._worldbank = self._worldbank[lower & upper]
        self._worldbank = self._worldbank[["Country Name", "Year", "Series Name", "Indicator"]]

    def _prep_antiobiotics(self, atc_3_class):
        self._antibiotics = self._antibiotics[self._antibiotics["ATC level 3 class"] == atc_3_class]
        self._antibiotics = self._antibiotics[["Country Name", "Year", "Antibiotic consumption (DDD/1,000/day)"]]

    def __len__(self): ...


train_dataloader = DataLoader(AntibioticDataset(), batch_size=64)
