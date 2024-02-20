import torch
import pandas as pd

from torch import nn
from torch.utils.data import Dataset, DataLoader


WORLDBANK = "./data/worldbank/2022-2000_worldbank_data.csv"
ANTIBIOTICS = "./data/antibiotics/2018-2000_total_antibiotic_consumption_estimates.csv"


class AntibioticDataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.train = train

        self._worldbank = pd.read_csv(WORLDBANK)
        self._prep_worldbank(2000, 2019)

        self._antibiotics = pd.read_csv(ANTIBIOTICS)

    def _format_worldbank_year(self, year):
        return f"{year} [YR{year}]"

    def _prep_worldbank(self, start_year, end_year): 
        keys = ["Country Name", "Country Code", "Series Name", "Series Code"]
        years = [self._format_worldbank_year(i) for i in range(start_year, end_year)]
        year_key = "Year"

        self._worldbank = self._worldbank[[*keys, *years]]
        self._worldbank = pd.melt(self._worldbank, id_vars=keys, value_vars=years, var_name=year_key)


    def __len__(self): ...


train_dataloader = DataLoader(AntibioticDataset(), batch_size=64)

