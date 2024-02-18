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

        worldbank = pd.read_csv(WORLDBANK)
        antibiotics = pd.read_csv(ANTIBIOTICS)

    def __len__(self): ...
