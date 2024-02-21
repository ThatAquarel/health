import pandas as pd
from tqdm import tqdm

ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_atc3.csv"

ATC3 = "ATC level 3 class"
CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"

antibiotics = pd.read_csv(ANTIBIOTICS)
antibiotics = antibiotics[
    ["Country Name", 
    "Year", ATC3, CONSUMPTION]
]

for drug in tqdm(antibiotics[ATC3].drop_duplicates()):
    x = antibiotics.loc[antibiotics[ATC3] == drug, CONSUMPTION]
    antibiotics.loc[antibiotics[ATC3] == drug, CONSUMPTION] = (x-x.mean())/x.std()

antibiotics.to_csv("./data/antibiotics/2022-2000_antibiotic_normalized.csv")
