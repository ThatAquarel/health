import pandas as pd

ANTIBIOTICS = "./data/antibiotics/2018-2000_antibiotic_total.csv"
CONSUMPTION = "Antibiotic consumption (DDD/1,000/day)"

antibiotics = pd.read_csv(ANTIBIOTICS)
antibiotics = antibiotics[["Country Name", "Year", CONSUMPTION]]

x = antibiotics[CONSUMPTION]
antibiotics[CONSUMPTION] = (x - x.mean()) / x.std()

antibiotics.to_csv("./data/antibiotics/2018-2000_antibiotic_normalized.csv")
