import pandas as pd

raw = pd.read_csv(
    "./data/antibiotics/raw/2018-2000_antiobiotic_consumption_estimates_atc3.csv"
)

link = pd.read_csv("./data/antibiotics/links/Location_Country_Name.csv")
link = link.dropna()


...
