import pandas as pd

WORLDBANK = "./data/worldbank/2022-2000_worldbank_data.csv"

LINKS = [["Country Name", "Country Code"], ["Series Name", "Series Code"]]
OUT = [
    "./data/worldbank/links/Country_Name_Country_Code.csv",
    "./data/worldbank/links/Series_Name_Series_Code.csv",
]

worldbank = pd.read_csv(WORLDBANK)

for link, out in zip(LINKS, OUT):
    worldbank.filter(items=link).drop_duplicates().to_csv(out)
