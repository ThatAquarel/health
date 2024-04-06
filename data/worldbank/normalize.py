import pandas as pd
from tqdm import tqdm

WORLDBANK = "./data/worldbank/2022-2003_worldbank_filtered.csv"

worldbank = pd.read_csv(WORLDBANK)

keys = ["Country Name", "Country Code", "Series Name", "Series Code"]
years = {f"{i} [YR{i}]": f"{i}" for i in range(2003, 2023)}

worldbank = worldbank[[*keys, *list(years.keys())]]
worldbank = worldbank.rename(columns=years)

var_key = "Year"
val_key = "Indicator"
worldbank = pd.melt(
    worldbank,
    id_vars=keys,
    value_vars=list(years.values()),
    var_name=var_key,
    value_name=val_key,
)

worldbank = worldbank[["Country Name", "Series Name", "Year", "Indicator"]]

for indicator in tqdm(worldbank["Series Name"].drop_duplicates()):
    x = worldbank.loc[worldbank["Series Name"] == indicator, "Indicator"]
    worldbank.loc[worldbank["Series Name"] == indicator, "Indicator"] = (
        x - x.mean()
    ) / x.std()

worldbank.to_csv("./data/worldbank/2022-2003_worldbank_normalized.csv")
