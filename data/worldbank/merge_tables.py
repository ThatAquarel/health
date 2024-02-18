import functools
import pandas as pd

RAW_WORLDDEV = [
    "./data/worldbank/raw_worlddev/2022-2018/6dc94099-4e41-45c8-8912-574bea187d7f_Data.csv",
    "./data/worldbank/raw_worlddev/2017-2013/47ff21b6-e875-4d3a-b10f-f68a4dce0382_Data.csv",
    "./data/worldbank/raw_worlddev/2012-2008/3a0480ca-afb4-4a43-bc27-376bf22ef49b_Data.csv",
    "./data/worldbank/raw_worlddev/2007-2003/30a79e0f-128a-4512-a1fc-faccb5c043bb_Data.csv",
]

RAW_HEALTH = [
    "./data/worldbank/raw_health/2022-2012/813f407b-0ba3-402a-8fed-0ec3a80639b4_Data.csv",
    "./data/worldbank/raw_health/2011-2000/b42c4f6f-717c-412b-ade7-f70353593af7_Data.csv",
]

# read both datasets
worlddev_tables = [pd.read_csv(path) for path in RAW_WORLDDEV]
health_tables = [pd.read_csv(path) for path in RAW_HEALTH]

# put country as first col, series as second col
for i in range(len(health_tables)):
    cols = health_tables[i].columns.values
    new_cols = ["Country Name", "Country Code", "Series Name", "Series Code", *cols[4:]]

    health_tables[i] = health_tables[i][new_cols].sort_values(
        ["Country Name", "Series Name"]
    )

# combine worlddev
worlddev = functools.reduce(lambda a, b: a.merge(b), worlddev_tables)

# combine health
health = functools.reduce(lambda a, b: a.merge(b), health_tables)


out = worlddev.merge(health, how="outer")

years = [f"{i} [YR{i}]" for i in range(2000, 2023)]
for year in years:
    out.loc[out[year] == "..", year] = ""
    out[year] = pd.to_numeric(out[year])

out.to_csv("./data/worldbank/2022-2000_worldbank_data.csv")
