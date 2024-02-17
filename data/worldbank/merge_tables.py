import functools
import pandas as pd

RAW = [
    "./data/worldbank/raw/2022-2018/6dc94099-4e41-45c8-8912-574bea187d7f_Data.csv",
    "./data/worldbank/raw/2017-2013/47ff21b6-e875-4d3a-b10f-f68a4dce0382_Data.csv",
    "./data/worldbank/raw/2012-2008/3a0480ca-afb4-4a43-bc27-376bf22ef49b_Data.csv",
    "./data/worldbank/raw/2007-2003/30a79e0f-128a-4512-a1fc-faccb5c043bb_Data.csv",
]

tables = [pd.read_csv(path) for path in RAW]

out = functools.reduce(lambda a, b: a.merge(b), tables)

years = [f"{i} [YR{i}]" for i in range(2003, 2023)]
for year in years:
    out.loc[out[year] == "..", year] = ""
    out[year] = pd.to_numeric(out[year])

out.to_csv("./data/worldbank/2022-2003_worldbank_data.csv")
