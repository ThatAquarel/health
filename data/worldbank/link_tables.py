import pandas as pd

RAW = [
    "./data/worldbank/raw/2022-2018/6dc94099-4e41-45c8-8912-574bea187d7f_Data.csv",
    "./data/worldbank/raw/2017-2013/47ff21b6-e875-4d3a-b10f-f68a4dce0382_Data.csv",
    "./data/worldbank/raw/2012-2008/3a0480ca-afb4-4a43-bc27-376bf22ef49b_Data.csv",
    "./data/worldbank/raw/2007-2003/30a79e0f-128a-4512-a1fc-faccb5c043bb_Data.csv",
]

LINKS = [["Country Name", "Country Code"], ["Series Name", "Series Code"]]
OUT = [
    "./data/worldbank/Country_Name_Country_Code.csv",
    "./data/worldbank/Series_Name_Series_Code.csv",
]

tables = [pd.read_csv(path) for path in RAW]


for link, out in zip(LINKS, OUT):
    filtered_tables = [table.filter(items=link).drop_duplicates() for table in tables]

    for i in range(len(filtered_tables) - 1):
        assert filtered_tables[i].equals(filtered_tables[i + 1])

    filtered_tables[0].to_csv(out)
