import pandas as pd

# link tables for data and filtered
for x in ["filtered", "data"]:
    WORLDBANK = f"./data/worldbank/2022-2003_worldbank_{x}.csv"

    LINKS = [["Country Name", "Country Code"], ["Series Name", "Series Code"]]
    OUT = [
        f"./data/worldbank/links/{x}/Country_Name_Country_Code.csv",
        f"./data/worldbank/links/{x}/Series_Name_Series_Code.csv",
    ]

    worldbank = pd.read_csv(WORLDBANK)

    for link, out in zip(LINKS, OUT):
        worldbank.filter(items=link).drop_duplicates().to_csv(out)


# categories
series_categories = pd.read_csv("./data/worldbank/links/Series_Name_Category.csv")
series_categories = series_categories[["Category", "Series Name", "Series Code"]]

for category in series_categories["Category"].drop_duplicates():
    series_categories.loc[series_categories["Category"] == category].to_csv(
        f"./data/worldbank/links/category/{category}.csv"
    )

# used Country_Name
a = pd.read_csv("data/antibiotics/links/Location_Country_Name.csv")
a = a[["Country Name"]].drop_duplicates()

b = pd.read_csv("data/antibiotics/2018-2000_antibiotic_total.csv")
b = b[["Country Name"]].drop_duplicates()

c = pd.read_csv("data/worldbank/links/data/Country_Name_Country_Code.csv")
c = b[["Country Name"]].drop_duplicates()

c = a.merge(b, how="inner", on=["Country Name"]).merge(
    c, how="inner", on=["Country Name"]
)
c.to_csv("data/worldbank/links/Country_Name.csv")
