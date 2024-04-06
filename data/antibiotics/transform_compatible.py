import pandas as pd

raw = pd.read_csv(
    "./data/antibiotics/raw/2018-2000_total_antibiotic_consumption_estimates_raw.csv"
)

link = pd.read_csv("./data/antibiotics/links/Location_Country_Name.csv")
link = link.dropna()
link = link[["Location", "Country Name"]]

raw = raw.merge(link, on="Location")
raw = raw[
    [
        "Country Name",
        "Year",
        "Antibiotic consumption (DDD/1,000/day)",
    ]
]

raw.to_csv("./data/antibiotics/2018-2000_antibiotic_total.csv")
