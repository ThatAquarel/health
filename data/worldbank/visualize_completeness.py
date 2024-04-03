import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("./data/worldbank/2022-2000_worldbank_data.csv")

YEARS = [f"{i} [YR{i}]" for i in range(2003, 2023)]

# a = data.copy()
# a = a[["Country Name", "Series Name", *YEARS]]
# a.loc[:, YEARS] = pd.isna(a[YEARS])
# a = a.groupby(by=["Series Name"]).sum()

for YEAR in YEARS:
    columns = ["Country Name", "Series Name", YEAR]

    current = data[columns]
    current.loc[:, YEAR] = pd.isna(current[YEAR]) * 1

    completeness = current.pivot_table(YEAR, "Country Name", "Series Name")

    print(completeness.shape)
    print(YEAR)

    plt.imshow(completeness.to_numpy())
    plt.show()
