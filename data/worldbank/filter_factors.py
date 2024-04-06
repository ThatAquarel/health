import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


#
a = data.copy()
a = a[["Country Name", "Series Name", *YEARS]]
a.loc[:, YEARS] = pd.isna(a[YEARS])
a = a.groupby(by=["Series Name"]).sum()
b = pd.melt(a.reset_index(), id_vars=["Series Name"], value_vars=YEARS, var_name="Year")
counts = b.groupby(by=["Series Name"]).sum()

...
counts.loc[counts["value"] <= 2000]
