import pandas as pd

a = pd.read_csv("./data/worldbank/2022-2018_worldbank_data.csv")
b = pd.read_csv("./data/worldbank/2017-2013_worldbank_data.csv")
c = pd.read_csv("./data/worldbank/2012-2008_worldbank_data.csv")
d = pd.read_csv("./data/worldbank/2007-2003_worldbank_data.csv")

nca = a.filter(items=["Country Name", "Country Code"]).drop_duplicates()
ncb = b.filter(items=["Country Name", "Country Code"]).drop_duplicates()
ncc = c.filter(items=["Country Name", "Country Code"]).drop_duplicates()
ncd = d.filter(items=["Country Name", "Country Code"]).drop_duplicates()

print(nca.equals(ncb))
print(nca.equals(ncc))
print(nca.equals(ncd))

print(ncb.equals(ncc))
print(ncb.equals(ncd))

print(ncc.equals(ncd))

...
