import pandas as pd

predicted_categories = pd.read_csv(
    "./prediction/results/2003-2022_predicted_categories.csv"
)
predicted_categories = predicted_categories[
    ["Country Name", "Year", "Predicted Category"]
]

country_category = predicted_categories.groupby(["Country Name"])[
    "Predicted Category"
].max()

country_category.to_csv("./prediction/results/predicted_categories.csv")
...
