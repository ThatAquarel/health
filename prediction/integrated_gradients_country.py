import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients
from antibiotic_usage_train import AntibioticPredictor, MODEL


model = AntibioticPredictor()
model.load_state_dict(torch.load(MODEL))
model.eval()

# use ./prediction/db_x_infer.pt to generate
# ordered_factors_2003_2022_high_countries.csv

level, label = 4, "high"
db_x_infer = torch.load("./prediction/db_x_infer.pt")

db_x_infer_cases = pd.read_csv("./prediction/db_x_infer_cases.csv")
db_x_infer_cases = db_x_infer_cases[["Country Name", "Year"]]
countries = db_x_infer_cases[["Country Name"]].drop_duplicates()["Country Name"]
countries = countries.loc[countries != "World"]

factors = pd.read_csv("./data/worldbank/links/Series_Name_Series_Code.csv")
factors = factors[["Series Name", "Series Code"]]

ig = IntegratedGradients(model)

for i, country in enumerate(countries):
    country_idx = db_x_infer_cases.loc[
        db_x_infer_cases["Country Name"] == country
    ].index

    db_x_infer_country = db_x_infer[country_idx, :]
    baseline = torch.zeros(db_x_infer_country.shape)

    ...

    attributions, approximation_error = ig.attribute(
        db_x_infer_country,
        baselines=baseline,
        method="gausslegendre",
        return_convergence_delta=True,
        target=level,
    )

    importance = np.mean(attributions.cpu().numpy(), axis=0)
    factors.insert(2 + i, country, list(importance), True)

categories = pd.read_csv("./data/worldbank/links/Series_Name_Category.csv")
categories = categories[["Category", "Series Name"]]

factors = factors.merge(categories, how="inner", on="Series Name")
factors = factors[["Category", "Series Name", "Series Code", *countries.to_list()]]

factors.to_csv(f"./prediction/results/ordered_factors_2003_2022_{label}_countries.csv")
