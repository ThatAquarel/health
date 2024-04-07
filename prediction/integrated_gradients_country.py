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
db_x_infer = torch.load("./prediction/x_2003-2022_infer.pt")

db_x_infer_cases = pd.read_csv("./prediction/x_2003-2022_infer_cases.csv")
db_x_infer_cases = db_x_infer_cases[["Country Name", "Year"]]
countries = db_x_infer_cases[["Country Name"]].drop_duplicates()["Country Name"]

factors = pd.read_csv("./prediction/x_2003-2022_infer_factors.csv")
factors = factors[["Series Name"]]

ig = IntegratedGradients(model)

attributions_country = np.zeros((len(factors), len(countries)), dtype=np.float32)
for i, country in enumerate(countries):
    country_idx = db_x_infer_cases.loc[
        db_x_infer_cases["Country Name"] == country
    ].index

    db_x_infer_country = db_x_infer[country_idx, :]
    baseline = torch.zeros(db_x_infer_country.shape)

    attributions, approximation_error = ig.attribute(
        db_x_infer_country,
        baselines=baseline,
        # method="gausslegendre",
        return_convergence_delta=True,
        target=level,
    )

    attributions_country[:, i] = np.mean(attributions.cpu().numpy(), axis=0)

categories = pd.read_csv("./data/worldbank/links/Series_Name_Category.csv")
categories = categories[["Category", "Series Name"]]
attributions = pd.read_csv("./prediction/results/ordered_factors_2003_2022_high.csv")
attributions = attributions[["Series Name", "Attribution"]]

attributions_df = pd.DataFrame(attributions_country, columns=countries)
factors = pd.concat([factors, attributions_df], axis=1)
factors = factors.merge(categories, how="inner", on="Series Name")
factors = factors[["Category", "Series Name", *countries.to_list()]]

positive = attributions.copy()
negative = attributions.copy()

attributions["Attribution_abs"] = attributions[["Attribution"]].abs()
attributions = attributions.sort_values(by=["Attribution_abs"])
attributions = attributions[["Series Name", "Attribution"]]

positive = positive.sort_values(by=["Attribution"])
negative = negative.sort_values(by=["Attribution"], ascending=False)

import matplotlib.pyplot as plt

plt.hist(attributions[["Attribution"]])
plt.show()
# generate results/factor_attribution_distribution.png

for n in [10, 25, 50, 100]:
    attributions.tail(n).to_csv(f"./prediction/results/top{n}_factors.csv")

for n in [10, 20, 50, 100]:
    each = n // 2

    top_positive = positive.tail(each)
    top_negative = negative.tail(each)

    pd.merge(top_positive, top_negative, how="outer").to_csv(
        f"./prediction/results/top{n}_balanced_factors.csv"
    )
    ...

filtered_factors = attributions.tail(100)[["Series Name"]]
factors = factors.merge(filtered_factors, how="inner", on="Series Name")

factors.to_csv(f"./prediction/results/ordered_factors_2003_2022_{label}_countries.csv")
