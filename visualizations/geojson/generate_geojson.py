import json
import pandas as pd

PRED = "Predicted Category"
NAME = "Country Name"
CODE = "Country Code"

predicted_countries = pd.read_csv("prediction/results/predicted_categories.csv")
country_code = pd.read_csv("data/worldbank/links/Country_Name_Country_Code.csv")
country_code = country_code[[NAME, CODE]]

predicted_countries = predicted_countries.merge(country_code, how="inner", on=NAME)
predicted_countries = predicted_countries[[CODE, PRED]]

with open("visualizations/geojson/countries.geo.json") as f:
    raw_geojson = json.load(f)

for feature in raw_geojson["features"]:
    country_code = feature["id"]

    try:
        prediction = predicted_countries.loc[predicted_countries[CODE] == country_code][
            PRED
        ].values[0]
        prediction = int(prediction)
    except IndexError:
        prediction = -1

    feature["properties"]["prediction"] = prediction

out = json.dumps(dict(raw_geojson))
js = f"let pred_geojson = {out};"

with open("docs/predictions.js", "w+") as f:
    f.write(js)
