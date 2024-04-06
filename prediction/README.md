# Model Training

## Dataset formatting

### `antibiotic_dataset.py`

Generate
- `db_x_test.pt`
- `db_x.pt`
- `db_y_test.pt`
- `db_y.pt`

### `antibiotic_dataset_infer.py`

Generate (worldbank from 2003 to 2022)
- `db_x_infer.pt`

### `antibiotic_hyperopt_train.py`

Determine hyperparameters
Confirm neural network design of averaging in and out

### `antibiotic_usage_train.py`

Train model and save as `2024_02_28_AntitiobicPredictor.pt`

### `integrated_gradients.py`

Compute attributions for each factor and save
- `results/ordered_factors_2018_high.csv`
- `results/ordered_factors_2018_low.csv`
- `results/ordered_factors_2003_2018_high.csv`
- `results/ordered_factors_2003_2018_low.csv`
- `results/ordered_factors_2003_2022_high.csv`
- `results/ordered_factors_2003_2022_low.csv`

### `integrated_gradients_country.py`

Compute attributions for each factor and save
- `results/ordered_factors_2003_2022_high_countries.csv`
Remove "World"
Remove impertinent factors

Generate `results/factor_attribution_distribution.png`

Compute top_n important factors
- `results/top{n}_factors.csv`

### `antibiotic_infer.py`

- Generate accuracies at `./results/accuracy.md`
- Generate predicted categories at `results/2003-2022_predicted_categories.csv`

### `antibiotic_usage.md`

- `./data/antibiotics/normalize.py`
- `./prediction/antibiotic_dataset.py`


### `country_category.py`

Over all years, generate:
- `results/predicted_categories.csv`
Manually remove "World"

Copy to `results/countries_continents.csv`
Manually map to continents

Copy to `results/visualized_countries_continents.csv`
Remove non pertinent regions

## Results

Accuracy for testing 2018
Accuracy for training 2003-2017

Importance of factors in 2003-2018
Importance of factors in 2022

Estimates for 2003-2022
Actual values for 2003-2018

Usage intervals for antibiotic usage


