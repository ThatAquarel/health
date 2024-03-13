# Model Training

## Dataset formatting

### `antibiotic_dataset.py`

Generate
- `db_x_test.py`
- `db_x.py`
- `db_y_test.py`
- `db_y.py`


### `antibiotic_hyperopt_train.py`

Determine hyperparameters
Confirm neural network design of averaging in and out

### `antibiotic_usage_train.py`

Train model and save as `2024_02_28_AntitiobicPredictor.pt`

### `integrated_gradients.py`

Compute attributions for each factor and save
- `results/ordered_factors_2018_high.csv`
- `results/ordered_factors_2018_low.csv`

## Results

Accuracy for testing

Accuracy for training

Importance of factors in 2018

Importance of factors in 2022