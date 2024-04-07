# Datasets Processing

## Worldbank

Manually remove last few rows with date for all datasets

### `merge_tables.py`

Sources
- World bank (world developement indicators) `raw_worlddev`
    - combine years 2000 to 2022
- World bank (health indicators) `raw_health`
    - change column order to `Country Name`, `Country Code`, `Series Name`, `Series Code`
    - combine years 2000 to 2022

Combine world_dev and health

Export to `2022-2000_worldbank_data.csv`


### `normalize.py`

Normalize every indicator by longitude

Export to `2022-2000_worldbank_normalized.csv`


## Antibiotics

Create link files manually
- `ATC level 3 class`
- `Location` to `Country Name`

### `transform_compatible.py`

- Merge `Location` to `Country Name` for compatibility

### `normalize.py`

Normalize by every antibiotic category

Export to `2022-2000_worldbank_normalized.csv`
