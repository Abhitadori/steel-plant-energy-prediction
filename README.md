# Steel Plant Energy Forecasting (Next 15-Minute Interval)

A machine-learning model that predicts `next_interval_Usage_kWh`, the energy a steel plant will use in the **next 15-minute interval**, from the current interval's telemetry and the time of day/year.

## Project contents

| File | Purpose |
|---|---|
| `steel_energy_model.ipynb` | Main notebook: data cleaning, features, baseline, model, evaluation, saving |
| `train.csv` | Original labeled dataset (not modified) |
| `cleaned_train.csv` | Cleaned dataset produced by the notebook |
| `steel_energy_model.joblib` | Final trained model (created by running the notebook) |
| `predict.py` | Script that applies the saved model to new, unseen data |
| `pyproject.toml`, `uv.lock` | Project dependencies (used by `uv`) |
| `README.md` | This file |

## Data

- Source file: `train.csv`, 28,031 rows x 12 columns.
- Intervals: 15 minutes, starting 01/01/2018 00:15.
- `date` format: `day/month/year hour:minute`.
- Inputs include `Usage_kWh`, lagging/leading reactive power, `CO2(tCO2)`, power factors and other plant telemetry columns.
- Target: `next_interval_Usage_kWh`.

### Cleaning steps
1. Verify the required columns (`date`, `Usage_kWh`, target) exist.
2. Parse `date` into datetimes; values that cannot be parsed become `NaT`.
3. Drop rows with a missing date or missing target.
4. Sort chronologically and remove duplicate timestamps.
5. Save the result as `cleaned_train.csv`.

## Method

**Features** (built from the current row only, so no future data leaks into the model):
- Time features: hour, minute, day of week, month, day of year.
- Cyclical features: `hour_sin`, `hour_cos`, `year_sin`, `year_cos`, so that 23:45 is treated as close to 00:00 and 31 Dec as close to 1 Jan.
- All other telemetry columns from the dataset.

**Validation:** chronological 80/20 split. The model trains on the earliest 80% of rows and is validated on the latest 20%. Rows are never shuffled, because shuffling time-series data gives overly optimistic scores.

**Baseline:** persistence, i.e. the next interval's usage is assumed to equal the current `Usage_kWh`.

**Model:** a scikit-learn `Pipeline` containing
- median imputation for numeric columns,
- most-frequent imputation + one-hot encoding for any text columns,
- `HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, max_leaf_nodes=25, l2_regularization=1.0, random_state=42)`.

**Metrics:** MAE and RMSE in kWh (lower is better).

**Final model:** after validation, the pipeline is retrained on all cleaned rows and saved as `steel_energy_model.joblib`.

## Results

Validation set (latest 20% of the data):

| Model | MAE (kWh) | RMSE (kWh) |
|---|---|---|
| Persistence baseline | _fill in_ | _fill in_ |
| HistGradientBoosting | _fill in_ | _fill in_ |

(Copy these numbers from section 8 of the notebook after running it.)

## Setup and reproduction

Requires Python 3.9 or newer.

**Libraries:** pandas, numpy, scikit-learn, matplotlib, joblib, jupyter (ipykernel).

### Option A: using uv (matches `pyproject.toml` / `uv.lock`)

```bash
uv sync
uv run jupyter notebook steel_energy_model.ipynb
```

### Option B: using pip

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib joblib jupyter
jupyter notebook steel_energy_model.ipynb
```

### Run the notebook

1. Make sure `train.csv` is in the same folder as the notebook.
2. Choose **Restart & Run All**.
3. The notebook creates `cleaned_train.csv` and `steel_energy_model.joblib` in the same folder.

The random seed is fixed (`random_state=42`), so results are repeatable.

## Predict on unseen data

After running the notebook (so `steel_energy_model.joblib` exists):

```bash
python predict.py new_data.csv
```

- `new_data.csv` must have the same columns as `train.csv`, with dates in `day/month/year hour:minute` format. The target column is optional and ignored if present.
- The script builds the same features as the notebook and writes `predictions.csv` with an added `predicted_next_interval_Usage_kWh` column.
- To use a different output name: `python predict.py new_data.csv my_output.csv`.

From Python:

```python
from predict import predict
predictions = predict("new_data.csv", "predictions.csv")
```

## Notes and limitations

- The model uses only the current interval's readings plus time features. It does not use lagged values from earlier intervals; adding lag and rolling-window features is a natural next improvement.
- Scores come from the chronological validation split, not from a separate test set.
- If every date becomes `NaT`, reload the original CSV (the raw-loading cell in the notebook) before converting dates again, and check that the format string matches your file.
