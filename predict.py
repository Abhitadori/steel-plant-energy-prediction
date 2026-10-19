"""
Predict next-interval energy usage (kWh) on unseen data.

Usage:
    python predict.py new_data.csv                 # writes predictions.csv
    python predict.py new_data.csv my_output.csv   # custom output name

Input CSV: same columns as train.csv (the target column
`next_interval_Usage_kWh` is optional and ignored if present).
Requires steel_energy_model.joblib (created by the notebook) in this folder.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

TARGET = "next_interval_Usage_kWh"
MODEL_PATH = Path("steel_energy_model.joblib")


def make_features(data):
    """Identical to the notebook: time + cyclical features from the current row."""
    x = data.drop(columns=[TARGET], errors="ignore").copy()
    dt = pd.to_datetime(x.pop("date"), errors="coerce")

    x["hour"] = dt.dt.hour
    x["minute"] = dt.dt.minute
    x["day_of_week_from_date"] = dt.dt.dayofweek
    x["month"] = dt.dt.month
    x["day_of_year"] = dt.dt.dayofyear

    minute_of_day = x["hour"] * 60 + x["minute"]
    x["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    x["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    x["year_sin"] = np.sin(2 * np.pi * x["day_of_year"] / 365.25)
    x["year_cos"] = np.cos(2 * np.pi * x["day_of_year"] / 365.25)
    return x


def predict(input_path, output_path="predictions.csv"):
    model = joblib.load(MODEL_PATH)

    raw = pd.read_csv(input_path)
    if "date" not in raw.columns:
        raise ValueError("Input file needs a 'date' column.")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M", errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        print(f"Warning: {n_bad} rows have unparseable dates; their time features will be missing.")

    X = make_features(df)

    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing columns the model needs: {missing}")
    X = X[expected]  # same columns, same order as training

    out = raw.copy()
    out["predicted_next_interval_Usage_kWh"] = model.predict(X)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} predictions to {output_path}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <input.csv> [output.csv]")
    predict(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "predictions.csv")
