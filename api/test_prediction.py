
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from src.preprocess import (
    load_data,
    merge_data,
    create_features,
    split_data,
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the saved model and preprocessor
model = joblib.load(BASE_DIR / "models/lightgbm_model.joblib")
preprocessor = joblib.load(BASE_DIR / "models/preprocessor.joblib")

# Recreate the validation dataset
transactions, identity = load_data()
df = merge_data(transactions, identity)
df = create_features(df, identity)

train_df, val_df = split_data(df)

# Select one real validation transaction
transaction = val_df.iloc[[0]]

expected_features = list(preprocessor.feature_names_in_)
X_raw = transaction[expected_features]

# Direct model prediction
X_processed = preprocessor.transform(X_raw)
direct_score = float(model.predict_proba(X_processed)[0, 1])

# Convert missing values to JSON-compatible nulls
features = X_raw.iloc[0].to_dict()

features = {
    key: (
        None
        if pd.isna(value)
        else value.item()
        if isinstance(value, np.generic)
        else value
    )
    for key, value in features.items()
}

# Send transaction to FastAPI
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": features},
    timeout=30
)

print("\nAPI status code:", response.status_code)
print("API response:", response.text)

response.raise_for_status()

api_score = response.json()["fraud_score"]


# Debug JSON serialization and preprocessing


print("\nDirect model score:", direct_score)
print("API model score:", api_score)

# API rounds scores to six decimal places
assert abs(direct_score - api_score) < 1e-6

print("\nSUCCESS: API and direct model predictions match!")
