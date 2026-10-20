from pathlib import Path
import joblib
from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Any

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Machine learning API for detecting fraudulent transactions using LightGBM.",
    version="1.0.0"
)


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load trained model and preprocessing transformer
print("Loading trained LightGBM model...")

model = joblib.load(MODEL_DIR / "lightgbm_model.joblib")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

print("LightGBM model and preprocessor loaded successfully!")

class TransactionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Transaction features using original dataset column names"
    )



@app.get("/")
def home():
    return {
        "message": "Financial Fraud Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fraud-detection-api"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "LightGBM",
        "status": "loaded",
        "input_features": len(preprocessor.feature_names_in_),
        "transformed_features": len(
            preprocessor.get_feature_names_out()
        ),
        "experimental_threshold": 0.80
    }


@app.post("/predict")
def predict_fraud(request: TransactionRequest):
    try:
        expected_features = list(preprocessor.feature_names_in_)

        # Convert request to a single-row DataFrame
        transaction = pd.DataFrame([request.features])

        # Reject unexpected feature names
        unknown_features = set(transaction.columns) - set(expected_features)

        if unknown_features:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown features: {sorted(unknown_features)}"
            )

        # Add missing features as null values

        # Restore categorical types and missing-value representation
        categorical_columns = list(preprocessor.transformers_[1][2])

        for column in categorical_columns:
            transaction[column] = transaction[column].astype("object")
            transaction[column] = transaction[column].where(
                transaction[column].notna(),
                np.nan
            )


        # Restore categorical data types used during training
        categorical_columns = preprocessor.transformers_[1][2]

        for column in categorical_columns:
            transaction[column] = transaction[column].astype("object")


        # Apply the fitted training preprocessor
        X = preprocessor.transform(transaction)

        # Predict fraud score
        fraud_score = float(model.predict_proba(X)[0, 1])

        threshold = 0.80
        prediction = int(fraud_score >= threshold)

        return {
            "prediction": prediction,
            "classification": "Fraud" if prediction else "Legitimate",
            "fraud_score": round(fraud_score, 6),
            "threshold": threshold
        }

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid transaction data: {exc}"
        )
