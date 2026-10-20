
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.preprocess import (
    load_data,
    merge_data,
    create_features,
    split_data,
)

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Financial Fraud Detection")
st.write(
    "Analyze transaction fraud risk using a trained "
    "LightGBM machine learning model."
)

st.info(
    "Portfolio demonstration using the IEEE-CIS dataset. "
    "Model scores are not calibrated fraud probabilities."
)


@st.cache_data(show_spinner="Loading validation transactions...")
def get_validation_data():
    transactions, identity = load_data()

    df = merge_data(transactions, identity)
    df = create_features(df, identity)

    _, val_df = split_data(df)

    return val_df


val_df = get_validation_data()

st.sidebar.header("Transaction Selection")

transaction_index = st.sidebar.number_input(
    "Validation transaction index",
    min_value=0,
    max_value=len(val_df) - 1,
    value=0,
    step=1
)

transaction = val_df.iloc[int(transaction_index)]

st.subheader("Selected Transaction")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Transaction Amount",
    f"${transaction['TransactionAmt']:,.2f}"
)

col2.metric(
    "Transaction ID",
    str(int(transaction["TransactionID"]))
)

col3.metric(
    "Actual Label",
    "Fraud" if transaction["isFraud"] == 1 else "Legitimate"
)

st.write("Transaction features")

st.dataframe(
    transaction.to_frame(name="Value"),
    use_container_width=True,
    height=300
)

if st.button("Analyze Transaction", type="primary"):
    excluded = ["isFraud", "TransactionID", "TransactionDT"]

    features = transaction.drop(labels=excluded).to_dict()

    # Convert missing values to JSON null
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

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Results")

        score = result["fraud_score"]
        threshold = result["threshold"]

        st.metric("Fraud Model Score", f"{score:.4f}")

        st.progress(min(max(score, 0.0), 1.0))

        if result["prediction"] == 1:
            st.error("⚠️ Transaction Flagged as Potential Fraud")
        else:
            st.success("✓ Transaction Classified as Legitimate")


        # Compare model prediction against the known validation label
        actual_label = int(transaction["isFraud"])
        predicted_label = int(result["prediction"])

        st.subheader("Prediction Evaluation")

        if predicted_label == actual_label:
            if predicted_label == 1:
                st.success("True Positive — Fraud correctly detected")
            else:
                st.success("True Negative — Legitimate transaction correctly identified")
        else:
            if actual_label == 1:
                st.error("False Negative — Fraudulent transaction missed by the model")
            else:
                st.warning("False Positive — Legitimate transaction flagged as fraud")


        st.caption(
            f"Classification threshold: {threshold:.2f}. "
            "This score is not a calibrated probability."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach prediction API: {exc}")
