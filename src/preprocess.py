from sklearn.preprocessing import StandardScaler

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data():
    """Load the IEEE-CIS training datasets."""

    transactions = pd.read_csv(
        RAW_DATA_DIR / "train_transaction.csv"
    )

    identity = pd.read_csv(
        RAW_DATA_DIR / "train_identity.csv"
    )

    print("Transactions:", transactions.shape)
    print("Identity:", identity.shape)

    return transactions, identity


def merge_data(transactions, identity):
    """Merge transaction and identity datasets."""

    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one"
    )

    print("Merged dataset:", merged.shape)

    return merged


def create_features(df, identity):
    """Create time-based and identity features."""

    df = df.copy()

    # Relative time features; TransactionDT is not a calendar date.
    df["TransactionDay"] = (
        df["TransactionDT"] // 86400
    ).astype(int)

    df["TransactionHour"] = (
        (df["TransactionDT"] // 3600) % 24
    ).astype(int)

    # Flag whether identity data was available.
    df["HasIdentity"] = df["TransactionID"].isin(
        identity["TransactionID"]
    ).astype(int)
    # Preserve missingness as a possible fraud signal.
    df["MissingFeatureCount"] = df.isna().sum(axis=1)

    return df


def split_data(df, train_ratio=0.8):
    """Split transactions chronologically into train and validation."""

    df = df.sort_values(
        ["TransactionDT", "TransactionID"]
    ).reset_index(drop=True)

    split_index = int(len(df) * train_ratio)

    train_df = df.iloc[:split_index].copy()
    val_df = df.iloc[split_index:].copy()

    print("\nTraining dataset:", train_df.shape)
    print("Validation dataset:", val_df.shape)

    print(
        "Training fraud rate:",
        round(train_df["isFraud"].mean() * 100, 2),
        "%"
    )

    print(
        "Validation fraud rate:",
        round(val_df["isFraud"].mean() * 100, 2),
        "%"
    )

    return train_df, val_df


def prepare_model_data(train_df, val_df):

    print("\nAll preprocessing validation checks passed!")
    """Prepare numeric model features without data leakage."""

    excluded_columns = [
        "isFraud",
        "TransactionID",
        "TransactionDT"
    ]

    feature_columns = [
        col for col in train_df.columns
        if col not in excluded_columns
    ]

    X_train = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()

    y_train = train_df["isFraud"].copy()
    y_val = val_df["isFraud"].copy()

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Missing"
        )),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("\nProcessed training shape:", X_train_processed.shape)
    print("Processed validation shape:", X_val_processed.shape)

    print("Numeric features:", len(numeric_columns))
    print("Categorical features:", len(categorical_columns))

    return (
        X_train_processed,
        X_val_processed,
        y_train,
        y_val,
        preprocessor
    )



if __name__ == "__main__":
    transactions, identity = load_data()

    merged = merge_data(transactions, identity)

    featured = create_features(merged, identity)

    print("\nEngineered dataset:", featured.shape)

    print(
        featured[
            [
                "TransactionID",
                "TransactionDay",
                "TransactionHour",
                "HasIdentity",
                "MissingFeatureCount"
            ]
        ].head()
    )
    train_df, val_df = split_data(featured)
    (
        X_train,
        X_val,
        y_train,
        y_val,
        preprocessor
    ) = prepare_model_data(train_df, val_df)
    # Validate processed datasets
    assert X_train.shape[0] == len(y_train)
    assert X_val.shape[0] == len(y_val)

    assert X_train.shape[1] == X_val.shape[1]

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()

    assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min()

    print("\nAll preprocessing validation checks passed!")