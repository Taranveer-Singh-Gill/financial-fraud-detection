import numpy as np
from pathlib import Path
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.preprocess import (
    load_data,
    merge_data,
    create_features,
    split_data,
    prepare_model_data,
)


def evaluate_model(name, model, X_val, y_val):
    """Evaluate a fraud detection classifier."""

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    results = {
        "Model": name,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_val, probabilities
        ),
        "PR-AUC": average_precision_score(
            y_val, probabilities
        ),
    }

    return results


def main():
    print("Loading and preparing data...")

    transactions, identity = load_data()
    merged = merge_data(transactions, identity)
    featured = create_features(merged, identity)

    train_df, val_df = split_data(featured)

    (
        X_train,
        X_val,
        y_train,
        y_val,
        preprocessor,
    ) = prepare_model_data(train_df, val_df)

    results = []

    # Baseline model
    print("\nTraining Dummy Classifier...")

    dummy = DummyClassifier(
        strategy="prior"
    )

    dummy.fit(X_train, y_train)

    results.append(
        evaluate_model(
            "Dummy Classifier",
            dummy,
            X_val,
            y_val
        )
    )


    # Train LightGBM
    print("\nTraining LightGBM...")

    lightgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    lightgbm.fit(X_train, y_train)
    # Save trained model and fitted preprocessing transformer
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lightgbm, model_dir / "lightgbm_model.joblib")
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    print("\nLightGBM model and preprocessor saved!")

    results.append(
        evaluate_model(
            "LightGBM",
            lightgbm,
            X_val,
            y_val
        )
    )


    # Logistic Regression
    print("\nTraining Logistic Regression...")

    logistic = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        C=0.1,
        max_iter=1500,
        tol=1e-3,
        random_state=42
    )
    logistic.fit(X_train, y_train)

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic,
            X_val,
            y_val
        )
    )

    print("\nModel comparison:")

    for result in results:
        print(result)
    # Save model evaluation results
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        reports_dir / "model_comparison.csv",
        index=False
    )

    print("\nModel comparison saved to reports/model_comparison.csv")

if __name__ == "__main__":
    main()
