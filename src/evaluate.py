
from pathlib import Path
import shap
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocess import (
    load_data,
    merge_data,
    create_features,
    split_data,
    prepare_model_data
)


def main():
    print("Loading fraud detection dataset...")

    # Load and prepare the dataset
    transactions, identity = load_data()

    df = merge_data(transactions, identity)
    df = create_features(df, identity)

    train_df, val_df = split_data(df)

    print("\nLoading saved LightGBM model...")

    model = joblib.load("models/lightgbm_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")

    # Use the same feature columns as training
    excluded_columns = ["isFraud", "TransactionID", "TransactionDT"]

    X_train_raw = train_df.drop(columns=excluded_columns)
    X_val_raw = val_df.drop(columns=excluded_columns)

    y_val = val_df["isFraud"]

    # Preserve the feature order used when fitting the preprocessor
    feature_columns = list(preprocessor.feature_names_in_)
    X_val_raw = X_val_raw[feature_columns]

    # Transform validation data without refitting
    X_val = preprocessor.transform(X_val_raw)

    print("Validation dataset shape:", X_val.shape)

    # Generate predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix visualization
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("LightGBM Fraud Detection - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix saved successfully!")


    # ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = roc_auc_score(y_val, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"LightGBM (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LightGBM - ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"LightGBM (AP = {pr_auc:.4f})")
    plt.axhline(
        y=float(np.mean(y_val)),
        linestyle="--",
        label="Fraud prevalence baseline"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("LightGBM - Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print("ROC and Precision-Recall curves saved successfully!")


    # Evaluate different classification thresholds
    print("\nEvaluating classification thresholds...")

    thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = []

    for threshold in thresholds:
        predictions = (y_proba >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val, predictions, labels=[0, 1]
        ).ravel()

        threshold_results.append({
            "Threshold": round(float(threshold), 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "True Positives": int(tp)
        })

    import pandas as pd

    threshold_df = pd.DataFrame(threshold_results)

    print("\nThreshold Comparison:")
    print(threshold_df.to_string(index=False))

    # Save results
    threshold_df.to_csv(
        "reports/threshold_comparison.csv",
        index=False
    )

    # Plot precision, recall and F1 across thresholds
    plt.figure(figsize=(9, 6))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1"],
        marker="o",
        label="F1-score"
    )

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("LightGBM - Threshold Optimization")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_threshold_optimization.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nThreshold comparison saved successfully!")


    # LightGBM feature importance
    print("\nCalculating feature importance...")

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 20 Important Features:")
    print(importance_df.head(20).to_string(index=False))

    importance_df.to_csv(
        "reports/lightgbm_feature_importance.csv",
        index=False
    )

    # Plot top 20 features
    top_features = importance_df.head(20).iloc[::-1]

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Feature Importance (Split Count)")
    plt.ylabel("Feature")
    plt.title("LightGBM - Top 20 Important Features")
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nFeature importance saved successfully!")


    # SHAP explainability
    print("\nGenerating SHAP explanations...")

    # Use a small, reproducible validation sample
    rng = np.random.default_rng(42)

    sample_size = min(500, X_val.shape[0])

    sample_indices = rng.choice(
        X_val.shape[0],
        size=sample_size,
        replace=False
    )

    X_sample = X_val[sample_indices]

    # Recover transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Explain the trained LightGBM model
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # Handle binary classification SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif np.asarray(shap_values).ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Global feature impact summary
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=20,
        show=False
    )

    plt.title("LightGBM - SHAP Feature Impact")
    plt.tight_layout()

    plt.savefig(
        output_dir / "lightgbm_shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("SHAP summary plot saved successfully!")





if __name__ == "__main__":
    main()
