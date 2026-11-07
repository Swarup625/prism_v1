"""
PRISM - Error Detector

This module learns to estimate whether a baseline ML prediction
is likely to be incorrect.

Training data:
    Leakage-safe Out-of-Fold (OOF) predictions from 2020-2023.

Validation data:
    Predictions from the 2024 validation season.

The 2025 and 2026 test sets are not used here.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = PROJECT_ROOT / "results" / "reliability_signals_oof.parquet"
VALIDATION_PATH = PROJECT_ROOT / "results" / "reliability_signals.parquet"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODEL_DIR / "error_detector.joblib"
METRICS_PATH = RESULTS_DIR / "detector_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "detector_validation_predictions.parquet"


# =====================================================================
# CONFIGURATION
# =====================================================================

SIGNAL_FEATURES = [
    "confidence",
    "probability_margin",
    "predictive_entropy",
]

RANDOM_STATE = 42


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def calculate_metrics(y_true, y_pred, risk_scores):
    """Calculate evaluation metrics for the error detector."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true, risk_scores
        ),
    }


def print_metrics(name, metrics, matrix):
    """Print evaluation results."""

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)


def find_best_threshold(y_true, risk_scores):
    """
    Find the risk-score threshold with the highest validation F1 score.
    """

    thresholds = np.arange(0.05, 0.96, 0.01)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (risk_scores >= threshold).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("PRISM - ERROR DETECTOR")
    print("=" * 70)

    # -----------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------

    print("\nLoading reliability signal datasets...")

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF signal file not found:\n{OOF_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation signal file not found:\n{VALIDATION_PATH}"
        )

    train_df = pd.read_parquet(OOF_PATH)
    val_df = pd.read_parquet(VALIDATION_PATH)

    print(f"OOF training predictions : {len(train_df):,}")
    print(f"OOF training matches     : {train_df['match_id'].nunique():,}")
    print(f"Validation predictions   : {len(val_df):,}")
    print(f"Validation matches       : {val_df['match_id'].nunique():,}")

    # -----------------------------------------------------------------
    # Standardize error labels
    # -----------------------------------------------------------------

    if "error_label" not in train_df.columns:
        train_df["error_label"] = (
            train_df["predicted_class"].astype(int)
            != train_df["chasing_team_won"].astype(int)
        ).astype(int)

    if "prediction_error" in val_df.columns:
        val_df["error_label"] = val_df["prediction_error"].astype(int)

    elif "error_label" not in val_df.columns:
        val_df["error_label"] = (
            val_df["predicted_class"].astype(int)
            != val_df["chasing_team_won"].astype(int)
        ).astype(int)

    # -----------------------------------------------------------------
    # Validate features
    # -----------------------------------------------------------------

    missing_train = [
        feature for feature in SIGNAL_FEATURES
        if feature not in train_df.columns
    ]

    missing_val = [
        feature for feature in SIGNAL_FEATURES
        if feature not in val_df.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training features: {missing_train}"
        )

    if missing_val:
        raise ValueError(
            f"Missing validation features: {missing_val}"
        )

    X_train = train_df[SIGNAL_FEATURES].copy()
    y_train = train_df["error_label"].astype(int)

    X_val = val_df[SIGNAL_FEATURES].copy()
    y_val = val_df["error_label"].astype(int)

    print("\n" + "=" * 70)
    print("ERROR DATASET SUMMARY")
    print("=" * 70)

    print(f"\nTraining error rate   : {y_train.mean() * 100:.2f}%")
    print(f"Validation error rate : {y_val.mean() * 100:.2f}%")

    # -----------------------------------------------------------------
    # Train detector
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING ERROR DETECTOR")
    print("=" * 70)

    detector = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    start_time = time.perf_counter()
    detector.fit(X_train, y_train)
    training_time = time.perf_counter() - start_time

    print(f"\nTraining completed in {training_time:.4f} seconds.")

    # -----------------------------------------------------------------
    # Generate validation risk scores
    # -----------------------------------------------------------------

    risk_scores = detector.predict_proba(X_val)[:, 1]

    default_predictions = (risk_scores >= 0.50).astype(int)

    default_metrics = calculate_metrics(
        y_val,
        default_predictions,
        risk_scores,
    )

    default_matrix = confusion_matrix(
        y_val,
        default_predictions,
    )

    print_metrics(
        "ERROR DETECTOR - DEFAULT THRESHOLD (0.50)",
        default_metrics,
        default_matrix,
    )

    # -----------------------------------------------------------------
    # Select threshold on validation data
    # -----------------------------------------------------------------

    best_threshold, best_threshold_f1 = find_best_threshold(
        y_val.to_numpy(),
        risk_scores,
    )

    optimized_predictions = (
        risk_scores >= best_threshold
    ).astype(int)

    optimized_metrics = calculate_metrics(
        y_val,
        optimized_predictions,
        risk_scores,
    )

    optimized_matrix = confusion_matrix(
        y_val,
        optimized_predictions,
    )

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    print(f"\nSelected threshold : {best_threshold:.2f}")
    print(
        "Selection criterion: highest validation F1 score "
        f"({best_threshold_f1:.4f})"
    )

    print_metrics(
        "ERROR DETECTOR - OPTIMIZED THRESHOLD",
        optimized_metrics,
        optimized_matrix,
    )

    # -----------------------------------------------------------------
    # Confidence-only baseline
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIMPLE CONFIDENCE BASELINE")
    print("=" * 70)

    confidence_thresholds = np.arange(0.50, 1.00, 0.01)

    best_conf_threshold = 0.50
    best_conf_f1 = -1.0

    confidence_values = val_df["confidence"].to_numpy()

    for threshold in confidence_thresholds:

        predictions = (
            confidence_values < threshold
        ).astype(int)

        score = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        if score > best_conf_f1:
            best_conf_f1 = score
            best_conf_threshold = threshold

    confidence_predictions = (
        confidence_values < best_conf_threshold
    ).astype(int)

    confidence_metrics = calculate_metrics(
        y_val,
        confidence_predictions,
        1.0 - confidence_values,
    )

    confidence_matrix = confusion_matrix(
        y_val,
        confidence_predictions,
    )

    print(
        f"\nSelected confidence threshold: "
        f"{best_conf_threshold:.2f}"
    )

    print(
        "Selection criterion: highest validation F1 score "
        f"({best_conf_f1:.4f})"
    )

    print_metrics(
        "CONFIDENCE-BASED ERROR DETECTION",
        confidence_metrics,
        confidence_matrix,
    )

    # -----------------------------------------------------------------
    # Compare methods
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DETECTOR COMPARISON")
    print("=" * 70)

    comparison_df = pd.DataFrame([
        {
            "method": "logistic_error_detector",
            "threshold": best_threshold,
            **optimized_metrics,
        },
        {
            "method": "confidence_baseline",
            "threshold": best_conf_threshold,
            **confidence_metrics,
        },
    ])

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Detector coefficients
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DETECTOR SIGNAL COEFFICIENTS")
    print("=" * 70)

    coefficient_df = pd.DataFrame({
        "signal": SIGNAL_FEATURES,
        "coefficient": detector.coef_[0],
    })

    coefficient_df["absolute_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print(
        coefficient_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -----------------------------------------------------------------
    # Save model and results
    # -----------------------------------------------------------------

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": detector,
            "features": SIGNAL_FEATURES,
            "threshold": best_threshold,
        },
        MODEL_PATH,
    )

    comparison_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    output_df = val_df.copy()

    output_df["error_risk_score"] = risk_scores
    output_df["error_detector_flag"] = optimized_predictions
    output_df["selected_threshold"] = best_threshold

    output_df.to_parquet(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ERROR DETECTOR COMPLETE")
    print("=" * 70)

    print("\nFiles saved:")
    print(f"  Model:\n  {MODEL_PATH}")
    print(f"  Metrics:\n  {METRICS_PATH}")
    print(f"  Validation predictions:\n  {PREDICTIONS_PATH}")

    print(
        "\nIMPORTANT: The 2025 and 2026 test sets "
        "remain completely untouched."
    )


if __name__ == "__main__":
    main()
    