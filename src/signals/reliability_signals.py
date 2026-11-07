"""
PRISM - Reliability Signals Module

This module calculates reliability and uncertainty signals for
baseline model predictions.

Input:
    results/baseline_validation_predictions.parquet

Output:
    results/reliability_signals.parquet
    results/signal_analysis.csv

Signals:
    - confidence
    - probability_margin
    - predictive_entropy

The output will later be used by PRISM's error detection module.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_validation_predictions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "reliability_signals.parquet"
)

ANALYSIS_PATH = (
    PROJECT_ROOT
    / "results"
    / "signal_analysis.csv"
)


# ============================================================
# SIGNAL FUNCTIONS
# ============================================================

def calculate_confidence(probability):
    """
    Confidence in the predicted class.

    For binary classification:
        confidence = max(P(class=1), P(class=0))
    """
    return np.maximum(probability, 1.0 - probability)


def calculate_margin(probability):
    """
    Difference between the probabilities of the two classes.

    A small margin indicates a more uncertain prediction.
    """
    return np.abs(2.0 * probability - 1.0)


def calculate_entropy(probability):
    """
    Binary predictive entropy.

    Higher entropy indicates greater uncertainty.
    """
    epsilon = 1e-12

    p = np.clip(probability, epsilon, 1.0 - epsilon)

    entropy = -(
        p * np.log2(p)
        + (1.0 - p) * np.log2(1.0 - p)
    )

    return entropy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRISM - RELIABILITY SIGNALS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print("\nLoading baseline validation predictions...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{INPUT_PATH}"
        )

    df = pd.read_parquet(INPUT_PATH)

    print(f"Predictions loaded: {len(df):,}")
    print(f"Matches            : {df['match_id'].nunique():,}")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    required_columns = [
        "match_id",
        "over_completed",
        "chasing_team_won",
        "predicted_class",
        "predicted_probability",
        "model",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    probability = df["predicted_probability"].astype(float)

    invalid_probability_count = (
        ((probability < 0) | (probability > 1))
        .sum()
    )

    if invalid_probability_count > 0:
        raise ValueError(
            f"Found {invalid_probability_count} "
            "invalid probabilities outside [0, 1]."
        )

    # --------------------------------------------------------
    # Calculate reliability signals
    # --------------------------------------------------------

    print("\nCalculating reliability signals...")

    df["confidence"] = calculate_confidence(probability)

    df["probability_margin"] = calculate_margin(probability)

    df["predictive_entropy"] = calculate_entropy(probability)

    # --------------------------------------------------------
    # Calculate prediction correctness
    # --------------------------------------------------------

    df["prediction_correct"] = (
        df["predicted_class"].astype(int)
        == df["chasing_team_won"].astype(int)
    ).astype(int)

    df["prediction_error"] = (
        1 - df["prediction_correct"]
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print("\nSignal validation:")

    checks = {
        "Invalid confidence values":
            ((df["confidence"] < 0.5)
             | (df["confidence"] > 1.0)).sum(),

        "Invalid margin values":
            ((df["probability_margin"] < 0.0)
             | (df["probability_margin"] > 1.0)).sum(),

        "Invalid entropy values":
            ((df["predictive_entropy"] < 0.0)
             | (df["predictive_entropy"] > 1.0 + 1e-9)).sum(),
    }

    for name, errors in checks.items():

        status = "PASS" if errors == 0 else "FAIL"

        print(
            f"{status}: {name} = {errors}"
        )

    # --------------------------------------------------------
    # Compare correct vs incorrect predictions
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS")
    print("=" * 70)

    signal_columns = [
        "confidence",
        "probability_margin",
        "predictive_entropy",
    ]

    analysis = (
        df.groupby("prediction_error")[signal_columns]
        .agg(["count", "mean", "median", "std"])
    )

    # Make the table easier to save and read
    analysis.columns = [
        "_".join(column).strip()
        for column in analysis.columns
    ]

    analysis = analysis.reset_index()

    analysis["prediction_group"] = analysis[
        "prediction_error"
    ].map({
        0: "Correct prediction",
        1: "Incorrect prediction",
    })

    # Put the descriptive column first
    columns = (
        ["prediction_group", "prediction_error"]
        + [
            column
            for column in analysis.columns
            if column not in [
                "prediction_group",
                "prediction_error",
            ]
        ]
    )

    analysis = analysis[columns]

    print()

    display_columns = [
        "prediction_group",
        "confidence_mean",
        "probability_margin_mean",
        "predictive_entropy_mean",
    ]

    print(
        analysis[display_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Error rate by confidence level
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR RATE BY CONFIDENCE RANGE")
    print("=" * 70)

    confidence_bins = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01,
    ]

    confidence_labels = [
        "0.50-0.60",
        "0.60-0.70",
        "0.70-0.80",
        "0.80-0.90",
        "0.90-1.00",
    ]

    df["confidence_range"] = pd.cut(
        df["confidence"],
        bins=confidence_bins,
        labels=confidence_labels,
        include_lowest=True,
        right=False,
    )

    confidence_analysis = (
        df.groupby(
            "confidence_range",
            observed=False,
        )
        .agg(
            predictions=(
                "prediction_error",
                "count",
            ),
            errors=(
                "prediction_error",
                "sum",
            ),
            error_rate=(
                "prediction_error",
                "mean",
            ),
        )
        .reset_index()
    )

    confidence_analysis["error_rate"] = (
        confidence_analysis["error_rate"] * 100
    )

    print(
        confidence_analysis.to_string(
            index=False,
            formatters={
                "error_rate":
                    "{:.2f}%".format
            },
        )
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    analysis.to_csv(
        ANALYSIS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    total_predictions = len(df)

    total_errors = int(
        df["prediction_error"].sum()
    )

    overall_error_rate = (
        total_errors / total_predictions
    )

    print("\n" + "=" * 70)
    print("RELIABILITY SIGNAL SUMMARY")
    print("=" * 70)

    print(
        f"Total predictions : "
        f"{total_predictions:,}"
    )

    print(
        f"Correct predictions: "
        f"{total_predictions - total_errors:,}"
    )

    print(
        f"Incorrect predictions: "
        f"{total_errors:,}"
    )

    print(
        f"Baseline error rate: "
        f"{overall_error_rate:.2%}"
    )

    print("\nFiles saved:")

    print(f"  {OUTPUT_PATH}")
    print(f"  {ANALYSIS_PATH}")

    print("\n" + "=" * 70)
    print("RELIABILITY SIGNAL CALCULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()