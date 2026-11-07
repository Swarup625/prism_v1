"""
PRISM - OOF Reliability Signal Generation

This script calculates reliability signals from leakage-safe
out-of-fold baseline predictions.

The resulting dataset will be used to train the PRISM error detector.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "results" / "baseline_oof_predictions.parquet"
OUTPUT_PATH = PROJECT_ROOT / "results" / "reliability_signals_oof.parquet"


# ---------------------------------------------------------------------
# SIGNAL FUNCTIONS
# ---------------------------------------------------------------------

def calculate_confidence(probabilities):
    """
    Confidence is the probability assigned to the predicted class.

    The input probability is assumed to be the probability of class 1
    (chasing team wins).
    """

    probabilities = np.asarray(probabilities)

    return np.maximum(probabilities, 1.0 - probabilities)


def calculate_probability_margin(probabilities):
    """
    Difference between the probabilities of the two possible outcomes.

    For binary classification:
        margin = |P(class 1) - P(class 0)|
    """

    probabilities = np.asarray(probabilities)

    return np.abs(2.0 * probabilities - 1.0)


def calculate_entropy(probabilities):
    """
    Binary predictive entropy.

    Higher entropy means greater uncertainty.
    """

    probabilities = np.asarray(probabilities)

    # Prevent log(0)
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    entropy = -(
        probabilities * np.log2(probabilities)
        + (1.0 - probabilities) * np.log2(1.0 - probabilities)
    )

    return entropy


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PRISM - OOF RELIABILITY SIGNAL GENERATION")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------------

    print("\nLoading OOF baseline predictions...")

    if not INPUT_PATH.exists():
        print(f"\nERROR: File not found:\n{INPUT_PATH}")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)

    print(f"Predictions loaded : {len(df):,}")
    print(f"Matches            : {df['match_id'].nunique():,}")

    # --------------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------------

    required_columns = [
        "match_id",
        "chasing_team_won",
        "predicted_class",
        "predicted_probability",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------------
    # Calculate reliability signals
    # --------------------------------------------------------------

    print("\nCalculating reliability signals...")

    probabilities = df["predicted_probability"].astype(float)

    df["confidence"] = calculate_confidence(probabilities)

    df["probability_margin"] = (
        calculate_probability_margin(probabilities)
    )

    df["predictive_entropy"] = (
        calculate_entropy(probabilities)
    )

    # --------------------------------------------------------------
    # Create error label
    # --------------------------------------------------------------

    print("Creating error labels...")

    df["error_label"] = (
        df["predicted_class"].astype(int)
        != df["chasing_team_won"].astype(int)
    ).astype(int)

    # --------------------------------------------------------------
    # Validation checks
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SIGNAL VALIDATION")
    print("=" * 70)

    invalid_confidence = (
        (df["confidence"] < 0.5)
        | (df["confidence"] > 1.0)
    ).sum()

    invalid_margin = (
        (df["probability_margin"] < 0.0)
        | (df["probability_margin"] > 1.0)
    ).sum()

    invalid_entropy = (
        (df["predictive_entropy"] < 0.0)
        | (df["predictive_entropy"] > 1.0)
    ).sum()

    invalid_error_labels = (
        ~df["error_label"].isin([0, 1])
    ).sum()

    print(
        f"Invalid confidence values : {invalid_confidence}"
    )

    print(
        f"Invalid margin values     : {invalid_margin}"
    )

    print(
        f"Invalid entropy values    : {invalid_entropy}"
    )

    print(
        f"Invalid error labels      : {invalid_error_labels}"
    )

    if (
        invalid_confidence
        or invalid_margin
        or invalid_entropy
        or invalid_error_labels
    ):
        raise RuntimeError(
            "Signal validation failed."
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    total_errors = int(df["error_label"].sum())
    total_correct = len(df) - total_errors

    print("\n" + "=" * 70)
    print("OOF SIGNAL SUMMARY")
    print("=" * 70)

    print(f"Total predictions : {len(df):,}")
    print(f"Correct predictions: {total_correct:,}")
    print(f"Incorrect predictions: {total_errors:,}")
    print(
        f"Baseline error rate: "
        f"{df['error_label'].mean() * 100:.2f}%"
    )

    print("\nSignal averages:")

    summary = (
        df.groupby("error_label")[
            [
                "confidence",
                "probability_margin",
                "predictive_entropy",
            ]
        ]
        .mean()
        .rename(
            index={
                0: "Correct prediction",
                1: "Incorrect prediction",
            }
        )
    )

    print(summary.to_string())

    # --------------------------------------------------------------
    # Save output
    # --------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OOF RELIABILITY SIGNAL GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nSignals saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()