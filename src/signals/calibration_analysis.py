"""
PRISM - BASELINE MODEL CALIBRATION ANALYSIS

This script evaluates whether the predicted probabilities from the
selected baseline model correspond to observed outcome frequencies.

Input:
    results/baseline_validation_predictions.parquet

Outputs:
    results/calibration_bins.csv
    results/calibration_metrics.csv
    docs/calibration_report.md
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------
# PROJECT PATH SETUP
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

INPUT_FILE = RESULTS_DIR / "baseline_validation_predictions.parquet"

CALIBRATION_BINS_FILE = RESULTS_DIR / "calibration_bins.csv"
CALIBRATION_METRICS_FILE = RESULTS_DIR / "calibration_metrics.csv"
CALIBRATION_REPORT_FILE = DOCS_DIR / "calibration_report.md"


# ---------------------------------------------------------------------
# CALIBRATION FUNCTIONS
# ---------------------------------------------------------------------

def calculate_calibration_bins(y_true, probabilities, n_bins=10):
    """
    Divide predictions into probability bins and compare the average
    predicted probability with the observed fraction of positive outcomes.
    """

    df = pd.DataFrame({
        "actual": y_true,
        "probability": probabilities
    })

    # Include probability 1.0 in the final bin.
    bin_edges = np.linspace(0, 1, n_bins + 1)

    df["bin"] = pd.cut(
        df["probability"],
        bins=bin_edges,
        include_lowest=True,
        right=True
    )

    grouped = df.groupby("bin", observed=False)

    rows = []

    for interval, group in grouped:
        if len(group) == 0:
            continue

        rows.append({
            "probability_range": str(interval),
            "predictions": len(group),
            "mean_predicted_probability": group["probability"].mean(),
            "observed_win_rate": group["actual"].mean(),
            "calibration_gap": (
                group["probability"].mean()
                - group["actual"].mean()
            )
        })

    return pd.DataFrame(rows)


def calculate_ece(y_true, probabilities, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    ECE is the weighted average difference between predicted probability
    and observed outcome frequency across probability bins.
    """

    df = pd.DataFrame({
        "actual": y_true,
        "probability": probabilities
    })

    bin_edges = np.linspace(0, 1, n_bins + 1)

    df["bin"] = pd.cut(
        df["probability"],
        bins=bin_edges,
        include_lowest=True,
        right=True
    )

    total = len(df)
    ece = 0.0

    for _, group in df.groupby("bin", observed=False):
        if len(group) == 0:
            continue

        avg_probability = group["probability"].mean()
        observed_frequency = group["actual"].mean()

        bin_weight = len(group) / total

        ece += bin_weight * abs(
            avg_probability - observed_frequency
        )

    return float(ece)


def write_report(metrics, calibration_bins):
    """
    Create a simple Markdown calibration report.
    """

    brier = metrics["brier_score"]
    ece = metrics["expected_calibration_error"]
    total = metrics["total_predictions"]

    report = f"""# PRISM Baseline Model Calibration Report

## Purpose

This analysis evaluates whether the probabilities produced by the selected
baseline model correspond to the actual frequency of the predicted outcome.

The evaluation uses the validation dataset only.

## Dataset

- Total validation predictions: **{total}**
- Model: **{metrics["model"]}**

## Calibration Metrics

| Metric | Value |
|---|---:|
| Brier Score | {brier:.6f} |
| Expected Calibration Error (ECE) | {ece:.6f} |

### Interpretation

A lower Brier Score indicates better probability predictions.

A lower Expected Calibration Error indicates that predicted probabilities
are closer to the actual observed outcome frequencies.

## Calibration by Probability Range

"""

    report += "| Probability Range | Predictions | Mean Predicted Probability | Observed Win Rate | Calibration Gap |\n"
    report += "|---|---:|---:|---:|---:|\n"

    for _, row in calibration_bins.iterrows():
        report += (
            f"| {row['probability_range']} "
            f"| {int(row['predictions'])} "
            f"| {row['mean_predicted_probability']:.4f} "
            f"| {row['observed_win_rate']:.4f} "
            f"| {row['calibration_gap']:.4f} |\n"
        )

    report += """

## Conclusion

The calibration analysis provides an additional view of prediction
reliability. Calibration information can be used alongside confidence,
probability margin, and predictive entropy when studying whether a
prediction is likely to be unreliable.

This analysis does not use the 2025 or 2026 final test datasets.
"""

    CALIBRATION_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(CALIBRATION_REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PRISM - BASELINE MODEL CALIBRATION ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:\n{INPUT_FILE}")
        sys.exit(1)

    print("\nLoading validation predictions...")

    df = pd.read_parquet(INPUT_FILE)

    required_columns = [
        "chasing_team_won",
        "predicted_probability",
        "model"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    y_true = df["chasing_team_won"].astype(int)
    probabilities = df["predicted_probability"].astype(float)

    print(f"Predictions loaded : {len(df):,}")
    print(f"Matches            : {df['match_id'].nunique():,}")
    print(f"Model              : {df['model'].iloc[0]}")

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    print("\nValidating probabilities...")

    invalid_probabilities = (
        (probabilities < 0) | (probabilities > 1)
    ).sum()

    print(
        f"Invalid probabilities: {invalid_probabilities}"
    )

    if invalid_probabilities > 0:
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    # --------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------

    print("\nCalculating calibration metrics...")

    brier = brier_score_loss(
        y_true,
        probabilities
    )

    ece = calculate_ece(
        y_true,
        probabilities,
        n_bins=10
    )

    metrics = {
        "model": df["model"].iloc[0],
        "total_predictions": len(df),
        "brier_score": brier,
        "expected_calibration_error": ece
    }

    metrics_df = pd.DataFrame([metrics])

    # --------------------------------------------------------------
    # CALIBRATION BINS
    # --------------------------------------------------------------

    calibration_bins = calculate_calibration_bins(
        y_true,
        probabilities,
        n_bins=10
    )

    # --------------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CALIBRATION METRICS")
    print("=" * 70)

    print(f"Brier Score : {brier:.6f}")
    print(f"ECE         : {ece:.6f}")

    print("\n" + "=" * 70)
    print("CALIBRATION BY PROBABILITY RANGE")
    print("=" * 70)

    print(
        calibration_bins.to_string(
            index=False,
            formatters={
                "mean_predicted_probability": "{:.4f}".format,
                "observed_win_rate": "{:.4f}".format,
                "calibration_gap": "{:.4f}".format
            }
        )
    )

    # --------------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    calibration_bins.to_csv(
        CALIBRATION_BINS_FILE,
        index=False
    )

    metrics_df.to_csv(
        CALIBRATION_METRICS_FILE,
        index=False
    )

    write_report(
        metrics,
        calibration_bins
    )

    print("\n" + "=" * 70)
    print("CALIBRATION ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nFiles saved:")

    print(f"  {CALIBRATION_BINS_FILE}")
    print(f"  {CALIBRATION_METRICS_FILE}")
    print(f"  {CALIBRATION_REPORT_FILE}")

    print("\nIMPORTANT:")
    print(
        "Only validation data was used. "
        "The 2025 and 2026 test datasets remain untouched."
    )


if __name__ == "__main__":
    main()