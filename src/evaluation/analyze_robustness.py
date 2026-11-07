"""
PRISM - ROBUSTNESS RESULTS ANALYSIS

Creates paper-ready tables and visualizations from the frozen
robustness evaluation results.

This script performs analysis only.
No models are loaded, retrained, or modified.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = (
    RESULTS_DIR
    / "robustness_results.csv"
)

TABLE_PATH = (
    RESULTS_DIR
    / "robustness_summary.csv"
)

NOISE_TABLE_PATH = (
    RESULTS_DIR
    / "noise_robustness_summary.csv"
)

FIGURE_PATH = (
    RESULTS_DIR
    / "robustness_accuracy.png"
)


# ============================================================================
# LOAD RESULTS
# ============================================================================

def main():

    print("=" * 70)
    print("PRISM - ROBUSTNESS RESULTS ANALYSIS")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Robustness results not found:\n{INPUT_PATH}"
        )

    data = pd.read_csv(INPUT_PATH)

    print(
        f"\nLoaded {len(data)} robustness evaluations."
    )

    # ------------------------------------------------------------------
    # ADD IMPROVEMENT METRICS
    # ------------------------------------------------------------------

    data["accuracy_change"] = (
        data["final_accuracy"]
        - data["baseline_accuracy"]
    )

    data["accuracy_change_pct"] = (
        data["accuracy_change"] * 100
    )

    # ------------------------------------------------------------------
    # CREATE SUMMARY TABLE
    # ------------------------------------------------------------------

    summary = data[
    [
        "scenario",
        "split",
        "total_states",
        "baseline_accuracy",
        "final_accuracy",
        "accuracy_change",
        "accuracy_change_pct",
        "coverage",
        "selective_risk",
        "error_detection_rate",
        "error_recovery_rate",
        "harmful_correction_rate",
    ]
].copy()

    summary = summary.sort_values(
        ["scenario", "split"]
    )

    summary.to_csv(
        TABLE_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # NOISE ROBUSTNESS SUMMARY
    # ------------------------------------------------------------------

    noise_data = data[
        data["scenario"].str.startswith(
            "noise_"
        )
    ].copy()

    noise_summary = noise_data[
        [
            "scenario",
            "split",
            "baseline_accuracy",
            "final_accuracy",
            "accuracy_change",
            "coverage",
            "selective_risk",
        ]
    ]

    noise_summary.to_csv(
        NOISE_TABLE_PATH,
        index=False,
    )

    # ------------------------------------------------------------------
    # CREATE ACCURACY FIGURE
    # ------------------------------------------------------------------

    scenarios = data["scenario"].unique()

    test_2025 = data[
        data["split"] == "test_2025"
    ].set_index("scenario")

    test_2026 = data[
        data["split"] == "test_2026"
    ].set_index("scenario")

    x = range(len(scenarios))

    width = 0.35

    plt.figure(figsize=(14, 7))

    plt.bar(
        [i - width / 2 for i in x],
        [
            test_2025.loc[s, "final_accuracy"]
            for s in scenarios
        ],
        width,
        label="Test 2025",
    )

    plt.bar(
        [i + width / 2 for i in x],
        [
            test_2026.loc[s, "final_accuracy"]
            for s in scenarios
        ],
        width,
        label="Test 2026",
    )

    plt.xticks(
        list(x),
        scenarios,
        rotation=45,
        ha="right",
    )

    plt.ylabel("PRISM Selective Accuracy")
    plt.xlabel("Robustness Scenario")
    plt.title(
        "PRISM Performance Across Robustness and Failure Scenarios"
    )

    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ------------------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    display_columns = [
        "scenario",
        "split",
        "baseline_accuracy",
        "final_accuracy",
        "accuracy_change_pct",
        "coverage",
        "selective_risk",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nFiles created:")
    print(f"  Summary table : {TABLE_PATH}")
    print(f"  Noise table   : {NOISE_TABLE_PATH}")
    print(f"  Figure        : {FIGURE_PATH}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()