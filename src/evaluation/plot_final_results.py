"""
PRISM - FINAL RESULTS VISUALIZATION

Creates publication-ready plots from the final ablation results.
All values are read directly from results/final_results.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "final_results.csv"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

VARIANT_LABELS = {
    "baseline_only": "Baseline",
    "baseline_error_detector": "Baseline +\nError Detector",
    "similarity_correction": "Similarity\nCorrection",
    "alternative_model_selection": "Alternative\nSelection",
    "full_prism_without_verification": "PRISM without\nVerification",
    "full_prism": "Full PRISM",
    "full_prism_no_abstention": "PRISM without\nAbstention",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_results():
    """Load the completed ablation results."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{RESULTS_PATH}"
        )

    return pd.read_csv(RESULTS_PATH)


def prepare_split_data(results, split):
    """Prepare one test split for plotting."""

    data = results[
        results["split"] == split
    ].copy()

    data["variant_label"] = data[
        "variant"
    ].map(VARIANT_LABELS)

    return data


def save_figure(filename):
    """Save the current figure."""

    output_path = PLOTS_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Created: {output_path}")


# ============================================================================
# PLOT 1: ACCURACY COMPARISON
# ============================================================================

def plot_accuracy(results):

    for split in results["split"].unique():

        data = prepare_split_data(
            results,
            split,
        )

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            data["variant_label"],
            data["accuracy"],
        )

        plt.ylabel("Accuracy")
        plt.xlabel("Evaluation Variant")
        plt.title(
            f"PRISM Ablation Accuracy — {split}"
        )

        plt.xticks(rotation=30, ha="right")

        plt.ylim(
            0,
            min(
                1.0,
                data["accuracy"].max() + 0.1,
            ),
        )

        save_figure(
            f"ablation_accuracy_{split}.png"
        )


# ============================================================================
# PLOT 2: COVERAGE COMPARISON
# ============================================================================

def plot_coverage(results):

    for split in results["split"].unique():

        data = prepare_split_data(
            results,
            split,
        )

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            data["variant_label"],
            data["coverage"],
        )

        plt.ylabel("Coverage")
        plt.xlabel("Evaluation Variant")
        plt.title(
            f"Prediction Coverage — {split}"
        )

        plt.xticks(rotation=30, ha="right")

        plt.ylim(0, 1.05)

        save_figure(
            f"ablation_coverage_{split}.png"
        )


# ============================================================================
# PLOT 3: SELECTIVE RISK VS COVERAGE
# ============================================================================

def plot_risk_vs_coverage(results):

    plt.figure(
        figsize=(9, 7)
    )

    for split in results["split"].unique():

        data = prepare_split_data(
            results,
            split,
        )

        plt.scatter(
            data["coverage"],
            data["selective_risk"],
            s=100,
            label=split,
        )

        for _, row in data.iterrows():

            plt.annotate(
                row["variant_label"].replace(
                    "\n",
                    " ",
                ),
                (
                    row["coverage"],
                    row["selective_risk"],
                ),
                fontsize=7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk")
    plt.title(
        "PRISM Risk–Coverage Trade-off"
    )

    plt.legend()

    save_figure(
        "risk_vs_coverage.png"
    )


# ============================================================================
# PLOT 4: ERROR RECOVERY
# ============================================================================

def plot_error_recovery(results):

    for split in results["split"].unique():

        data = prepare_split_data(
            results,
            split,
        )

        plt.figure(
            figsize=(11, 6)
        )

        plt.bar(
            data["variant_label"],
            data["error_recovery_rate"],
        )

        plt.ylabel("Error Recovery Rate")
        plt.xlabel("Evaluation Variant")
        plt.title(
            f"Baseline Error Recovery — {split}"
        )

        plt.xticks(rotation=30, ha="right")

        plt.ylim(0, 1.05)

        save_figure(
            f"error_recovery_{split}.png"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("PRISM - FINAL RESULTS VISUALIZATION")
    print("=" * 70)

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results()

    print(
        f"\nLoaded {len(results)} "
        f"ablation results."
    )

    plot_accuracy(results)
    plot_coverage(results)
    plot_risk_vs_coverage(results)
    plot_error_recovery(results)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()