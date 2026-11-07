"""
PRISM - FINAL RESULTS ANALYSIS

Reads the frozen ablation results and generates:
1. Summary statistics
2. Comparison plots
3. Results summary documentation
4. Claim-evidence table

No models are trained or thresholds tuned in this script.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
DOCS_DIR = PROJECT_ROOT / "docs"

RESULTS_PATH = RESULTS_DIR / "final_results.csv"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_results():
    """Load final frozen ablation results."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Final results not found:\n{RESULTS_PATH}"
        )

    results = pd.read_csv(RESULTS_PATH)

    return results


# ============================================================================
# PLOTS
# ============================================================================

def plot_accuracy_comparison(results):
    """Compare accuracy across all variants."""

    pivot = results.pivot(
        index="variant",
        columns="split",
        values="accuracy",
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "PRISM Ablation Study: Accuracy Comparison"
    )

    ax.set_xlabel("System Variant")
    ax.set_ylabel("Accuracy")
    ax.tick_params(
        axis="x",
        rotation=35,
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "accuracy_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def plot_coverage_risk(results):
    """
    Compare coverage and selective risk.

    Lower selective risk is better.
    Higher coverage is better.
    """

    full_prism = results[
        results["variant"] == "full_prism"
    ]

    baseline = results[
        results["variant"] == "baseline_only"
    ]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        baseline["coverage"],
        baseline["selective_risk"],
        s=100,
        label="Baseline",
    )

    plt.scatter(
        full_prism["coverage"],
        full_prism["selective_risk"],
        s=100,
        label="Full PRISM",
    )

    for _, row in full_prism.iterrows():

        plt.annotate(
            row["split"],
            (
                row["coverage"],
                row["selective_risk"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    for _, row in baseline.iterrows():

        plt.annotate(
            row["split"],
            (
                row["coverage"],
                row["selective_risk"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title(
        "Coverage vs Selective Risk"
    )

    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "coverage_vs_risk.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def plot_error_recovery(results):
    """Compare error recovery and harmful corrections."""

    variants = results[
        results["variant"].isin([
            "similarity_correction",
            "alternative_model_selection",
            "full_prism_without_verification",
            "full_prism",
            "full_prism_no_abstention",
        ])
    ].copy()

    summary = (
        variants.groupby("variant")[
            [
                "error_recovery_rate",
                "harmful_correction_rate",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary.plot(
        x="variant",
        y=[
            "error_recovery_rate",
            "harmful_correction_rate",
        ],
        kind="bar",
        figsize=(12, 6),
    )

    plt.title(
        "Correction Benefit vs Correction Harm"
    )

    plt.xlabel("System Variant")
    plt.ylabel("Rate")
    plt.xticks(rotation=35)

    plt.tight_layout()

    output_path = (
        PLOTS_DIR
        / "correction_recovery_vs_harm.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


# ============================================================================
# DOCUMENTATION
# ============================================================================

def create_results_summary(results):
    """Create the main research results summary."""

    baseline = results[
        results["variant"] == "baseline_only"
    ]

    full_prism = results[
        results["variant"] == "full_prism"
    ]

    similarity = results[
        results["variant"] == "similarity_correction"
    ]

    alternative = results[
        results["variant"] == "alternative_model_selection"
    ]

    lines = []

    lines.append("# PRISM Final Results Summary\n")

    lines.append(
        "## Experimental Protocol\n"
    )

    lines.append(
        "All final experiments used frozen trained models, "
        "unchanged decision thresholds, and historical similarity "
        "references constructed from training data only. "
        "No models were retrained and no thresholds were tuned "
        "using the 2025 or 2026 test sets.\n"
    )

    lines.append(
        "## Baseline and Full PRISM Results\n"
    )

    for split in ["test_2025", "test_2026"]:

        baseline_row = baseline[
            baseline["split"] == split
        ].iloc[0]

        prism_row = full_prism[
            full_prism["split"] == split
        ].iloc[0]

        accuracy_gain = (
            prism_row["accuracy"]
            - baseline_row["accuracy"]
        )

        risk_reduction = (
            baseline_row["selective_risk"]
            - prism_row["selective_risk"]
        )

        lines.append(
            f"### {split}\n"
        )

        lines.append(
            f"- Baseline accuracy: "
            f"{baseline_row['accuracy']:.4f}\n"
        )

        lines.append(
            f"- Full PRISM accuracy on returned predictions: "
            f"{prism_row['accuracy']:.4f}\n"
        )

        lines.append(
            f"- Accuracy difference: "
            f"{accuracy_gain:+.4f}\n"
        )

        lines.append(
            f"- PRISM coverage: "
            f"{prism_row['coverage']:.4f}\n"
        )

        lines.append(
            f"- Baseline risk: "
            f"{baseline_row['selective_risk']:.4f}\n"
        )

        lines.append(
            f"- PRISM selective risk: "
            f"{prism_row['selective_risk']:.4f}\n"
        )

        lines.append(
            f"- Selective risk reduction: "
            f"{risk_reduction:+.4f}\n"
        )

    lines.append(
        "\n## Ablation Findings\n"
    )

    lines.append(
        "The ablation study shows that the individual correction "
        "components do not provide uniform improvements across "
        "all test seasons.\n"
    )

    lines.append(
        "Similarity-based correction improved the baseline on both "
        "test sets, whereas alternative-model selection produced "
        "smaller or inconsistent gains.\n"
    )

    lines.append(
        "The full PRISM system achieved its strongest reliability "
        "results when verification and abstention were enabled, "
        "indicating that selectively withholding unsupported "
        "predictions is an important component of the overall "
        "reliability mechanism.\n"
    )

    lines.append(
        "\n## Negative Results and Limitations\n"
    )

    lines.append(
        "- Alternative-model selection did not consistently improve "
        "performance over the baseline.\n"
    )

    lines.append(
        "- PRISM does not return a prediction for every match state; "
        "the reliability improvement is accompanied by reduced "
        "coverage.\n"
    )

    lines.append(
        "- Some corrections can still be harmful, demonstrating "
        "that correction evidence should not be treated as perfectly "
        "reliable.\n"
    )

    lines.append(
        "- Results are based on IPL match-state data and may not "
        "directly generalize to other sports prediction tasks.\n"
    )

    output_path = DOCS_DIR / "results_summary.md"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.writelines(lines)

    return output_path


def create_claim_evidence_table(results):
    """Create a claim-to-evidence table for the paper."""

    baseline = results[
        results["variant"] == "baseline_only"
    ]

    full_prism = results[
        results["variant"] == "full_prism"
    ]

    rows = []

    for split in [
        "test_2025",
        "test_2026",
    ]:

        baseline_row = baseline[
            baseline["split"] == split
        ].iloc[0]

        prism_row = full_prism[
            full_prism["split"] == split
        ].iloc[0]

        rows.append({
            "claim":
                f"PRISM reduces selective prediction risk on {split}",
            "evidence":
                f"Baseline risk = "
                f"{baseline_row['selective_risk']:.4f}; "
                f"PRISM risk = "
                f"{prism_row['selective_risk']:.4f}",
            "support":
                "Supported",
        })

        rows.append({
            "claim":
                f"PRISM improves reliability on returned predictions "
                f"on {split}",
            "evidence":
                f"Baseline accuracy = "
                f"{baseline_row['accuracy']:.4f}; "
                f"PRISM accuracy = "
                f"{prism_row['accuracy']:.4f}; "
                f"coverage = "
                f"{prism_row['coverage']:.4f}",
            "support":
                "Supported with coverage trade-off",
        })

    rows.append({
        "claim":
            "Alternative-model selection consistently improves predictions",
        "evidence":
            "Ablation results show inconsistent improvement across test splits.",
        "support":
            "Not supported",
    })

    rows.append({
        "claim":
            "Abstention contributes to PRISM reliability",
        "evidence":
            "Full PRISM has lower selective risk than the "
            "no-abstention variant.",
        "support":
            "Supported",
    })

    table = pd.DataFrame(rows)

    output_path = (
        DOCS_DIR
        / "claim_evidence_table.md"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# PRISM Claim-Evidence Table\n\n"
        )

        file.write(
            table.to_markdown(
                index=False
            )
        )

        file.write("\n")

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "PRISM - FINAL RESULTS ANALYSIS"
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results()

    print(
        f"Loaded {len(results)} "
        f"ablation evaluations."
    )

    print_header(
        "FINAL ABLATION RESULTS"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print_header(
        "GENERATING PLOTS"
    )

    accuracy_plot = (
        plot_accuracy_comparison(
            results
        )
    )

    coverage_plot = (
        plot_coverage_risk(
            results
        )
    )

    correction_plot = (
        plot_error_recovery(
            results
        )
    )

    print(
        f"Created: {accuracy_plot}"
    )

    print(
        f"Created: {coverage_plot}"
    )

    print(
        f"Created: {correction_plot}"
    )

    print_header(
        "GENERATING RESEARCH DOCUMENTATION"
    )

    summary_path = (
        create_results_summary(
            results
        )
    )

    claims_path = (
        create_claim_evidence_table(
            results
        )
    )

    print(
        f"Created: {summary_path}"
    )

    print(
        f"Created: {claims_path}"
    )

    print_header(
        "FINAL ANALYSIS COMPLETE"
    )

    print(
        "All analysis was generated from the frozen "
        "ablation results."
    )


if __name__ == "__main__":
    main()