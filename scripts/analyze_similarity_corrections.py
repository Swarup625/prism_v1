from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "results" / "neighbor_analysis.csv"


def main():

    print("=" * 70)
    print("PRISM - SIMILARITY CORRECTION POTENTIAL ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(RESULTS_PATH)

    required_columns = [
        "actual_outcome",
        "similarity_prediction",
        "baseline_prediction",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Determine whether each system is correct.
    df["baseline_correct"] = (
        df["baseline_prediction"]
        == df["actual_outcome"]
    )

    df["similarity_correct"] = (
        df["similarity_prediction"]
        == df["actual_outcome"]
    )

    # Agreement/disagreement.
    df["systems_agree"] = (
        df["baseline_prediction"]
        == df["similarity_prediction"]
    )

    total = len(df)

    both_correct = (
        df["baseline_correct"]
        & df["similarity_correct"]
    ).sum()

    baseline_wrong_similarity_correct = (
        ~df["baseline_correct"]
        & df["similarity_correct"]
    ).sum()

    baseline_correct_similarity_wrong = (
        df["baseline_correct"]
        & ~df["similarity_correct"]
    ).sum()

    both_wrong = (
        ~df["baseline_correct"]
        & ~df["similarity_correct"]
    ).sum()

    disagreements = (~df["systems_agree"]).sum()

    print("\nOVERALL RESULTS")
    print("-" * 70)

    print(f"Total validation states : {total:,}")
    print(f"System disagreements    : {disagreements:,} "
          f"({disagreements / total:.2%})")

    print("\nCORRECTION POTENTIAL")
    print("-" * 70)

    print(
        f"Both correct                         : "
        f"{both_correct:,} ({both_correct / total:.2%})"
    )

    print(
        f"Baseline wrong -> Similarity correct : "
        f"{baseline_wrong_similarity_correct:,} "
        f"({baseline_wrong_similarity_correct / total:.2%})"
    )

    print(
        f"Baseline correct -> Similarity wrong : "
        f"{baseline_correct_similarity_wrong:,} "
        f"({baseline_correct_similarity_wrong / total:.2%})"
    )

    print(
        f"Both wrong                           : "
        f"{both_wrong:,} ({both_wrong / total:.2%})"
    )

    # Among disagreements, determine which model is correct.
    disagreement_df = df[~df["systems_agree"]]

    print("\nDISAGREEMENT ANALYSIS")
    print("-" * 70)

    if len(disagreement_df) > 0:

        similarity_wins = (
            ~disagreement_df["baseline_correct"]
            & disagreement_df["similarity_correct"]
        ).sum()

        baseline_wins = (
            disagreement_df["baseline_correct"]
            & ~disagreement_df["similarity_correct"]
        ).sum()

        print(
            f"Similarity correct in disagreements : "
            f"{similarity_wins:,} "
            f"({similarity_wins / len(disagreement_df):.2%})"
        )

        print(
            f"Baseline correct in disagreements   : "
            f"{baseline_wins:,} "
            f"({baseline_wins / len(disagreement_df):.2%})"
        )

    # Calculate the hypothetical result of always switching
    # to similarity when the systems disagree.
    hypothetical_prediction = (
        df["baseline_prediction"].copy()
    )

    disagreement_mask = ~df["systems_agree"]

    hypothetical_prediction.loc[
        disagreement_mask
    ] = df.loc[
        disagreement_mask,
        "similarity_prediction"
    ]

    hypothetical_accuracy = (
        hypothetical_prediction
        == df["actual_outcome"]
    ).mean()

    baseline_accuracy = df["baseline_correct"].mean()

    print("\nHYPOTHETICAL 'ALWAYS SWITCH' POLICY")
    print("-" * 70)

    print(
        f"Baseline accuracy              : "
        f"{baseline_accuracy:.4f}"
    )

    print(
        f"Accuracy after always switching: "
        f"{hypothetical_accuracy:.4f}"
    )

    print(
        f"Accuracy change                : "
        f"{hypothetical_accuracy - baseline_accuracy:+.4f}"
    )

    print("\nINTERPRETATION")
    print("-" * 70)

    if baseline_wrong_similarity_correct > baseline_correct_similarity_wrong:
        print(
            "Similarity shows positive correction potential: "
            "it fixes more baseline errors than it would damage "
            "correct baseline predictions."
        )
    else:
        print(
            "Similarity should not be used as an automatic "
            "correction mechanism. It may still provide useful "
            "evidence to the PRISM verifier."
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()