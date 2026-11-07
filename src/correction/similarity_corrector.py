"""
PRISM - Historical Similarity Corrector

Member 7: Finding Similar Historical Situations

This module finds historical IPL match states similar to a current
match situation and uses their outcomes as an independent source of
prediction evidence.

Leakage policy:
    - Historical reference states come only from the training split.
    - Validation states are used only for evaluation and parameter tuning.
    - Reserved 2025 and 2026 test sets are not used.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "match_states.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)


# =====================================================================
# CONFIGURATION
# =====================================================================

FEATURES = [
    "current_score",
    "wickets_lost",
    "runs_required",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
    "runs_last_3_overs",
    "wickets_last_3_overs",
]

TARGET = "chasing_team_won"

# Candidate neighborhood sizes to evaluate on validation data.
K_VALUES = [5, 10, 15, 25, 50]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """
    Calculate classification metrics from predicted probabilities.
    """
    probabilities = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
    }


def find_neighbors(reference_features, query_features, k):
    """
    Find K nearest historical states.

    Returns:
        distances : Distance to each neighbor.
        indices   : Row indices of the neighbors in the reference set.
    """
    model = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
    )

    model.fit(reference_features)

    distances, indices = model.kneighbors(
        query_features,
        return_distance=True,
    )

    return distances, indices


def build_similarity_predictions(
    reference_df,
    query_df,
    distances,
    indices,
):
    """
    Convert neighbor outcomes into similarity-based predictions.

    The predicted probability is the fraction of historical neighbors
    in which the chasing team won.

    Also records evidence statistics for each query state.
    """

    reference_outcomes = reference_df[TARGET].to_numpy()

    records = []

    for query_position in range(len(query_df)):
        neighbor_indices = indices[query_position]
        neighbor_distances = distances[query_position]

        neighbor_outcomes = reference_outcomes[neighbor_indices]

        alternative_probability = float(np.mean(neighbor_outcomes))
        alternative_class = int(alternative_probability >= 0.5)

        support_for_win = int(np.sum(neighbor_outcomes == 1))
        support_for_loss = int(np.sum(neighbor_outcomes == 0))

        records.append({
            "match_id": query_df.iloc[query_position]["match_id"],
            "date": query_df.iloc[query_position]["date"],
            "season": query_df.iloc[query_position]["season"],
            "over_completed": query_df.iloc[query_position]["over_completed"],

            "actual_outcome": int(
                query_df.iloc[query_position][TARGET]
            ),

            "similarity_probability": alternative_probability,
            "similarity_prediction": alternative_class,

            "neighbor_count": len(neighbor_indices),
            "support_chasing_win": support_for_win,
            "support_chasing_loss": support_for_loss,

            "mean_neighbor_distance": float(
                np.mean(neighbor_distances)
            ),
            "min_neighbor_distance": float(
                np.min(neighbor_distances)
            ),
            "max_neighbor_distance": float(
                np.max(neighbor_distances)
            ),
        })

    return pd.DataFrame(records)


# =====================================================================
# MAIN
# =====================================================================

def main():

    print_header(
        "PRISM - HISTORICAL SIMILARITY CORRECTOR"
    )

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    print("\nLoading match-state dataset...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Total states : {len(df):,}")
    print(f"Total matches: {df['match_id'].nunique():,}")

    missing_features = [
        feature for feature in FEATURES + [TARGET, "split"]
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required columns: {missing_features}"
        )

    # -----------------------------------------------------------------
    # Create leakage-safe splits
    # -----------------------------------------------------------------

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    validation_df = (
        df[df["split"] == "validation"]
        .copy()
        .reset_index(drop=True)
    )

    test_2025_df = df[df["split"] == "test_2025"]
    test_2026_df = df[df["split"] == "test_2026"]

    print_header("DATA SPLITS")

    print(
        f"Historical reference states (train): "
        f"{len(train_df):,}"
    )
    print(
        f"Historical reference matches      : "
        f"{train_df['match_id'].nunique():,}"
    )

    print(
        f"\nValidation states                : "
        f"{len(validation_df):,}"
    )
    print(
        f"Validation matches               : "
        f"{validation_df['match_id'].nunique():,}"
    )

    print(
        f"\nReserved 2025 test states        : "
        f"{len(test_2025_df):,}"
    )
    print(
        f"Reserved 2026 test states        : "
        f"{len(test_2026_df):,}"
    )

    # Safety check.
    train_matches = set(train_df["match_id"])
    validation_matches = set(validation_df["match_id"])

    overlap = train_matches.intersection(validation_matches)

    if overlap:
        raise RuntimeError(
            "Leakage detected: training and validation matches overlap."
        )

    print("\nPASS: Training and validation matches do not overlap.")
    print(
        "PASS: Reserved test datasets are not used for tuning."
    )

    # -----------------------------------------------------------------
    # Feature scaling
    # -----------------------------------------------------------------

    print_header("SIMILARITY FEATURE CONFIGURATION")

    print("\nFeatures used for historical similarity:")

    for feature in FEATURES:
        print(f"  - {feature}")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train_df[FEATURES]
    )

    X_validation = scaler.transform(
        validation_df[FEATURES]
    )

    y_validation = validation_df[TARGET].to_numpy()

    # -----------------------------------------------------------------
    # Tune K using validation data
    # -----------------------------------------------------------------

    print_header(
        "VALIDATION SEARCH FOR NEIGHBORHOOD SIZE"
    )

    tuning_results = []

    start_time = time.perf_counter()

    max_k = max(K_VALUES)

    # Find the maximum required number of neighbors once.
    all_distances, all_indices = find_neighbors(
        X_train,
        X_validation,
        max_k,
    )

    for k in K_VALUES:

        distances = all_distances[:, :k]
        indices = all_indices[:, :k]

        neighbor_outcomes = (
            train_df[TARGET]
            .to_numpy()[indices]
        )

        probabilities = neighbor_outcomes.mean(axis=1)

        metrics = calculate_metrics(
            y_validation,
            probabilities,
        )

        tuning_results.append({
            "k": k,
            **metrics,
            "mean_neighbor_distance": float(
                distances.mean()
            ),
        })

        print(
            f"K = {k:<3} | "
            f"Accuracy = {metrics['accuracy']:.4f} | "
            f"F1 = {metrics['f1_score']:.4f} | "
            f"ROC-AUC = {metrics['roc_auc']:.4f}"
        )

    tuning_df = pd.DataFrame(tuning_results)

    # Select K using validation ROC-AUC.
    best_row = tuning_df.loc[
        tuning_df["roc_auc"].idxmax()
    ]

    best_k = int(best_row["k"])

    tuning_time = (
        time.perf_counter() - start_time
    )

    print_header(
        "SELECTED SIMILARITY CONFIGURATION"
    )

    print(f"Selected K          : {best_k}")
    print(
        "Selection criterion : "
        "highest validation ROC-AUC"
    )
    print(
        f"Validation ROC-AUC  : "
        f"{best_row['roc_auc']:.4f}"
    )
    print(
        f"Validation accuracy : "
        f"{best_row['accuracy']:.4f}"
    )
    print(
        f"Validation F1       : "
        f"{best_row['f1_score']:.4f}"
    )

    # -----------------------------------------------------------------
    # Build detailed predictions for selected K
    # -----------------------------------------------------------------

    print_header(
        "BUILDING HISTORICAL EVIDENCE"
    )

    distances = all_distances[:, :best_k]
    indices = all_indices[:, :best_k]

    predictions_df = build_similarity_predictions(
        train_df,
        validation_df,
        distances,
        indices,
    )

    # Add correctness information.
    predictions_df["similarity_correct"] = (
        predictions_df["similarity_prediction"]
        == predictions_df["actual_outcome"]
    ).astype(int)

    predictions_df["similarity_confidence"] = np.maximum(
        predictions_df["similarity_probability"],
        1 - predictions_df["similarity_probability"],
    )

    # -----------------------------------------------------------------
    # Compare with baseline prediction
    # -----------------------------------------------------------------

    baseline_path = (
        RESULTS_DIR
        / "baseline_validation_predictions.parquet"
    )

    if baseline_path.exists():

        print(
            "\nLoading frozen baseline validation predictions..."
        )

        baseline_df = pd.read_parquet(
            baseline_path
        )

        merge_columns = [
            "match_id",
            "date",
            "season",
            "over_completed",
        ]

        baseline_columns = merge_columns + [
            "predicted_class",
            "predicted_probability",
        ]

        baseline_subset = baseline_df[
            baseline_columns
        ].copy()

        baseline_subset = baseline_subset.rename(
            columns={
                "predicted_class": "baseline_prediction",
                "predicted_probability": "baseline_probability",
            }
        )

        predictions_df = predictions_df.merge(
            baseline_subset,
            on=merge_columns,
            how="left",
            validate="one_to_one",
        )

        if predictions_df[
            "baseline_prediction"
        ].isna().any():
            print(
                "WARNING: Some validation states could not be "
                "matched with baseline predictions."
            )
        else:

            predictions_df[
                "model_disagrees_with_similarity"
            ] = (
                predictions_df["baseline_prediction"]
                != predictions_df["similarity_prediction"]
            ).astype(int)

            disagreement_rate = (
                predictions_df[
                    "model_disagrees_with_similarity"
                ].mean()
            )

            print(
                f"Baseline/similarity disagreement rate: "
                f"{disagreement_rate:.2%}"
            )

    else:
        print(
            "\nWARNING: Baseline validation predictions not found."
        )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    output_path = (
        RESULTS_DIR / "neighbor_analysis.csv"
    )

    predictions_df.to_csv(
        output_path,
        index=False,
    )

    tuning_path = (
        RESULTS_DIR
        / "similarity_tuning_results.csv"
    )

    tuning_df.to_csv(
        tuning_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Create experiment report
    # -----------------------------------------------------------------

    report_path = (
        DOCS_DIR
        / "similarity_correction_report.md"
    )

    best_metrics = calculate_metrics(
        y_validation,
        predictions_df[
            "similarity_probability"
        ].to_numpy(),
    )

    report = f"""# PRISM Historical Similarity Corrector

## Purpose

The similarity corrector provides an independent source of evidence by
finding historical IPL match situations that resemble a current match
state.

It does not modify the original baseline prediction directly. Instead,
it produces an alternative prediction together with evidence about the
historical situations supporting that prediction.

## Leakage-Safe Design

Historical reference data contains only the training split.

- Training reference states: {len(train_df):,}
- Training matches: {train_df['match_id'].nunique():,}
- Validation states: {len(validation_df):,}
- Validation matches: {validation_df['match_id'].nunique():,}

Validation data was used only to select the neighborhood size.

The reserved 2025 and 2026 test sets were not used.

## Similarity Features

The following match-state variables were standardized before computing
Euclidean distances:

{chr(10).join(f"- `{feature}`" for feature in FEATURES)}

## Selected Configuration

- Neighborhood size (K): {best_k}
- Selection criterion: Highest validation ROC-AUC
- Tuning runtime: {tuning_time:.4f} seconds

## Validation Performance

| Metric | Value |
|---|---:|
| Accuracy | {best_metrics['accuracy']:.4f} |
| Precision | {best_metrics['precision']:.4f} |
| Recall | {best_metrics['recall']:.4f} |
| F1 Score | {best_metrics['f1_score']:.4f} |
| ROC-AUC | {best_metrics['roc_auc']:.4f} |
| Log Loss | {best_metrics['log_loss']:.4f} |

## Evidence Produced

For every validation prediction, the system records:

- Similarity-based win probability
- Alternative predicted class
- Number of historical neighbors
- Historical support for chasing-team wins and losses
- Mean, minimum, and maximum neighbor distances

## Role in PRISM

The similarity corrector is an evidence source for the later PRISM
verification stage. A disagreement between the baseline model and
historical similarity does not automatically trigger a correction.

A later verification policy will combine this evidence with error-risk
signals and alternative model predictions before deciding whether to:

- ACCEPT the original prediction
- CORRECT the prediction
- ABSTAIN from making a definitive prediction

## Limitations

The current implementation uses Euclidean distance in a standardized
feature space. Similarity quality may depend on feature selection,
distance weighting, and the choice of neighborhood size. These settings
should remain frozen before final testing.
"""

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    print_header(
        "SIMILARITY CORRECTOR COMPLETE"
    )

    print(
        f"\nValidation predictions: "
        f"{len(predictions_df):,}"
    )

    print(
        f"Selected K            : {best_k}"
    )

    print(
        f"\nFiles saved:"
        f"\n  {output_path}"
        f"\n  {tuning_path}"
        f"\n  {report_path}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe 2025 and 2026 test sets have not been used."
        "\nThe similarity corrector currently provides evidence;"
        "\nit does not automatically replace baseline predictions."
    )


if __name__ == "__main__":
    main()