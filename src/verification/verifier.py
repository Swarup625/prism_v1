"""
PRISM - Prediction Verification and Decision Module

Member 9: Verifying Predictions and Deciding What to Do

This module combines:
    1. Baseline prediction
    2. Error detector risk score
    3. Historical similarity evidence
    4. Alternative model evidence

For every prediction, PRISM makes one of three decisions:

    ACCEPT  - Keep the original baseline prediction.
    CORRECT - Replace the baseline prediction with a supported alternative.
    ABSTAIN - Do not provide a definitive prediction.

IMPORTANT:
    - All policy selection is performed using validation data only.
    - The 2025 and 2026 test sets are not loaded or used.
    - High error risk alone never causes an automatic correction.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ======================================================================
# PATH CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

DETECTOR_FILE = RESULTS_DIR / "detector_validation_predictions.parquet"
SIMILARITY_FILE = RESULTS_DIR / "neighbor_analysis.csv"
DYNAMIC_FILE = RESULTS_DIR / "dynamic_selector_predictions.parquet"

METRICS_FILE = RESULTS_DIR / "verification_metrics.csv"
AUDIT_FILE = RESULTS_DIR / "correction_audit.csv"
POLICY_FILE = DOCS_DIR / "decision_policy.md"


# ======================================================================
# CONFIGURATION
# ======================================================================

MERGE_KEYS = [
    "match_id",
    "date",
    "season",
    "over_completed",
]

# Candidate thresholds tested using validation data.
RISK_THRESHOLDS = [0.50, 0.55, 0.59, 0.65, 0.70]
SIMILARITY_CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]
SIMILARITY_SUPPORT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_divide(numerator, denominator):
    """Return NaN instead of raising an error for zero denominators."""
    if denominator == 0:
        return np.nan
    return numerator / denominator


def ensure_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_files():
    files = [
        DETECTOR_FILE,
        SIMILARITY_FILE,
        DYNAMIC_FILE,
    ]

    missing = [str(file) for file in files if not file.exists()]

    if missing:
        raise FileNotFoundError(
            "Required input files are missing:\n"
            + "\n".join(missing)
        )


# ======================================================================
# DATA LOADING AND MERGING
# ======================================================================

def load_and_merge_data():

    print("Loading verification inputs...")

    detector = pd.read_parquet(DETECTOR_FILE)
    similarity = pd.read_csv(SIMILARITY_FILE)
    dynamic = pd.read_parquet(DYNAMIC_FILE)

    # Standardize merge key data types before merging.
    for dataframe in [detector, similarity, dynamic]:

        dataframe["match_id"] = dataframe["match_id"].astype(str)
        dataframe["date"] = dataframe["date"].astype(str)
        dataframe["season"] = dataframe["season"].astype(str)

        dataframe["over_completed"] = pd.to_numeric(
            dataframe["over_completed"],
            errors="raise"
        ).astype(int)

    print(f"Error detector states : {len(detector):,}")
    print(f"Similarity states     : {len(similarity):,}")
    print(f"Dynamic selector      : {len(dynamic):,}")

    # Continue with the existing code below...
    # ------------------------------------------------------------------
    # Select only the fields required from each source.
    # ------------------------------------------------------------------

    detector_cols = MERGE_KEYS + [
        "chasing_team_won",
        "predicted_class",
        "predicted_probability",
        "confidence",
        "error_risk_score",
        "error_detector_flag",
        "selected_threshold",
    ]

    similarity_cols = MERGE_KEYS + [
        "actual_outcome",
        "similarity_probability",
        "similarity_prediction",
        "neighbor_count",
        "support_chasing_win",
        "support_chasing_loss",
        "similarity_correct",
        "similarity_confidence",
        "baseline_prediction",
        "baseline_probability",
        "model_disagrees_with_similarity",
    ]

    dynamic_cols = MERGE_KEYS + [
        "baseline_prediction",
        "baseline_probability",
        "baseline_confidence",
        "baseline_correct",
        "baseline_error",
        "max_model_agreement",
        "all_models_agree",
        "model_disagreement",
        "alternative_prediction",
        "alternative_confidence",
        "alternative_consensus",
    ]

    detector = detector[detector_cols].copy()
    similarity = similarity[similarity_cols].copy()
    dynamic = dynamic[dynamic_cols].copy()

    # Rename potentially overlapping columns.
    similarity = similarity.rename(
        columns={
            "baseline_prediction": "similarity_baseline_prediction",
            "baseline_probability": "similarity_baseline_probability",
        }
    )

    dynamic = dynamic.rename(
        columns={
            "baseline_prediction": "dynamic_baseline_prediction",
            "baseline_probability": "dynamic_baseline_probability",
        }
    )

    # ------------------------------------------------------------------
    # Merge detector + similarity evidence.
    # ------------------------------------------------------------------

    merged = detector.merge(
        similarity,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Merge dynamic model evidence.
    # ------------------------------------------------------------------

    merged = merged.merge(
        dynamic,
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise RuntimeError("No rows remain after merging verification inputs.")

    # ------------------------------------------------------------------
    # Consistency checks.
    # ------------------------------------------------------------------

    outcome_mismatch = (
        merged["chasing_team_won"]
        != merged["actual_outcome"]
    ).sum()

    baseline_mismatch_similarity = (
        merged["predicted_class"]
        != merged["similarity_baseline_prediction"]
    ).sum()

    baseline_mismatch_dynamic = (
        merged["predicted_class"]
        != merged["dynamic_baseline_prediction"]
    ).sum()

    print()
    print(f"Merged validation states : {len(merged):,}")
    print(f"Merged validation matches: {merged['match_id'].nunique():,}")
    print(f"Outcome mismatches       : {outcome_mismatch}")
    print(f"Baseline mismatch (similarity): {baseline_mismatch_similarity}")
    print(f"Baseline mismatch (dynamic)   : {baseline_mismatch_dynamic}")

    if outcome_mismatch > 0:
        raise ValueError(
            "Actual outcome mismatch detected between validation sources."
        )

    if baseline_mismatch_similarity > 0:
        raise ValueError(
            "Baseline predictions do not match between detector and "
            "similarity evidence."
        )

    if baseline_mismatch_dynamic > 0:
        raise ValueError(
            "Baseline predictions do not match between detector and "
            "dynamic selector evidence."
        )

    return merged


# ======================================================================
# EVIDENCE FUNCTIONS
# ======================================================================

def get_similarity_support(row):
    """
    Return the fraction of historical neighbors supporting the
    similarity prediction.
    """

    if row["similarity_prediction"] == 1:
        support = row["support_chasing_win"]
    else:
        support = row["support_chasing_loss"]

    return float(support)


def has_strong_similarity_evidence(row, confidence_threshold,
                                   support_threshold):
    """
    Similarity evidence is considered strong when:

        1. It disagrees with the baseline.
        2. Its confidence exceeds the threshold.
        3. Historical neighbor support exceeds the threshold.
    """

    disagrees = (
        row["similarity_prediction"]
        != row["predicted_class"]
    )

    support = get_similarity_support(row)

    return (
        disagrees
        and row["similarity_confidence"] >= confidence_threshold
        and support >= support_threshold
    )


def has_strong_model_evidence(row):
    """
    Alternative model evidence is considered strong when:

        1. The alternative models have consensus.
        2. Their alternative prediction disagrees with baseline.
        3. The alternative prediction is valid.
    """

    if not bool(row["alternative_consensus"]):
        return False

    alternative = row["alternative_prediction"]

    if pd.isna(alternative):
        return False

    return int(alternative) != int(row["predicted_class"])


# ======================================================================
# DECISION POLICY
# ======================================================================

def apply_policy(
    df,
    risk_threshold,
    similarity_confidence_threshold,
    similarity_support_threshold,
):
    """
    Apply the PRISM verification policy.

    Policy:

    ACCEPT:
        - Low risk, or
        - High risk but insufficient correction evidence.

    CORRECT:
        - High error risk AND
        - Strong similarity evidence AND
          strong alternative-model evidence supporting the same outcome.

        OR

        - High error risk AND
        - Both evidence sources agree on the same alternative.

    ABSTAIN:
        - High risk AND
        - Evidence suggests the baseline may be wrong,
        - But correction evidence is conflicting or insufficient.

    High risk alone never causes correction.
    """

    records = []

    for _, row in df.iterrows():

        baseline_prediction = int(row["predicted_class"])
        actual_outcome = int(row["chasing_team_won"])

        risk_score = float(row["error_risk_score"])
        high_risk = risk_score >= risk_threshold

        similarity_strong = has_strong_similarity_evidence(
            row,
            similarity_confidence_threshold,
            similarity_support_threshold,
        )

        model_strong = has_strong_model_evidence(row)

        similarity_prediction = int(row["similarity_prediction"])

        alternative_prediction = row["alternative_prediction"]
        if pd.notna(alternative_prediction):
            alternative_prediction = int(alternative_prediction)

        similarity_disagrees = (
            similarity_prediction != baseline_prediction
        )

        model_disagrees = (
            alternative_prediction is not None
            and alternative_prediction != baseline_prediction
        )

        # Default decision.
        decision = "ACCEPT"
        final_prediction = baseline_prediction
        reason = "Low error risk; baseline prediction retained."

        # --------------------------------------------------------------
        # Low-risk predictions are accepted directly.
        # --------------------------------------------------------------

        if not high_risk:
            decision = "ACCEPT"
            final_prediction = baseline_prediction
            reason = (
                "Error risk below selected threshold; "
                "baseline prediction retained."
            )

        else:
            # ----------------------------------------------------------
            # Check whether both correction mechanisms strongly support
            # the same alternative prediction.
            # ----------------------------------------------------------

            evidence_agrees = (
                similarity_strong
                and model_strong
                and alternative_prediction is not None
                and similarity_prediction == alternative_prediction
            )

            if evidence_agrees:

                decision = "CORRECT"
                final_prediction = similarity_prediction

                reason = (
                    "High error risk and both historical similarity "
                    "and alternative-model evidence strongly support "
                    "the same alternative prediction."
                )

            # ----------------------------------------------------------
            # If high risk exists and evidence conflicts, abstain.
            # ----------------------------------------------------------

            elif (
                similarity_strong
                or model_strong
                or similarity_disagrees
                or model_disagrees
            ):

                decision = "ABSTAIN"
                final_prediction = np.nan

                reason = (
                    "High error risk, but correction evidence is "
                    "conflicting or insufficient for a safe correction."
                )

            # ----------------------------------------------------------
            # No correction evidence: retain baseline.
            # ----------------------------------------------------------

            else:

                decision = "ACCEPT"
                final_prediction = baseline_prediction

                reason = (
                    "High error risk detected, but no supported "
                    "alternative prediction was available."
                )

        baseline_correct = (
            baseline_prediction == actual_outcome
        )

        if decision == "CORRECT":
            final_correct = (
                int(final_prediction) == actual_outcome
            )
        elif decision == "ACCEPT":
            final_correct = baseline_correct
        else:
            final_correct = np.nan

        records.append(
            {
                **row.to_dict(),

                "high_error_risk": high_risk,

                "similarity_support_for_prediction":
                    get_similarity_support(row),

                "strong_similarity_evidence":
                    similarity_strong,

                "strong_model_evidence":
                    model_strong,

                "decision": decision,
                "final_prediction": final_prediction,
                "decision_reason": reason,

                "baseline_correct_verification":
                    baseline_correct,

                "final_correct":
                    final_correct,
            }
        )

    return pd.DataFrame(records)


# ======================================================================
# METRIC CALCULATION
# ======================================================================

def calculate_metrics(df):
    """
    Calculate verification metrics.

    Definitions:

    Error Detection Rate
        Fraction of baseline errors that PRISM identifies as high risk.

    Error Recovery Rate
        Fraction of original baseline errors successfully corrected.

    Harmful Correction Rate
        Fraction of corrections that changed an originally correct
        baseline prediction into an incorrect prediction.

    Coverage
        Fraction of examples receiving ACCEPT or CORRECT.

    Selective Risk
        Error rate among returned predictions only.
    """

    total = len(df)

    baseline_correct = (
        df["predicted_class"]
        == df["chasing_team_won"]
    )

    baseline_errors = (~baseline_correct).sum()

    high_risk_errors = (
        (~baseline_correct)
        & df["high_error_risk"]
    ).sum()

    corrections = df["decision"] == "CORRECT"
    abstentions = df["decision"] == "ABSTAIN"

    returned = ~abstentions

    corrected_errors = (
        (~baseline_correct)
        & corrections
        & (
            df["final_prediction"]
            == df["chasing_team_won"]
        )
    ).sum()

    harmful_corrections = (
        baseline_correct
        & corrections
        & (
            df["final_prediction"]
            != df["chasing_team_won"]
        )
    ).sum()

    returned_predictions = df.loc[
        returned,
        "final_prediction"
    ]

    returned_actual = df.loc[
        returned,
        "chasing_team_won"
    ]

    # Classification metrics for returned predictions.
    if len(returned_predictions) > 0:

        final_accuracy = accuracy_score(
            returned_actual,
            returned_predictions,
        )

        final_precision = precision_score(
            returned_actual,
            returned_predictions,
            zero_division=0,
        )

        final_recall = recall_score(
            returned_actual,
            returned_predictions,
            zero_division=0,
        )

        final_f1 = f1_score(
            returned_actual,
            returned_predictions,
            zero_division=0,
        )

        selective_errors = (
            returned_predictions
            != returned_actual
        ).sum()

    else:

        final_accuracy = np.nan
        final_precision = np.nan
        final_recall = np.nan
        final_f1 = np.nan
        selective_errors = 0

    # Baseline metrics.
    baseline_accuracy = accuracy_score(
        df["chasing_team_won"],
        df["predicted_class"],
    )

    # Decision counts.
    accept_count = (df["decision"] == "ACCEPT").sum()
    correct_count = corrections.sum()
    abstain_count = abstentions.sum()

    metrics = {
        "total_states": total,
        "total_matches": df["match_id"].nunique(),

        "baseline_accuracy": baseline_accuracy,

        "final_accuracy": final_accuracy,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": final_f1,

        "baseline_errors": baseline_errors,
        "errors_flagged_high_risk": high_risk_errors,

        "error_detection_rate": safe_divide(
            high_risk_errors,
            baseline_errors,
        ),

        "corrections": correct_count,

        "successful_corrections": corrected_errors,

        "error_recovery_rate": safe_divide(
            corrected_errors,
            baseline_errors,
        ),

        "harmful_corrections": harmful_corrections,

        "harmful_correction_rate": safe_divide(
            harmful_corrections,
            correct_count,
        ),

        "accept_count": accept_count,
        "correct_count": correct_count,
        "abstain_count": abstain_count,

        "coverage": safe_divide(
            len(returned_predictions),
            total,
        ),

        "abstention_rate": safe_divide(
            abstain_count,
            total,
        ),

        "selective_risk": safe_divide(
            selective_errors,
            len(returned_predictions),
        ),

        "returned_predictions": len(returned_predictions),
    }

    return metrics


# ======================================================================
# POLICY SELECTION
# ======================================================================

def select_policy(validation_data):
    """
    Search a small, interpretable policy grid.

    The objective is deliberately conservative:

        1. Prefer higher final accuracy.
        2. Among comparable policies, prefer lower harmful correction.
        3. Prefer reasonable coverage.

    Only validation data is used.
    """

    print_section("VALIDATION POLICY SELECTION")

    results = []

    total_configs = (
        len(RISK_THRESHOLDS)
        * len(SIMILARITY_CONFIDENCE_THRESHOLDS)
        * len(SIMILARITY_SUPPORT_THRESHOLDS)
    )

    print(f"Candidate policies: {total_configs}")

    for risk_threshold in RISK_THRESHOLDS:

        for similarity_confidence_threshold in (
            SIMILARITY_CONFIDENCE_THRESHOLDS
        ):

            for similarity_support_threshold in (
                SIMILARITY_SUPPORT_THRESHOLDS
            ):

                evaluated = apply_policy(
                    validation_data,
                    risk_threshold,
                    similarity_confidence_threshold,
                    similarity_support_threshold,
                )

                metrics = calculate_metrics(evaluated)

                metrics.update(
                    {
                        "risk_threshold": risk_threshold,
                        "similarity_confidence_threshold":
                            similarity_confidence_threshold,
                        "similarity_support_threshold":
                            similarity_support_threshold,
                    }
                )

                results.append(metrics)

    policy_results = pd.DataFrame(results)

    # --------------------------------------------------------------
    # Conservative policy ranking.
    #
    # First maximize final accuracy.
    # Then minimize harmful correction rate.
    # Then maximize coverage.
    #
    # NaN harmful rates are treated as zero only for sorting because
    # they usually correspond to zero corrections.
    # --------------------------------------------------------------

    ranking = policy_results.copy()

    ranking["harmful_correction_rate_sort"] = (
        ranking["harmful_correction_rate"]
        .fillna(0.0)
    )

    ranking = ranking.sort_values(
        by=[
            "final_accuracy",
            "harmful_correction_rate_sort",
            "coverage",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    )

    best = ranking.iloc[0].to_dict()

    print()
    print("Top policy:")
    print(
        f"  Risk threshold                 : "
        f"{best['risk_threshold']:.2f}"
    )
    print(
        f"  Similarity confidence threshold: "
        f"{best['similarity_confidence_threshold']:.2f}"
    )
    print(
        f"  Similarity support threshold   : "
        f"{best['similarity_support_threshold']:.2f}"
    )
    print(
        f"  Final validation accuracy      : "
        f"{best['final_accuracy']:.4f}"
    )
    print(
        f"  Coverage                       : "
        f"{best['coverage']:.4f}"
    )
    print(
        f"  Harmful correction rate        : "
        f"{best['harmful_correction_rate']}"
    )

    return best, policy_results


# ======================================================================
# REPORTING
# ======================================================================

def print_final_summary(metrics, policy, runtime_seconds):

    print_section("PRISM VERIFICATION RESULTS")

    print("Decision distribution:")
    print(
        f"  ACCEPT  : {metrics['accept_count']:,}"
    )
    print(
        f"  CORRECT : {metrics['correct_count']:,}"
    )
    print(
        f"  ABSTAIN : {metrics['abstain_count']:,}"
    )

    print()
    print("Performance:")
    print(
        f"  Baseline accuracy : "
        f"{metrics['baseline_accuracy']:.4f}"
    )
    print(
        f"  Final accuracy    : "
        f"{metrics['final_accuracy']:.4f}"
    )
    print(
        f"  Coverage          : "
        f"{metrics['coverage']:.4f}"
    )
    print(
        f"  Selective risk    : "
        f"{metrics['selective_risk']:.4f}"
    )

    print()
    print("Reliability metrics:")
    print(
        f"  Error detection rate : "
        f"{metrics['error_detection_rate']:.4f}"
    )
    print(
        f"  Error recovery rate  : "
        f"{metrics['error_recovery_rate']:.4f}"
    )
    print(
        f"  Harmful corrections  : "
        f"{metrics['harmful_corrections']:,}"
    )
    print(
        f"  Harmful correction rate: "
        f"{metrics['harmful_correction_rate']}"
    )

    print()
    print("Selected policy:")
    print(
        f"  Error risk threshold: "
        f"{policy['risk_threshold']:.2f}"
    )
    print(
        f"  Similarity confidence threshold: "
        f"{policy['similarity_confidence_threshold']:.2f}"
    )
    print(
        f"  Similarity support threshold: "
        f"{policy['similarity_support_threshold']:.2f}"
    )

    print()
    print(f"Verification runtime: {runtime_seconds:.4f} seconds")


def save_policy_document(metrics, policy):

    document = f"""# PRISM Verification Decision Policy

## Purpose

The PRISM verifier combines the original baseline prediction with:

1. Error-risk estimation.
2. Historical similarity evidence.
3. Alternative-model evidence.

The verifier makes one of three decisions:

- **ACCEPT** — retain the original prediction.
- **CORRECT** — replace the original prediction with a supported alternative.
- **ABSTAIN** — return no definitive prediction.

## Selected Validation Policy

| Parameter | Selected Value |
|---|---:|
| Error-risk threshold | {policy['risk_threshold']:.2f} |
| Similarity confidence threshold | {policy['similarity_confidence_threshold']:.2f} |
| Similarity support threshold | {policy['similarity_support_threshold']:.2f} |

## Decision Rules

### ACCEPT

The original baseline prediction is retained when:

- The estimated error risk is below the selected threshold, or
- The prediction is high-risk but no sufficiently supported alternative exists.

### CORRECT

A correction is made only when:

1. The prediction has high estimated error risk.
2. Historical similarity evidence strongly supports an alternative.
3. Alternative models provide consensus for an alternative.
4. Both evidence sources support the same alternative prediction.

High error risk alone is never sufficient to trigger a correction.

### ABSTAIN

PRISM abstains when:

- Error risk is high, and
- There is evidence against the baseline prediction,
- But the available correction evidence is conflicting or insufficient.

## Validation Results

| Metric | Value |
|---|---:|
| Baseline accuracy | {metrics['baseline_accuracy']:.4f} |
| Final accuracy | {metrics['final_accuracy']:.4f} |
| Coverage | {metrics['coverage']:.4f} |
| Abstention rate | {metrics['abstention_rate']:.4f} |
| Selective risk | {metrics['selective_risk']:.4f} |
| Error detection rate | {metrics['error_detection_rate']:.4f} |
| Error recovery rate | {metrics['error_recovery_rate']:.4f} |
| Harmful corrections | {metrics['harmful_corrections']} |

## Experimental Protocol

All policy thresholds were selected using the validation dataset only.

The 2025 and 2026 test datasets were not used during verification
policy selection.
"""

    with open(POLICY_FILE, "w", encoding="utf-8") as file:
        file.write(document)


# ======================================================================
# MAIN
# ======================================================================

def main():

    start_time = time.perf_counter()

    print("\n" + "=" * 70)
    print("PRISM - PREDICTION VERIFICATION MODULE")
    print("=" * 70)

    ensure_directories()
    validate_input_files()

    # ------------------------------------------------------------------
    # Load validation evidence.
    # ------------------------------------------------------------------

    print_section("LOADING VALIDATION EVIDENCE")

    validation_data = load_and_merge_data()

    # ------------------------------------------------------------------
    # Select decision policy.
    # ------------------------------------------------------------------

    best_policy, policy_search_results = select_policy(
        validation_data
    )

    # ------------------------------------------------------------------
    # Freeze the selected policy and generate final validation decisions.
    # ------------------------------------------------------------------

    print_section("APPLYING SELECTED PRISM POLICY")

    verification_results = apply_policy(
        validation_data,
        risk_threshold=best_policy["risk_threshold"],
        similarity_confidence_threshold=(
            best_policy["similarity_confidence_threshold"]
        ),
        similarity_support_threshold=(
            best_policy["similarity_support_threshold"]
        ),
    )

    metrics = calculate_metrics(
        verification_results
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    # Add policy parameters to metrics.
    metrics.update(
        {
            "risk_threshold":
                best_policy["risk_threshold"],

            "similarity_confidence_threshold":
                best_policy[
                    "similarity_confidence_threshold"
                ],

            "similarity_support_threshold":
                best_policy[
                    "similarity_support_threshold"
                ],

            "verification_runtime_seconds":
                runtime_seconds,
        }
    )

    # ------------------------------------------------------------------
    # Save outputs.
    # ------------------------------------------------------------------

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    verification_results.to_csv(
        AUDIT_FILE,
        index=False,
    )

    policy_search_results.to_csv(
        RESULTS_DIR / "verification_policy_search.csv",
        index=False,
    )

    save_policy_document(
        metrics,
        best_policy,
    )

    # ------------------------------------------------------------------
    # Display results.
    # ------------------------------------------------------------------

    print_final_summary(
        metrics,
        best_policy,
        runtime_seconds,
    )

    print_section("CORRECTION AUDIT")

    corrections = verification_results[
        verification_results["decision"] == "CORRECT"
    ]

    if len(corrections) == 0:
        print(
            "No corrections were made under the selected "
            "conservative policy."
        )
    else:
        successful = (
            corrections["final_prediction"]
            == corrections["chasing_team_won"]
        ).sum()

        harmful = (
            (
                corrections["predicted_class"]
                == corrections["chasing_team_won"]
            )
            &
            (
                corrections["final_prediction"]
                != corrections["chasing_team_won"]
            )
        ).sum()

        print(f"Total corrections    : {len(corrections):,}")
        print(f"Successful corrections: {successful:,}")
        print(f"Harmful corrections  : {harmful:,}")

    print_section("VERIFICATION COMPLETE")

    print("Files saved:")
    print(f"  Metrics: {METRICS_FILE}")
    print(f"  Audit:   {AUDIT_FILE}")
    print(f"  Policy:  {POLICY_FILE}")
    print(
        f"  Policy search: "
        f"{RESULTS_DIR / 'verification_policy_search.csv'}"
    )

    print()
    print("IMPORTANT:")
    print(
        "All verification policy selection was performed using "
        "validation data only."
    )
    print(
        "The 2025 and 2026 test datasets remain completely untouched."
    )


if __name__ == "__main__":
    main()