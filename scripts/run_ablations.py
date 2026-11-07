"""
PRISM - FINAL ABLATION EXPERIMENTS

Evaluates the contribution of individual PRISM components using
frozen models, thresholds, and untouched test data.

Variants:
1. Baseline only
2. Baseline + Error Detector
3. Baseline + Similarity Correction
4. Baseline + Alternative Model Selection
5. Full PRISM without verification
6. Full PRISM
7. Full PRISM without abstention
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)


# ============================================================================
# IMPORT FROZEN PRISM COMPONENTS
# ============================================================================

from src.evaluation.final_evaluation import (
    DATA_PATH,
    RESULTS_DIR,
    TARGET_COLUMN,
    RISK_THRESHOLD,
    SIMILARITY_CONFIDENCE_THRESHOLD,
    SIMILARITY_SUPPORT_THRESHOLD,
    load_models,
    build_similarity_reference,
    calculate_reliability_signals,
    generate_error_risk,
    generate_similarity_evidence,
    generate_model_predictions,
    apply_prism_policy,
)

from src.evaluation.evaluation_metrics import (
    evaluate_variant,
)


# ============================================================================
# OUTPUT
# ============================================================================

OUTPUT_PATH = (
    RESULTS_DIR
    / "final_results.csv"
)


# ============================================================================
# UTILITY
# ============================================================================

def print_header(title):
    """Print a consistent experiment header."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================================
# BUILD COMMON EVALUATION DATA
# ============================================================================

def prepare_evaluation_data(
    split_data,
    models,
    similarity_reference,
):
    """
    Generate all frozen evidence used by PRISM.

    This function performs no training and no threshold tuning.
    """

    evaluation_data = (
        split_data
        .reset_index(drop=True)
        .copy()
    )

    # ----------------------------------------------------------------
    # MODEL PREDICTIONS
    # ----------------------------------------------------------------

    model_predictions = (
        generate_model_predictions(
            evaluation_data,
            models,
        )
    )

    baseline_probability = (
        model_predictions[
            "random_forest"
        ]["probability"]
    )

    baseline_prediction = (
        model_predictions[
            "random_forest"
        ]["prediction"]
    )

    evaluation_data[
        "baseline_probability"
    ] = baseline_probability

    evaluation_data[
        "baseline_prediction"
    ] = baseline_prediction

    # ----------------------------------------------------------------
    # RELIABILITY SIGNALS AND ERROR RISK
    # ----------------------------------------------------------------

    reliability_signals = (
        calculate_reliability_signals(
            baseline_probability
        )
    )

    for column in reliability_signals.columns:

        evaluation_data[column] = (
            reliability_signals[column]
        )

    error_risk_score = (
        generate_error_risk(
            reliability_signals,
            models["error_detector"],
        )
    )

    evaluation_data[
        "error_risk_score"
    ] = error_risk_score

    evaluation_data[
        "error_detector_flag"
    ] = (
        error_risk_score
        >= RISK_THRESHOLD
    ).astype(int)

    # ----------------------------------------------------------------
    # HISTORICAL SIMILARITY
    # ----------------------------------------------------------------

    similarity_evidence = (
        generate_similarity_evidence(
            evaluation_data,
            similarity_reference,
        )
    )

    for column in similarity_evidence.columns:

        evaluation_data[column] = (
            similarity_evidence[column]
        )

    # ----------------------------------------------------------------
    # ALTERNATIVE MODEL EVIDENCE
    # ----------------------------------------------------------------

    logistic_prediction = (
        model_predictions[
            "logistic_regression"
        ]["prediction"]
    )

    hgb_prediction = (
        model_predictions[
            "hist_gradient_boosting"
        ]["prediction"]
    )

    logistic_confidence = (
        model_predictions[
            "logistic_regression"
        ]["confidence"]
    )

    hgb_confidence = (
        model_predictions[
            "hist_gradient_boosting"
        ]["confidence"]
    )

    alternative_consensus = (
        (logistic_prediction == hgb_prediction)
        & (
            logistic_prediction
            != baseline_prediction
        )
    )

    evaluation_data[
        "alternative_consensus"
    ] = alternative_consensus.astype(int)

    evaluation_data[
        "alternative_prediction"
    ] = logistic_prediction

    evaluation_data[
        "alternative_confidence"
    ] = np.minimum(
        logistic_confidence,
        hgb_confidence,
    )

    return (
        evaluation_data,
        baseline_probability,
    )


# ============================================================================
# EVIDENCE MASKS
# ============================================================================

def get_similarity_supported_mask(
    evaluation_data,
):
    """
    Identify states with sufficiently strong historical evidence.
    """

    similarity_support = (
        np.maximum(
            evaluation_data[
                "support_chasing_win"
            ].to_numpy(),
            evaluation_data[
                "support_chasing_loss"
            ].to_numpy(),
        )
        / np.maximum(
            evaluation_data[
                "neighbor_count"
            ].to_numpy(),
            1,
        )
    )

    return (
        evaluation_data[
            "similarity_confidence"
        ].to_numpy()
        >= SIMILARITY_CONFIDENCE_THRESHOLD
    ) & (
        similarity_support
        >= SIMILARITY_SUPPORT_THRESHOLD
    )


# ============================================================================
# VARIANT 1: BASELINE ONLY
# ============================================================================

def run_baseline_variant(
    data,
    probabilities,
):
    """Original Random Forest only."""

    predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
    )

    decisions = np.full(
        len(data),
        "ACCEPT",
        dtype=object,
    )

    return predictions, decisions, probabilities


# ============================================================================
# VARIANT 2: ERROR DETECTOR
# ============================================================================

def run_error_detector_variant(
    data,
    probabilities,
):
    """
    Baseline + error detector.

    The detector flags potentially unreliable predictions but does
    not change the baseline prediction.
    """

    predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
    )

    decisions = np.full(
        len(data),
        "ACCEPT",
        dtype=object,
    )

    return predictions, decisions, probabilities


# ============================================================================
# VARIANT 3: SIMILARITY CORRECTION
# ============================================================================

def run_similarity_variant(
    data,
    probabilities,
):
    """
    Correct the baseline when historical similarity strongly supports
    a different prediction.
    """

    predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
        .copy()
    )

    decisions = np.full(
        len(data),
        "ACCEPT",
        dtype=object,
    )

    supported = (
        get_similarity_supported_mask(data)
    )

    disagreement = (
        data[
            "similarity_prediction"
        ].to_numpy()
        != data[
            "baseline_prediction"
        ].to_numpy()
    )

    correction_mask = (
        supported
        & disagreement
    )

    predictions[correction_mask] = (
        data.loc[
            correction_mask,
            "similarity_prediction",
        ]
        .astype(int)
        .to_numpy()
    )

    decisions[
        correction_mask
    ] = "CORRECT"

    return predictions, decisions, probabilities


# ============================================================================
# VARIANT 4: ALTERNATIVE MODEL SELECTION
# ============================================================================

def run_alternative_model_variant(
    data,
    probabilities,
):
    """
    Replace the baseline when both alternative models agree with
    each other and disagree with the baseline.
    """

    predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
        .copy()
    )

    decisions = np.full(
        len(data),
        "ACCEPT",
        dtype=object,
    )

    correction_mask = (
        data[
            "alternative_consensus"
        ].to_numpy()
        == 1
    )

    predictions[correction_mask] = (
        data.loc[
            correction_mask,
            "alternative_prediction",
        ]
        .astype(int)
        .to_numpy()
    )

    decisions[
        correction_mask
    ] = "CORRECT"

    return predictions, decisions, probabilities


# ============================================================================
# VARIANT 5: FULL PRISM WITHOUT VERIFICATION
# ============================================================================

def run_unverified_prism_variant(
    data,
    probabilities,
):
    """
    Apply available correction evidence without the conservative
    PRISM verification policy.

    High-risk predictions are corrected whenever either similarity
    evidence or alternative-model consensus disagrees with the
    baseline.
    """

    predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
        .copy()
    )

    decisions = np.full(
        len(data),
        "ACCEPT",
        dtype=object,
    )

    high_risk = (
        data[
            "error_risk_score"
        ].to_numpy()
        >= RISK_THRESHOLD
    )

    similarity_supported = (
        get_similarity_supported_mask(data)
    )

    similarity_disagrees = (
        data[
            "similarity_prediction"
        ].to_numpy()
        != data[
            "baseline_prediction"
        ].to_numpy()
    )

    alternative_consensus = (
        data[
            "alternative_consensus"
        ].to_numpy()
        == 1
    )

    # Prefer similarity evidence when available.
    similarity_correction = (
        high_risk
        & similarity_supported
        & similarity_disagrees
    )

    alternative_correction = (
        high_risk
        & (~similarity_correction)
        & alternative_consensus
    )

    predictions[
        similarity_correction
    ] = (
        data.loc[
            similarity_correction,
            "similarity_prediction",
        ]
        .astype(int)
        .to_numpy()
    )

    predictions[
        alternative_correction
    ] = (
        data.loc[
            alternative_correction,
            "alternative_prediction",
        ]
        .astype(int)
        .to_numpy()
    )

    decisions[
        similarity_correction
    ] = "CORRECT"

    decisions[
        alternative_correction
    ] = "CORRECT"

    return predictions, decisions, probabilities


# ============================================================================
# VARIANT 6: FULL PRISM
# ============================================================================

def run_full_prism_variant(
    data,
    probabilities,
):
    """Apply the complete frozen PRISM verification policy."""

    result = apply_prism_policy(
        data
    )

    # Every PRISM decision must retain a valid prediction.
    # If a prediction is missing unexpectedly, preserve the
    # original baseline prediction rather than assigning a class.
    final_prediction = (
        result["final_prediction"]
        .fillna(result["baseline_prediction"])
        .astype(int)
        .to_numpy()
    )

    decisions = (
        result["decision"]
        .fillna("ABSTAIN")
        .to_numpy()
    )

    return (
        final_prediction,
        decisions,
        probabilities,
    )
# ============================================================================
# VARIANT 7: FULL PRISM WITHOUT ABSTENTION
# ============================================================================

def run_no_abstention_variant(
    data,
    probabilities,
):
    """
    Apply full PRISM, but force abstained states to return the
    original baseline prediction.

    This isolates the contribution of abstention.
    """

    result = apply_prism_policy(
        data
    ).copy()

    # Ensure every state has a valid prediction. If the verification
    # policy produced a missing value, preserve the baseline prediction.
    result["final_prediction"] = (
        result["final_prediction"]
        .fillna(result["baseline_prediction"])
    )

    decisions = (
        result["decision"]
        .fillna("ABSTAIN")
        .to_numpy()
        .copy()
    )

    predictions = (
        result["final_prediction"]
        .to_numpy()
        .copy()
    )

    # States originally marked ABSTAIN must return the baseline
    # prediction when abstention is disabled.
    abstain_mask = (
        decisions == "ABSTAIN"
    )

    baseline_predictions = (
        result["baseline_prediction"]
        .astype(int)
        .to_numpy()
    )

    predictions[abstain_mask] = (
        baseline_predictions[abstain_mask]
    )

    # With abstention disabled, every state is considered covered.
    decisions[abstain_mask] = "ACCEPT"

    predictions = predictions.astype(int)

    return (
        predictions,
        decisions,
        probabilities,
    )
# ============================================================================
# EVALUATE ONE VARIANT
# ============================================================================

def evaluate_one_variant(
    variant_name,
    variant_function,
    data,
    probabilities,
    split_name,
):
    """Run one ablation variant and calculate all metrics."""

    start_time = time.perf_counter()

    predictions, decisions, output_probabilities = (
        variant_function(
            data,
            probabilities,
        )
    )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    y_true = (
        data[
            TARGET_COLUMN
        ].to_numpy()
    )

    baseline_predictions = (
        data[
            "baseline_prediction"
        ].to_numpy()
    )

    error_risk_flags = (
        data[
            "error_detector_flag"
        ].to_numpy()
    )

    metrics = evaluate_variant(
        y_true=y_true,
        baseline_predictions=baseline_predictions,
        final_predictions=predictions,
        decisions=decisions,
        probabilities=output_probabilities,
        error_risk_flags=error_risk_flags,
    )

    metrics["variant"] = variant_name
    metrics["split"] = split_name
    metrics["runtime_seconds"] = runtime_seconds

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "PRISM - FINAL ABLATION EXPERIMENTS"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------------

    print("\nLoading frozen dataset...")

    data = pd.read_parquet(
        DATA_PATH
    )

    train_data = data[
        data["split"] == "train"
    ].copy()

    test_splits = {
        "test_2025": data[
            data["split"] == "test_2025"
        ].copy(),

        "test_2026": data[
            data["split"] == "test_2026"
        ].copy(),
    }

    if train_data.empty:

        raise ValueError(
            "Training split is empty."
        )

    # ----------------------------------------------------------------
    # LOAD FROZEN COMPONENTS
    # ----------------------------------------------------------------

    print("\nLoading frozen models...")

    models = load_models()

    print(
        "Building training-only "
        "similarity reference..."
    )

    similarity_reference = (
        build_similarity_reference(
            train_data
        )
    )

    # ----------------------------------------------------------------
    # DEFINE ABLATIONS
    # ----------------------------------------------------------------

    variants = [
        (
            "baseline_only",
            run_baseline_variant,
        ),
        (
            "baseline_error_detector",
            run_error_detector_variant,
        ),
        (
            "similarity_correction",
            run_similarity_variant,
        ),
        (
            "alternative_model_selection",
            run_alternative_model_variant,
        ),
        (
            "full_prism_without_verification",
            run_unverified_prism_variant,
        ),
        (
            "full_prism",
            run_full_prism_variant,
        ),
        (
            "full_prism_no_abstention",
            run_no_abstention_variant,
        ),
    ]

    all_results = []

    # ----------------------------------------------------------------
    # RUN FROZEN TEST SPLITS
    # ----------------------------------------------------------------

    for split_name, split_data in (
        test_splits.items()
    ):

        print_header(
            f"PREPARING {split_name.upper()}"
        )

        evaluation_data, probabilities = (
            prepare_evaluation_data(
                split_data,
                models,
                similarity_reference,
            )
        )

        for variant_name, variant_function in variants:

            print(
                f"Running: {variant_name}"
            )

            metrics = (
                evaluate_one_variant(
                    variant_name,
                    variant_function,
                    evaluation_data,
                    probabilities,
                    split_name,
                )
            )

            all_results.append(
                metrics
            )

    # ----------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    column_order = [
        "variant",
        "split",
        "total_states",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "baseline_errors",
        "errors_flagged_high_risk",
        "error_detection_rate",
        "corrections",
        "successful_corrections",
        "error_recovery_rate",
        "harmful_corrections",
        "harmful_correction_rate",
        "coverage",
        "abstention_rate",
        "selective_risk",
        "runtime_seconds",
    ]

    results = results[
        [
            column
            for column in column_order
            if column in results.columns
        ]
    ]

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ----------------------------------------------------------------
    # DISPLAY SUMMARY
    # ----------------------------------------------------------------

    print_header(
        "FINAL ABLATION RESULTS"
    )

    display_columns = [
        "variant",
        "split",
        "accuracy",
        "coverage",
        "selective_risk",
        "error_detection_rate",
        "error_recovery_rate",
        "harmful_correction_rate",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "All variants used frozen models, "
        "training-only historical reference data, "
        "and unchanged thresholds."
    )


if __name__ == "__main__":
    main()