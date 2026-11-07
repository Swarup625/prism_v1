"""
PRISM - COMMON EVALUATION METRICS

Provides consistent metric calculations for all PRISM ablation
experiments.

All metrics are calculated on frozen test predictions. No model
training or threshold tuning occurs in this module.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================================
# BASIC METRIC UTILITIES
# ============================================================================

def safe_roc_auc(y_true, probabilities):
    """
    Calculate ROC-AUC safely.

    Returns NaN when only one class is present in the evaluation data.
    """

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        probabilities,
    )


# ============================================================================
# STANDARD CLASSIFICATION METRICS
# ============================================================================

def calculate_classification_metrics(
    y_true,
    predictions,
    probabilities=None,
):
    """
    Calculate standard binary classification metrics.

    Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC, when probabilities are available
    """

    results = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }

    if probabilities is not None:

        results["roc_auc"] = safe_roc_auc(
            y_true,
            probabilities,
        )

    else:

        results["roc_auc"] = np.nan

    return results


# ============================================================================
# PRISM RELIABILITY METRICS
# ============================================================================

def calculate_reliability_metrics(
    y_true,
    baseline_predictions,
    final_predictions,
    decisions,
    error_risk_flags=None,
):
    """
    Calculate PRISM-specific reliability metrics.

    Definitions
    -----------

    Error Detection Rate:
        Number of baseline errors flagged as high risk
        divided by total baseline errors.

    Error Recovery Rate:
        Number of baseline errors successfully corrected
        divided by total baseline errors.

    Harmful Correction Rate:
        Number of corrections that changed an originally correct
        baseline prediction into an incorrect prediction
        divided by total corrections.

    Coverage:
        Number of non-abstained predictions
        divided by total evaluation examples.

    Selective Risk:
        Error rate among returned predictions.
    """

    y_true = np.asarray(y_true)
    baseline_predictions = np.asarray(
        baseline_predictions
    )
    final_predictions = np.asarray(
        final_predictions
    )
    decisions = np.asarray(decisions)

    total_examples = len(y_true)

    baseline_errors = (
        baseline_predictions != y_true
    )

    total_baseline_errors = int(
        np.sum(baseline_errors)
    )

    # ----------------------------------------------------------------
    # ERROR DETECTION
    # ----------------------------------------------------------------

    if error_risk_flags is not None:

        error_risk_flags = np.asarray(
            error_risk_flags
        ).astype(bool)

        errors_flagged = int(
            np.sum(
                baseline_errors
                & error_risk_flags
            )
        )

        error_detection_rate = (
            errors_flagged
            / total_baseline_errors
            if total_baseline_errors > 0
            else np.nan
        )

    else:

        errors_flagged = 0
        error_detection_rate = np.nan

    # ----------------------------------------------------------------
    # CORRECTION METRICS
    # ----------------------------------------------------------------

    correction_mask = (
        decisions == "CORRECT"
    )

    corrections = int(
        np.sum(correction_mask)
    )

    successful_corrections = int(
        np.sum(
            correction_mask
            & baseline_errors
            & (final_predictions == y_true)
        )
    )

    harmful_corrections = int(
        np.sum(
            correction_mask
            & (~baseline_errors)
            & (final_predictions != y_true)
        )
    )

    error_recovery_rate = (
        successful_corrections
        / total_baseline_errors
        if total_baseline_errors > 0
        else np.nan
    )

    harmful_correction_rate = (
        harmful_corrections
        / corrections
        if corrections > 0
        else 0.0
    )

    # ----------------------------------------------------------------
    # SELECTIVE PREDICTION METRICS
    # ----------------------------------------------------------------

    returned_mask = (
        decisions != "ABSTAIN"
    )

    returned_count = int(
        np.sum(returned_mask)
    )

    abstain_count = int(
        total_examples - returned_count
    )

    coverage = (
        returned_count / total_examples
        if total_examples > 0
        else np.nan
    )

    abstention_rate = (
        abstain_count / total_examples
        if total_examples > 0
        else np.nan
    )

    if returned_count > 0:

        selective_risk = float(
            np.mean(
                final_predictions[returned_mask]
                != y_true[returned_mask]
            )
        )

    else:

        selective_risk = np.nan

    return {
        "total_states": total_examples,

        "baseline_errors":
            total_baseline_errors,

        "errors_flagged_high_risk":
            errors_flagged,

        "error_detection_rate":
            error_detection_rate,

        "corrections":
            corrections,

        "successful_corrections":
            successful_corrections,

        "error_recovery_rate":
            error_recovery_rate,

        "harmful_corrections":
            harmful_corrections,

        "harmful_correction_rate":
            harmful_correction_rate,

        "returned_predictions":
            returned_count,

        "abstentions":
            abstain_count,

        "coverage":
            coverage,

        "abstention_rate":
            abstention_rate,

        "selective_risk":
            selective_risk,
    }


# ============================================================================
# COMPLETE VARIANT EVALUATION
# ============================================================================

def evaluate_variant(
    y_true,
    baseline_predictions,
    final_predictions,
    decisions,
    probabilities=None,
    error_risk_flags=None,
):
    """
    Calculate all metrics for one PRISM variant.

    Classification metrics are calculated on returned predictions.
    Reliability metrics are calculated relative to the original
    baseline predictions.
    """

    y_true = np.asarray(y_true)
    final_predictions = np.asarray(
        final_predictions
    )
    decisions = np.asarray(decisions)

    returned_mask = (
        decisions != "ABSTAIN"
    )

    if np.sum(returned_mask) > 0:

        returned_probabilities = None

        if probabilities is not None:

            probabilities = np.asarray(
                probabilities
            )

            returned_probabilities = (
                probabilities[returned_mask]
            )

        classification_metrics = (
            calculate_classification_metrics(
                y_true[returned_mask],
                final_predictions[returned_mask],
                returned_probabilities,
            )
        )

    else:

        classification_metrics = {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "roc_auc": np.nan,
        }

    reliability_metrics = (
        calculate_reliability_metrics(
            y_true=y_true,
            baseline_predictions=baseline_predictions,
            final_predictions=final_predictions,
            decisions=decisions,
            error_risk_flags=error_risk_flags,
        )
    )

    return {
        **classification_metrics,
        **reliability_metrics,
    }