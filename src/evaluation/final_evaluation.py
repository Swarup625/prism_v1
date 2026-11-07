"""
PRISM - FINAL EVALUATION

Evaluates the frozen PRISM pipeline on the untouched 2025 and 2026
test datasets.

IMPORTANT:
- No model training occurs in this script.
- No thresholds are tuned on test data.
- Historical similarity uses training data only.
- The validation-selected PRISM policy is applied unchanged.
"""

from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "match_states.parquet"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
HGB_MODEL_PATH = MODEL_DIR / "hist_gradient_boosting.joblib"
ERROR_DETECTOR_PATH = MODEL_DIR / "error_detector.joblib"


# ============================================================================
# FROZEN CONFIGURATION
# ============================================================================

BASELINE_MODEL_NAME = "random_forest"

# Selected during validation policy search.
# These values MUST NOT be changed using test results.
RISK_THRESHOLD = 0.50
SIMILARITY_CONFIDENCE_THRESHOLD = 0.65
SIMILARITY_SUPPORT_THRESHOLD = 0.50

# Selected during validation for the historical similarity corrector.
SIMILARITY_K = 50


# Features used by the baseline ML models.
MODEL_FEATURES = [
    "over_completed",
    "current_score",
    "wickets_lost",
    "target",
    "runs_required",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
    "runs_last_3_overs",
    "wickets_last_3_overs",
    "venue",
    "batting_team",
    "bowling_team",
]

# Features used for historical similarity.
SIMILARITY_FEATURES = [
    "current_score",
    "wickets_lost",
    "runs_required",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
    "runs_last_3_overs",
    "wickets_last_3_overs",
]

TARGET_COLUMN = "chasing_team_won"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    """Print a consistent section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_roc_auc(y_true, probabilities):
    """Calculate ROC-AUC safely."""
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(y_true, probabilities)


def calculate_reliability_signals(probabilities):
    """
    Calculate the same reliability signals used by the PRISM pipeline.

    Parameters
    ----------
    probabilities : array-like
        Probability of chasing_team_won = 1.

    Returns
    -------
    DataFrame
        confidence, probability_margin and predictive_entropy.
    """

    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    confidence = np.maximum(probabilities, 1 - probabilities)

    probability_margin = np.abs(probabilities - (1 - probabilities))

    entropy = -(
        probabilities * np.log2(probabilities)
        + (1 - probabilities) * np.log2(1 - probabilities)
    )

    return pd.DataFrame(
        {
            "confidence": confidence,
            "probability_margin": probability_margin,
            "predictive_entropy": entropy,
        }
    )


# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models():
    """Load all frozen PRISM models."""

    print_header("LOADING FROZEN MODELS")

    print("Loading Logistic Regression...")
    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)

    print("Loading Random Forest baseline...")
    random_forest_model = joblib.load(RF_MODEL_PATH)

    print("Loading HistGradientBoosting...")
    hist_gradient_boosting_model = joblib.load(HGB_MODEL_PATH)

    print("Loading Error Detector...")
    error_detector_bundle = joblib.load(ERROR_DETECTOR_PATH)

    if not isinstance(error_detector_bundle, dict):
        raise TypeError(
            "Expected error_detector.joblib to contain a dictionary."
        )

    required_keys = {"model", "features", "threshold"}

    missing_keys = required_keys - set(error_detector_bundle.keys())

    if missing_keys:
        raise KeyError(
            f"Error detector bundle is missing keys: {missing_keys}"
        )

    print("\nPASS: All frozen models loaded successfully.")

    return {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model,
        "hist_gradient_boosting": hist_gradient_boosting_model,
        "error_detector": error_detector_bundle,
    }


# ============================================================================
# BUILD HISTORICAL SIMILARITY REFERENCE
# ============================================================================

def build_similarity_reference(train_data):
    """
    Fit the historical similarity system using training data only.

    Test data is never used to fit the scaler or neighbor index.
    """

    print_header("BUILDING HISTORICAL SIMILARITY REFERENCE")

    reference_data = train_data.reset_index(drop=True).copy()

    scaler = StandardScaler()

    reference_features = scaler.fit_transform(
        reference_data[SIMILARITY_FEATURES]
    )

    n_neighbors = min(SIMILARITY_K, len(reference_data))

    neighbor_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
    )

    neighbor_model.fit(reference_features)

    print(f"Historical reference states : {len(reference_data):,}")
    print(
        f"Historical reference matches: "
        f"{reference_data['match_id'].nunique():,}"
    )
    print(f"Neighborhood size (K)       : {n_neighbors}")

    print("\nPASS: Similarity reference fitted using training data only.")

    return {
        "data": reference_data,
        "scaler": scaler,
        "neighbors": neighbor_model,
    }


# ============================================================================
# GENERATE SIMILARITY EVIDENCE
# ============================================================================

def generate_similarity_evidence(
    evaluation_data,
    similarity_reference,
):
    """
    Generate historical neighbor evidence for evaluation states.

    Returns the historical win probability, prediction, confidence,
    support counts and distance statistics.
    """

    reference_data = similarity_reference["data"]
    scaler = similarity_reference["scaler"]
    neighbor_model = similarity_reference["neighbors"]

    evaluation_features = scaler.transform(
        evaluation_data[SIMILARITY_FEATURES]
    )

    distances, indices = neighbor_model.kneighbors(
        evaluation_features
    )

    rows = []

    for row_index in range(len(evaluation_data)):

        neighbor_indices = indices[row_index]
        neighbor_distances = distances[row_index]

        neighbor_outcomes = reference_data.iloc[
            neighbor_indices
        ][TARGET_COLUMN].to_numpy()

        similarity_probability = float(
            np.mean(neighbor_outcomes)
        )

        similarity_prediction = int(
            similarity_probability >= 0.5
        )

        similarity_confidence = max(
            similarity_probability,
            1 - similarity_probability,
        )

        support_chasing_win = int(
            np.sum(neighbor_outcomes == 1)
        )

        support_chasing_loss = int(
            np.sum(neighbor_outcomes == 0)
        )

        rows.append(
            {
                "similarity_probability": similarity_probability,
                "similarity_prediction": similarity_prediction,
                "similarity_confidence": similarity_confidence,
                "neighbor_count": len(neighbor_outcomes),
                "support_chasing_win": support_chasing_win,
                "support_chasing_loss": support_chasing_loss,
                "mean_neighbor_distance": float(
                    np.mean(neighbor_distances)
                ),
                "min_neighbor_distance": float(
                    np.min(neighbor_distances)
                ),
                "max_neighbor_distance": float(
                    np.max(neighbor_distances)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# GENERATE MODEL EVIDENCE
# ============================================================================

def generate_model_predictions(
    evaluation_data,
    models,
):
    """Generate predictions from all frozen baseline models."""

    X = evaluation_data[MODEL_FEATURES]

    results = {}

    for model_name in [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]:

        model = models[model_name]

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        confidence = np.maximum(
            probabilities,
            1 - probabilities,
        )

        results[model_name] = {
            "probability": probabilities,
            "prediction": predictions,
            "confidence": confidence,
        }

    return results


# ============================================================================
# GENERATE ERROR RISK
# ============================================================================

def generate_error_risk(
    reliability_signals,
    error_detector_bundle,
):
    """Calculate risk scores using the frozen error detector."""

    detector = error_detector_bundle["model"]
    detector_features = error_detector_bundle["features"]

    missing_features = set(detector_features) - set(
        reliability_signals.columns
    )

    if missing_features:
        raise KeyError(
            f"Missing detector features: {missing_features}"
        )

    X_detector = reliability_signals[detector_features]

    risk_scores = detector.predict_proba(
        X_detector
    )[:, 1]

    return risk_scores


# ============================================================================
# APPLY FROZEN PRISM POLICY
# ============================================================================

def apply_prism_policy(data):
    """
    Apply the frozen PRISM verification policy.

    Decision logic:
    1. Low-risk baseline predictions are ACCEPTED.
    2. High-risk predictions can be CORRECTED only when historical
       similarity provides sufficiently strong and opposing evidence.
    3. Remaining high-risk predictions are ABSTAINED.

    The policy is deliberately conservative.
    """

    decisions = []
    final_predictions = []
    decision_reasons = []

    for _, row in data.iterrows():

        baseline_prediction = int(
            row["baseline_prediction"]
        )

        high_risk = (
            row["error_risk_score"]
            >= RISK_THRESHOLD
        )

        similarity_disagrees = (
            int(row["similarity_prediction"])
            != baseline_prediction
        )

        sufficient_similarity_confidence = (
            row["similarity_confidence"]
            >= SIMILARITY_CONFIDENCE_THRESHOLD
        )

        # The selected support threshold refers to the fraction of
        # neighbors supporting the proposed similarity prediction.
        if row["similarity_prediction"] == 1:
            similarity_support = (
                row["support_chasing_win"]
                / row["neighbor_count"]
            )
        else:
            similarity_support = (
                row["support_chasing_loss"]
                / row["neighbor_count"]
            )

        sufficient_similarity_support = (
            similarity_support
            >= SIMILARITY_SUPPORT_THRESHOLD
        )

        # --------------------------------------------------------------
        # ACCEPT
        # --------------------------------------------------------------

        if not high_risk:

            decisions.append("ACCEPT")
            final_predictions.append(baseline_prediction)
            decision_reasons.append(
                "Baseline prediction below frozen error-risk threshold."
            )

        # --------------------------------------------------------------
        # CORRECT
        # --------------------------------------------------------------

        elif (
            similarity_disagrees
            and sufficient_similarity_confidence
            and sufficient_similarity_support
        ):

            decisions.append("CORRECT")
            final_predictions.append(
                int(row["similarity_prediction"])
            )

            decision_reasons.append(
                "High-risk baseline prediction with strong "
                "historical similarity evidence supporting "
                "an alternative outcome."
            )

        # --------------------------------------------------------------
        # ABSTAIN
        # --------------------------------------------------------------

        else:

            decisions.append("ABSTAIN")
            final_predictions.append(np.nan)

            decision_reasons.append(
                "Prediction considered high-risk without sufficient "
                "evidence for a safe correction."
            )

    result = data.copy()

    result["decision"] = decisions
    result["final_prediction"] = final_predictions
    result["decision_reason"] = decision_reasons

    return result


# ============================================================================
# CALCULATE METRICS
# ============================================================================

def calculate_results(
    evaluation_data,
    baseline_probabilities,
):
    """Calculate baseline and PRISM evaluation metrics."""

    y_true = evaluation_data[TARGET_COLUMN].to_numpy()

    baseline_predictions = evaluation_data[
        "baseline_prediction"
    ].astype(int).to_numpy()

    baseline_accuracy = accuracy_score(
        y_true,
        baseline_predictions,
    )

    baseline_precision = precision_score(
        y_true,
        baseline_predictions,
        zero_division=0,
    )

    baseline_recall = recall_score(
        y_true,
        baseline_predictions,
        zero_division=0,
    )

    baseline_f1 = f1_score(
        y_true,
        baseline_predictions,
        zero_division=0,
    )

    baseline_roc_auc = safe_roc_auc(
        y_true,
        baseline_probabilities,
    )

    baseline_errors = (
        baseline_predictions != y_true
    )

    high_risk = (
        evaluation_data["error_risk_score"].to_numpy()
        >= RISK_THRESHOLD
    )

    errors_flagged = np.sum(
        baseline_errors & high_risk
    )

    total_baseline_errors = int(
        np.sum(baseline_errors)
    )

    error_detection_rate = (
        errors_flagged / total_baseline_errors
        if total_baseline_errors > 0
        else np.nan
    )

    returned_mask = (
        evaluation_data["decision"] != "ABSTAIN"
    ).to_numpy()

    returned_predictions = evaluation_data.loc[
        returned_mask,
        "final_prediction",
    ].astype(int).to_numpy()

    returned_truth = y_true[returned_mask]

    coverage = float(np.mean(returned_mask))

    abstention_rate = 1 - coverage

    if len(returned_predictions) > 0:

        final_accuracy = accuracy_score(
            returned_truth,
            returned_predictions,
        )

        final_precision = precision_score(
            returned_truth,
            returned_predictions,
            zero_division=0,
        )

        final_recall = recall_score(
            returned_truth,
            returned_predictions,
            zero_division=0,
        )

        final_f1 = f1_score(
            returned_truth,
            returned_predictions,
            zero_division=0,
        )

        selective_risk = 1 - final_accuracy

    else:

        final_accuracy = np.nan
        final_precision = np.nan
        final_recall = np.nan
        final_f1 = np.nan
        selective_risk = np.nan

    correction_mask = (
        evaluation_data["decision"] == "CORRECT"
    ).to_numpy()

    corrections = int(
        np.sum(correction_mask)
    )

    successful_corrections = int(
        np.sum(
            correction_mask
            & baseline_errors
            & (
                evaluation_data["final_prediction"].to_numpy()
                == y_true
            )
        )
    )

    harmful_corrections = int(
        np.sum(
            correction_mask
            & (~baseline_errors)
            & (
                evaluation_data["final_prediction"].to_numpy()
                != y_true
            )
        )
    )

    error_recovery_rate = (
        successful_corrections / total_baseline_errors
        if total_baseline_errors > 0
        else np.nan
    )

    harmful_correction_rate = (
        harmful_corrections / corrections
        if corrections > 0
        else 0.0
    )

    return {
        "total_states": len(evaluation_data),
        "total_matches": evaluation_data[
            "match_id"
        ].nunique(),

        "baseline_accuracy": baseline_accuracy,
        "baseline_precision": baseline_precision,
        "baseline_recall": baseline_recall,
        "baseline_f1": baseline_f1,
        "baseline_roc_auc": baseline_roc_auc,

        "baseline_errors": total_baseline_errors,
        "errors_flagged_high_risk": int(errors_flagged),
        "error_detection_rate": error_detection_rate,

        "final_accuracy": final_accuracy,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": final_f1,

        "corrections": corrections,
        "successful_corrections": successful_corrections,
        "error_recovery_rate": error_recovery_rate,
        "harmful_corrections": harmful_corrections,
        "harmful_correction_rate": harmful_correction_rate,

        "accept_count": int(
            np.sum(
                evaluation_data["decision"] == "ACCEPT"
            )
        ),
        "correct_count": corrections,
        "abstain_count": int(
            np.sum(
                evaluation_data["decision"] == "ABSTAIN"
            )
        ),

        "coverage": coverage,
        "abstention_rate": abstention_rate,
        "selective_risk": selective_risk,
        "returned_predictions": int(
            np.sum(returned_mask)
        ),

        "risk_threshold": RISK_THRESHOLD,
        "similarity_confidence_threshold":
            SIMILARITY_CONFIDENCE_THRESHOLD,
        "similarity_support_threshold":
            SIMILARITY_SUPPORT_THRESHOLD,
    }


# ============================================================================
# EVALUATE ONE TEST SPLIT
# ============================================================================

def evaluate_split(
    split_name,
    split_data,
    models,
    similarity_reference,
):
    """Run the complete frozen PRISM pipeline on one test split."""

    print_header(
        f"EVALUATING {split_name.upper()}"
    )

    start_time = time.perf_counter()

    evaluation_data = split_data.reset_index(
        drop=True
    ).copy()

    print(
        f"States  : {len(evaluation_data):,}"
    )
    print(
        f"Matches : "
        f"{evaluation_data['match_id'].nunique():,}"
    )

    # ----------------------------------------------------------------
    # BASELINE AND ALTERNATIVE MODEL PREDICTIONS
    # ----------------------------------------------------------------

    print("\nGenerating frozen model predictions...")

    model_predictions = generate_model_predictions(
        evaluation_data,
        models,
    )

    baseline_probability = model_predictions[
        BASELINE_MODEL_NAME
    ]["probability"]

    baseline_prediction = model_predictions[
        BASELINE_MODEL_NAME
    ]["prediction"]

    evaluation_data[
        "baseline_probability"
    ] = baseline_probability

    evaluation_data[
        "baseline_prediction"
    ] = baseline_prediction

    # ----------------------------------------------------------------
    # RELIABILITY SIGNALS
    # ----------------------------------------------------------------

    print("Calculating reliability signals...")

    reliability_signals = (
        calculate_reliability_signals(
            baseline_probability
        )
    )

    for column in reliability_signals.columns:
        evaluation_data[column] = (
            reliability_signals[column]
        )

    # ----------------------------------------------------------------
    # ERROR DETECTOR
    # ----------------------------------------------------------------

    print("Calculating error risk scores...")

    error_risk_score = generate_error_risk(
        reliability_signals,
        models["error_detector"],
    )

    evaluation_data[
        "error_risk_score"
    ] = error_risk_score

    evaluation_data[
        "error_detector_flag"
    ] = (
        error_risk_score >= RISK_THRESHOLD
    ).astype(int)

    # ----------------------------------------------------------------
    # HISTORICAL SIMILARITY
    # ----------------------------------------------------------------

    print("Generating historical similarity evidence...")

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

    print("Generating alternative model evidence...")

    for model_name, prediction_data in (
        model_predictions.items()
    ):

        evaluation_data[
            f"{model_name}_prediction"
        ] = prediction_data["prediction"]

        evaluation_data[
            f"{model_name}_probability"
        ] = prediction_data["probability"]

        evaluation_data[
            f"{model_name}_confidence"
        ] = prediction_data["confidence"]

    all_predictions = np.column_stack(
        [
            model_predictions[
                "logistic_regression"
            ]["prediction"],
            model_predictions[
                "random_forest"
            ]["prediction"],
            model_predictions[
                "hist_gradient_boosting"
            ]["prediction"],
        ]
    )

    votes_for_1 = np.sum(
        all_predictions == 1,
        axis=1,
    )

    votes_for_0 = 3 - votes_for_1

    evaluation_data["votes_for_0"] = votes_for_0
    evaluation_data["votes_for_1"] = votes_for_1

    evaluation_data[
        "max_model_agreement"
    ] = np.maximum(
        votes_for_0,
        votes_for_1,
    )

    evaluation_data[
        "all_models_agree"
    ] = (
        evaluation_data[
            "max_model_agreement"
        ] == 3
    ).astype(int)

    evaluation_data[
        "model_disagreement"
    ] = (
        evaluation_data[
            "all_models_agree"
        ] == 0
    ).astype(int)

    # Alternative consensus:
    # Both non-baseline models agree with each other and disagree
    # with the Random Forest baseline.
    logistic_prediction = model_predictions[
        "logistic_regression"
    ]["prediction"]

    hgb_prediction = model_predictions[
        "hist_gradient_boosting"
    ]["prediction"]

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
        model_predictions[
            "logistic_regression"
        ]["confidence"],
        model_predictions[
            "hist_gradient_boosting"
        ]["confidence"],
    )

    # ----------------------------------------------------------------
    # APPLY FROZEN PRISM POLICY
    # ----------------------------------------------------------------

    print("Applying frozen PRISM verification policy...")

    evaluation_data = apply_prism_policy(
        evaluation_data
    )

    runtime_seconds = (
        time.perf_counter() - start_time
    )

    # ----------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------

    metrics = calculate_results(
        evaluation_data,
        baseline_probability,
    )

    metrics["split"] = split_name
    metrics["runtime_seconds"] = runtime_seconds

    # ----------------------------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------------------------

    print_header(
        f"{split_name.upper()} RESULTS"
    )

    print(
        f"Baseline accuracy : "
        f"{metrics['baseline_accuracy']:.4f}"
    )

    print(
        f"PRISM accuracy    : "
        f"{metrics['final_accuracy']:.4f}"
    )

    print(
        f"Coverage          : "
        f"{metrics['coverage']:.4f}"
    )

    print(
        f"Selective risk    : "
        f"{metrics['selective_risk']:.4f}"
    )

    print("\nDecisions:")
    print(
        f"  ACCEPT  : "
        f"{metrics['accept_count']}"
    )
    print(
        f"  CORRECT : "
        f"{metrics['correct_count']}"
    )
    print(
        f"  ABSTAIN : "
        f"{metrics['abstain_count']}"
    )

    print("\nCorrections:")
    print(
        f"  Total      : "
        f"{metrics['corrections']}"
    )
    print(
        f"  Successful : "
        f"{metrics['successful_corrections']}"
    )
    print(
        f"  Harmful    : "
        f"{metrics['harmful_corrections']}"
    )

    print(
        f"\nRuntime: "
        f"{runtime_seconds:.4f} seconds"
    )

    return evaluation_data, metrics


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("PRISM - FINAL FROZEN TEST EVALUATION")
    print("=" * 70)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------------

    print("\nLoading match-state dataset...")

    data = pd.read_parquet(
        DATA_PATH
    )

    print(
        f"Total states : {len(data):,}"
    )

    print(
        f"Total matches: "
        f"{data['match_id'].nunique():,}"
    )

    # ----------------------------------------------------------------
    # SPLIT DATA
    # ----------------------------------------------------------------

    train_data = data[
        data["split"] == "train"
    ].copy()

    test_2025_data = data[
        data["split"] == "test_2025"
    ].copy()

    test_2026_data = data[
        data["split"] == "test_2026"
    ].copy()

    if (
        train_data.empty
        or test_2025_data.empty
        or test_2026_data.empty
    ):
        raise ValueError(
            "One or more required dataset splits are empty."
        )

    print_header("FROZEN EVALUATION CONFIGURATION")

    print(
        f"Baseline model : "
        f"{BASELINE_MODEL_NAME}"
    )

    print(
        f"Risk threshold : "
        f"{RISK_THRESHOLD:.2f}"
    )

    print(
        f"Similarity confidence threshold : "
        f"{SIMILARITY_CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Similarity support threshold : "
        f"{SIMILARITY_SUPPORT_THRESHOLD:.2f}"
    )

    print(
        f"Historical neighbors (K) : "
        f"{SIMILARITY_K}"
    )

    print(
        "\nIMPORTANT: No models or thresholds will be "
        "trained or tuned on the test data."
    )

    # ----------------------------------------------------------------
    # LOAD FROZEN MODELS
    # ----------------------------------------------------------------

    models = load_models()

    # ----------------------------------------------------------------
    # BUILD TRAINING-ONLY HISTORICAL REFERENCE
    # ----------------------------------------------------------------

    similarity_reference = (
        build_similarity_reference(
            train_data
        )
    )

    # ----------------------------------------------------------------
    # EVALUATE TEST 2025
    # ----------------------------------------------------------------

    test_2025_results, test_2025_metrics = (
        evaluate_split(
            "test_2025",
            test_2025_data,
            models,
            similarity_reference,
        )
    )

    # ----------------------------------------------------------------
    # EVALUATE TEST 2026
    # ----------------------------------------------------------------

    test_2026_results, test_2026_metrics = (
        evaluate_split(
            "test_2026",
            test_2026_data,
            models,
            similarity_reference,
        )
    )

    # ----------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------

    print_header("SAVING FINAL EVALUATION RESULTS")

    test_2025_path = (
        RESULTS_DIR
        / "final_predictions_test_2025.parquet"
    )

    test_2026_path = (
        RESULTS_DIR
        / "final_predictions_test_2026.parquet"
    )

    metrics_path = (
        RESULTS_DIR
        / "final_evaluation_metrics.csv"
    )

    test_2025_results.to_parquet(
        test_2025_path,
        index=False,
    )

    test_2026_results.to_parquet(
        test_2026_path,
        index=False,
    )

    metrics_df = pd.DataFrame(
        [
            test_2025_metrics,
            test_2026_metrics,
        ]
    )

    # Put split first for readability.
    columns = [
        "split"
    ] + [
        column
        for column in metrics_df.columns
        if column != "split"
    ]

    metrics_df = metrics_df[columns]

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"Test 2025 predictions:\n"
        f"  {test_2025_path}"
    )

    print(
        f"\nTest 2026 predictions:\n"
        f"  {test_2026_path}"
    )

    print(
        f"\nFinal metrics:\n"
        f"  {metrics_path}"
    )

    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------

    print_header("FINAL EVALUATION SUMMARY")

    display_columns = [
        "split",
        "total_states",
        "total_matches",
        "baseline_accuracy",
        "final_accuracy",
        "coverage",
        "selective_risk",
        "corrections",
        "successful_corrections",
        "harmful_corrections",
    ]

    print(
        metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FINAL FROZEN TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:\n"
        "The test results must now be treated as final evaluation "
        "results. Do not use them to retune the models or thresholds."
    )


if __name__ == "__main__":
    main()