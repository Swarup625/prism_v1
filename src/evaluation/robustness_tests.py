"""
PRISM - ROBUSTNESS AND FAILURE TESTING

Evaluates the frozen PRISM pipeline under difficult conditions.

IMPORTANT:
- No model training occurs in this script.
- No thresholds are tuned on test data.
- The frozen models and validation-selected PRISM policy are used unchanged.
- The 2025 and 2026 test datasets remain evaluation-only datasets.
"""

from pathlib import Path
import sys
import time
import copy

import joblib
import numpy as np
import pandas as pd
import yaml

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

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_DIR = PROJECT_ROOT / "configs"

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "match_states.parquet"
)

CONFIG_PATH = (
    CONFIG_DIR
    / "failure_scenarios.yaml"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "robustness_results.csv"
)


# ============================================================================
# FROZEN PRISM CONFIGURATION
# ============================================================================

BASELINE_MODEL_NAME = "random_forest"

RISK_THRESHOLD = 0.50
SIMILARITY_CONFIDENCE_THRESHOLD = 0.65
SIMILARITY_SUPPORT_THRESHOLD = 0.50
SIMILARITY_K = 50

TARGET_COLUMN = "chasing_team_won"

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
    """Calculate PRISM reliability signals."""

    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-12,
        1 - 1e-12,
    )

    confidence = np.maximum(
        probabilities,
        1 - probabilities,
    )

    probability_margin = np.abs(
        probabilities - (1 - probabilities)
    )

    predictive_entropy = -(
        probabilities * np.log2(probabilities)
        + (1 - probabilities)
        * np.log2(1 - probabilities)
    )

    return pd.DataFrame({
        "confidence": confidence,
        "probability_margin": probability_margin,
        "predictive_entropy": predictive_entropy,
    })


# ============================================================================
# DATA LOADING AND FROZEN SPLITS
# ============================================================================

def load_dataset():
    """Load the complete match-state dataset."""

    print_header("LOADING MATCH-STATE DATASET")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    data = pd.read_parquet(DATASET_PATH)

    print(f"Total states : {len(data):,}")
    print(
        f"Total matches: "
        f"{data['match_id'].nunique():,}"
    )

    return data


def create_splits(data):
    """
    Create frozen season-based splits.

    Season values may contain formats such as:
    - 2023
    - 2020/21

    The first four-digit year is used for split assignment.
    """

    season_year = (
        data["season"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
        .astype(int)
    )

    train_data = data[
        season_year <= 2023
    ].copy()

    validation_data = data[
        season_year == 2024
    ].copy()

    test_2025 = data[
        season_year == 2025
    ].copy()

    test_2026 = data[
        season_year == 2026
    ].copy()

    print_header("FROZEN DATA SPLITS")

    print(
        f"Training states : {len(train_data):,}"
    )
    print(
        f"Training matches: "
        f"{train_data['match_id'].nunique():,}"
    )

    print(
        f"\nValidation states : "
        f"{len(validation_data):,}"
    )
    print(
        f"Validation matches: "
        f"{validation_data['match_id'].nunique():,}"
    )

    print(
        f"\nTest 2025 states : "
        f"{len(test_2025):,}"
    )
    print(
        f"Test 2025 matches: "
        f"{test_2025['match_id'].nunique():,}"
    )

    print(
        f"\nTest 2026 states : "
        f"{len(test_2026):,}"
    )
    print(
        f"Test 2026 matches: "
        f"{test_2026['match_id'].nunique():,}"
    )

    return {
        "train": train_data,
        "validation": validation_data,
        "test_2025": test_2025,
        "test_2026": test_2026,
    }


def load_frozen_splits(data):
    """
    Load the exact frozen dataset splits used by the final evaluation.

    The split assignments are already stored in the dataset and must
    not be recreated from the season column.
    """

    if "split" not in data.columns:
        raise ValueError(
            "Dataset does not contain the required 'split' column."
        )

    train_data = data[
        data["split"] == "train"
    ].copy()

    validation_data = data[
        data["split"] == "validation"
    ].copy()

    test_2025_data = data[
        data["split"] == "test_2025"
    ].copy()

    test_2026_data = data[
        data["split"] == "test_2026"
    ].copy()

    if train_data.empty:
        raise ValueError("Frozen training split is empty.")

    if test_2025_data.empty:
        raise ValueError("Frozen test_2025 split is empty.")

    if test_2026_data.empty:
        raise ValueError("Frozen test_2026 split is empty.")

    print_header("LOADING FROZEN DATA SPLITS")

    print(
        f"Training states : {len(train_data):,}"
    )
    print(
        f"Training matches: "
        f"{train_data['match_id'].nunique():,}"
    )

    print(
        f"\nValidation states : "
        f"{len(validation_data):,}"
    )
    print(
        f"Validation matches: "
        f"{validation_data['match_id'].nunique():,}"
    )

    print(
        f"\nTest 2025 states : "
        f"{len(test_2025_data):,}"
    )
    print(
        f"Test 2025 matches: "
        f"{test_2025_data['match_id'].nunique():,}"
    )

    print(
        f"\nTest 2026 states : "
        f"{len(test_2026_data):,}"
    )
    print(
        f"Test 2026 matches: "
        f"{test_2026_data['match_id'].nunique():,}"
    )

    return {
        "train": train_data,
        "validation": validation_data,
        "test_2025": test_2025_data,
        "test_2026": test_2026_data,
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all frozen PRISM models."""

    print_header("LOADING FROZEN MODELS")

    model_paths = {
        "logistic_regression":
            MODELS_DIR / "logistic_regression.joblib",
        "random_forest":
            MODELS_DIR / "random_forest.joblib",
        "hist_gradient_boosting":
            MODELS_DIR / "hist_gradient_boosting.joblib",
        "error_detector":
            MODELS_DIR / "error_detector.joblib",
    }

    models = {}

    for name, path in model_paths.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{path}"
            )

        print(f"Loading {name}...")
        models[name] = joblib.load(path)

    print("\nPASS: All frozen models loaded.")

    return models


# ============================================================================
# SIMILARITY REFERENCE
# ============================================================================

def build_similarity_reference(train_data):
    """Build historical similarity reference using training data only."""

    print_header(
        "BUILDING HISTORICAL SIMILARITY REFERENCE"
    )

    reference_data = train_data[
        SIMILARITY_FEATURES + [TARGET_COLUMN]
    ].copy()

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        reference_data[SIMILARITY_FEATURES]
    )

    neighbors = NearestNeighbors(
        n_neighbors=SIMILARITY_K,
        metric="euclidean",
    )

    neighbors.fit(scaled_features)

    print(
        f"Historical states : "
        f"{len(reference_data):,}"
    )

    return {
        "data": reference_data,
        "scaler": scaler,
        "neighbors": neighbors,
    }


def generate_similarity_evidence(
    evaluation_data,
    similarity_reference,
):
    """Generate historical similarity evidence."""

    reference_data = similarity_reference["data"]
    scaler = similarity_reference["scaler"]
    neighbors = similarity_reference["neighbors"]

    query_features = (
        evaluation_data[
            SIMILARITY_FEATURES
        ]
        .copy()
        .fillna(
            reference_data[
                SIMILARITY_FEATURES
            ].median()
        )
    )

    query_scaled = scaler.transform(
        query_features
    )

    distances, indices = neighbors.kneighbors(
        query_scaled
    )

    neighbor_outcomes = (
        reference_data[TARGET_COLUMN]
        .to_numpy()[indices]
    )

    similarity_probability = (
        neighbor_outcomes.mean(axis=1)
    )

    similarity_prediction = (
        similarity_probability >= 0.5
    ).astype(int)

    similarity_confidence = np.maximum(
        similarity_probability,
        1 - similarity_probability,
    )

    support_chasing_win = np.sum(
        neighbor_outcomes == 1,
        axis=1,
    )

    support_chasing_loss = np.sum(
        neighbor_outcomes == 0,
        axis=1,
    )

    return pd.DataFrame({
        "similarity_probability":
            similarity_probability,
        "similarity_prediction":
            similarity_prediction,
        "similarity_confidence":
            similarity_confidence,
        "neighbor_count":
            len(reference_data)
            if len(reference_data) < SIMILARITY_K
            else SIMILARITY_K,
        "support_chasing_win":
            support_chasing_win,
        "support_chasing_loss":
            support_chasing_loss,
        "mean_neighbor_distance":
            distances.mean(axis=1),
        "min_neighbor_distance":
            distances.min(axis=1),
        "max_neighbor_distance":
            distances.max(axis=1),
    })


# ============================================================================
# MODEL PREDICTIONS
# ============================================================================

def generate_model_predictions(
    evaluation_data,
    models,
):
    """Generate predictions from all frozen baseline models."""

    X = evaluation_data[MODEL_FEATURES]

    predictions = {}

    for model_name in [
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]:

        model = models[model_name]

        probabilities = model.predict_proba(X)[:, 1]

        predicted_class = (
            probabilities >= 0.5
        ).astype(int)

        confidence = np.maximum(
            probabilities,
            1 - probabilities,
        )

        predictions[model_name] = {
            "probability": probabilities,
            "prediction": predicted_class,
            "confidence": confidence,
        }

    return predictions


def generate_error_risk(
    reliability_signals,
    detector_bundle,
):
    """Generate frozen error detector risk scores."""

    detector = detector_bundle["model"]
    features = detector_bundle["features"]

    return detector.predict_proba(
        reliability_signals[features]
    )[:, 1]


# ============================================================================
# PRISM VERIFICATION POLICY
# ============================================================================

def apply_prism_policy(data):
    """
    Apply the frozen PRISM verification policy.

    ACCEPT:
        Keep baseline prediction when risk is below threshold.

    CORRECT:
        Replace baseline only when high-risk and supported by
        strong historical similarity evidence.

    ABSTAIN:
        Do not return a definitive prediction when the baseline
        is high-risk without sufficient correction evidence.
    """

    result = data.copy()

    decisions = []
    final_predictions = []
    decision_reasons = []

    for _, row in result.iterrows():

        baseline_prediction = int(
            row["baseline_prediction"]
        )

        high_risk = (
            row["error_risk_score"]
            >= RISK_THRESHOLD
        )

        similarity_support = max(
            row["support_chasing_win"],
            row["support_chasing_loss"],
        ) / max(
            row["neighbor_count"],
            1,
        )

        similarity_supported = (
            row["similarity_confidence"]
            >= SIMILARITY_CONFIDENCE_THRESHOLD
            and similarity_support
            >= SIMILARITY_SUPPORT_THRESHOLD
        )

        similarity_disagrees = (
            int(row["similarity_prediction"])
            != baseline_prediction
        )

        alternative_supported = (
            int(row["alternative_consensus"]) == 1
        )

        if not high_risk:

            decisions.append("ACCEPT")
            final_predictions.append(
                baseline_prediction
            )
            decision_reasons.append(
                "low_error_risk"
            )

        elif (
            similarity_disagrees
            and similarity_supported
        ):

            decisions.append("CORRECT")
            final_predictions.append(
                int(row["similarity_prediction"])
            )
            decision_reasons.append(
                "high_risk_similarity_supported"
            )

        elif (
            alternative_supported
            and similarity_supported
        ):

            decisions.append("CORRECT")
            final_predictions.append(
                int(row["alternative_prediction"])
            )
            decision_reasons.append(
                "high_risk_alternative_supported"
            )

        else:

            decisions.append("ABSTAIN")
            final_predictions.append(
                baseline_prediction
            )
            decision_reasons.append(
                "high_risk_insufficient_evidence"
            )

    result["decision"] = decisions
    result["final_prediction"] = final_predictions
    result["decision_reason"] = decision_reasons

    return result


# ============================================================================
# METRIC CALCULATION
# ============================================================================

def calculate_results(
    evaluation_data,
    baseline_probabilities,
):
    """Calculate baseline and PRISM robustness metrics."""

    y_true = evaluation_data[
        TARGET_COLUMN
    ].to_numpy()

    baseline_predictions = (
        evaluation_data[
            "baseline_prediction"
        ]
        .astype(int)
        .to_numpy()
    )

    baseline_errors = (
        baseline_predictions != y_true
    )

    total_baseline_errors = int(
        np.sum(baseline_errors)
    )

    high_risk = (
        evaluation_data[
            "error_risk_score"
        ].to_numpy()
        >= RISK_THRESHOLD
    )

    errors_flagged = int(
        np.sum(
            baseline_errors & high_risk
        )
    )

    returned_mask = (
        evaluation_data["decision"]
        != "ABSTAIN"
    ).to_numpy()

    returned_predictions = (
        evaluation_data.loc[
            returned_mask,
            "final_prediction",
        ]
        .astype(int)
        .to_numpy()
    )

    returned_truth = y_true[returned_mask]

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

    coverage = float(
        np.mean(returned_mask)
    )

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

        selective_risk = (
            1 - final_accuracy
        )

    else:

        final_accuracy = np.nan
        final_precision = np.nan
        final_recall = np.nan
        final_f1 = np.nan
        selective_risk = np.nan

    correction_mask = (
        evaluation_data["decision"]
        == "CORRECT"
    ).to_numpy()

    corrections = int(
        np.sum(correction_mask)
    )

    final_predictions = (
        evaluation_data[
            "final_prediction"
        ]
        .astype(int)
        .to_numpy()
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

    error_detection_rate = (
        errors_flagged / total_baseline_errors
        if total_baseline_errors > 0
        else np.nan
    )

    error_recovery_rate = (
        successful_corrections
        / total_baseline_errors
        if total_baseline_errors > 0
        else np.nan
    )

    harmful_correction_rate = (
        harmful_corrections / corrections
        if corrections > 0
        else 0.0
    )

    return {
        "total_states":
            len(evaluation_data),

        "total_matches":
            evaluation_data[
                "match_id"
            ].nunique(),

        "baseline_accuracy":
            baseline_accuracy,

        "baseline_precision":
            baseline_precision,

        "baseline_recall":
            baseline_recall,

        "baseline_f1":
            baseline_f1,

        "baseline_roc_auc":
            baseline_roc_auc,

        "final_accuracy":
            final_accuracy,

        "final_precision":
            final_precision,

        "final_recall":
            final_recall,

        "final_f1":
            final_f1,

        "coverage":
            coverage,

        "selective_risk":
            selective_risk,

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
    }


# ============================================================================
# SCENARIO MODIFICATIONS
# ============================================================================

def apply_noise(
    data,
    features,
    noise_level,
    rng,
):
    """Add relative Gaussian noise to numerical features."""

    modified = data.copy()

    for feature in features:

        if feature not in modified.columns:
            continue

        values = modified[feature].astype(float)

        scale = np.maximum(
            np.abs(values),
            1.0,
        )

        noise = rng.normal(
            loc=0.0,
            scale=noise_level * scale,
            size=len(modified),
        )

        modified[feature] = (
            values + noise
        )

    return modified


def apply_missing_features(
    data,
    features,
):
    """
    Simulate unavailable features.

    Missing numerical values are replaced with NaN. The baseline
    preprocessing pipeline may not support NaN directly, so median
    values from the evaluation data are used as a deterministic
    fallback for model compatibility.
    """

    modified = data.copy()

    for feature in features:

        if feature not in modified.columns:
            continue

        if pd.api.types.is_numeric_dtype(
            modified[feature]
        ):

            median_value = (
                modified[feature]
                .median()
            )

            modified[feature] = median_value

        else:

            modified[feature] = (
                modified[feature]
                .mode()
                .iloc[0]
            )

    return modified


def filter_difficult_states(
    evaluation_data,
    scenario_name,
    scenario_config,
    baseline_confidence,
):
    """Select difficult subsets without modifying the test data."""

    mask = np.ones(
        len(evaluation_data),
        dtype=bool,
    )

    if scenario_name == "low_confidence":

        max_confidence = scenario_config[
            "max_confidence"
        ]

        mask &= (
            baseline_confidence
            <= max_confidence
        )

    elif scenario_name == "late_match":

        min_over = scenario_config[
            "min_over_completed"
        ]

        mask &= (
            evaluation_data[
                "over_completed"
            ].astype(float)
            >= min_over
        )

    elif scenario_name == "high_required_rate":

        minimum_rate = scenario_config[
            "min_required_run_rate"
        ]

        mask &= (
            evaluation_data[
                "required_run_rate"
            ].astype(float)
            >= minimum_rate
        )

    return evaluation_data.loc[
        mask
    ].copy()


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def evaluate_scenario(
    scenario_name,
    split_name,
    split_data,
    models,
    similarity_reference,
):
    """Run the complete frozen PRISM pipeline."""

    start_time = time.perf_counter()

    evaluation_data = (
        split_data
        .reset_index(drop=True)
        .copy()
    )

    model_predictions = (
        generate_model_predictions(
            evaluation_data,
            models,
        )
    )

    baseline_probability = (
        model_predictions[
            BASELINE_MODEL_NAME
        ]["probability"]
    )

    baseline_prediction = (
        model_predictions[
            BASELINE_MODEL_NAME
        ]["prediction"]
    )

    evaluation_data[
        "baseline_probability"
    ] = baseline_probability

    evaluation_data[
        "baseline_prediction"
    ] = baseline_prediction

    reliability_signals = (
        calculate_reliability_signals(
            baseline_probability
        )
    )

    for column in reliability_signals.columns:

        evaluation_data[column] = (
            reliability_signals[column]
        )

    error_risk_score = generate_error_risk(
        reliability_signals,
        models["error_detector"],
    )

    evaluation_data[
        "error_risk_score"
    ] = error_risk_score

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

    all_predictions = np.column_stack([
        model_predictions[
            "logistic_regression"
        ]["prediction"],

        model_predictions[
            "random_forest"
        ]["prediction"],

        model_predictions[
            "hist_gradient_boosting"
        ]["prediction"],
    ])

    votes_for_1 = np.sum(
        all_predictions == 1,
        axis=1,
    )

    votes_for_0 = (
        all_predictions.shape[1]
        - votes_for_1
    )

    evaluation_data[
        "votes_for_0"
    ] = votes_for_0

    evaluation_data[
        "votes_for_1"
    ] = votes_for_1

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

    evaluation_data = apply_prism_policy(
        evaluation_data
    )

    metrics = calculate_results(
        evaluation_data,
        baseline_probability,
    )

    metrics["scenario"] = scenario_name
    metrics["split"] = split_name
    metrics["runtime_seconds"] = (
        time.perf_counter() - start_time
    )

    return metrics


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():

    print("=" * 70)
    print(
        "PRISM - ROBUSTNESS AND FAILURE TESTING"
    )
    print("=" * 70)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration not found:\n{CONFIG_PATH}"
        )

    print_header(
        "LOADING ROBUSTNESS CONFIGURATION"
    )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    print(
        f"Experiment : "
        f"{config['experiment']['name']}"
    )

    random_seed = config[
        "experiment"
    ]["random_seed"]

    print(
        f"Random seed: {random_seed}"
    )

    rng = np.random.default_rng(
        random_seed
    )

    # ----------------------------------------------------------------
    # LOAD DATA AND MODELS
    # ----------------------------------------------------------------

    data = load_dataset()

    splits = load_frozen_splits(
        data
    )

    models = load_models()

    similarity_reference = (
        build_similarity_reference(
            splits["train"]
        )
    )

    test_splits = config[
        "datasets"
    ]["test_splits"]

    results = []

    # ----------------------------------------------------------------
    # CLEAN EVALUATION
    # ----------------------------------------------------------------

    if config.get(
        "clean",
        {},
    ).get("enabled", False):

        print_header(
            "SCENARIO: CLEAN DATA"
        )

        for split_name in test_splits:

            print(
                f"Evaluating {split_name}..."
            )

            metrics = evaluate_scenario(
                "clean",
                split_name,
                splits[split_name],
                models,
                similarity_reference,
            )

            results.append(metrics)

            print(
                f"  PRISM accuracy: "
                f"{metrics['final_accuracy']:.4f}"
            )

    # ----------------------------------------------------------------
    # NOISE SCENARIOS
    # ----------------------------------------------------------------

    noise_config = config.get(
        "noise",
        {},
    )

    if noise_config.get(
        "enabled",
        False,
    ):

        for noise_level in noise_config[
            "levels"
        ]:

            scenario_name = (
                f"noise_{noise_level:.2f}"
            )

            print_header(
                f"SCENARIO: {scenario_name.upper()}"
            )

            for split_name in test_splits:

                modified_data = apply_noise(
                    splits[split_name],
                    noise_config["features"],
                    noise_level,
                    rng,
                )

                metrics = evaluate_scenario(
                    scenario_name,
                    split_name,
                    modified_data,
                    models,
                    similarity_reference,
                )

                results.append(metrics)

                print(
                    f"{split_name}: "
                    f"PRISM accuracy = "
                    f"{metrics['final_accuracy']:.4f}"
                )

    # ----------------------------------------------------------------
    # MISSING FEATURE SCENARIOS
    # ----------------------------------------------------------------

    missing_config = config.get(
        "missing_features",
        {},
    )

    if missing_config.get(
        "enabled",
        False,
    ):

        for scenario_name, scenario in (
            missing_config[
                "scenarios"
            ].items()
        ):

            print_header(
                f"SCENARIO: {scenario_name.upper()}"
            )

            for split_name in test_splits:

                modified_data = (
                    apply_missing_features(
                        splits[split_name],
                        scenario["features"],
                    )
                )

                metrics = evaluate_scenario(
                    scenario_name,
                    split_name,
                    modified_data,
                    models,
                    similarity_reference,
                )

                results.append(metrics)

                print(
                    f"{split_name}: "
                    f"PRISM accuracy = "
                    f"{metrics['final_accuracy']:.4f}"
                )

    # ----------------------------------------------------------------
    # DIFFICULT STATE SCENARIOS
    # ----------------------------------------------------------------

    difficult_config = config.get(
        "difficult_states",
        {},
    )

    if difficult_config.get(
        "enabled",
        False,
    ):

        print_header(
            "IDENTIFYING DIFFICULT MATCH STATES"
        )

        for scenario_name, scenario in (
            difficult_config[
                "scenarios"
            ].items()
        ):

            for split_name in test_splits:

                original_data = (
                    splits[split_name]
                    .reset_index(drop=True)
                    .copy()
                )

                # Generate baseline confidence for subset selection.
                baseline_model = models[
                    BASELINE_MODEL_NAME
                ]

                probabilities = (
                    baseline_model.predict_proba(
                        original_data[
                            MODEL_FEATURES
                        ]
                    )[:, 1]
                )

                confidence = np.maximum(
                    probabilities,
                    1 - probabilities,
                )

                difficult_data = (
                    filter_difficult_states(
                        original_data,
                        scenario_name,
                        scenario,
                        confidence,
                    )
                )

                if len(difficult_data) == 0:

                    print(
                        f"{scenario_name} / "
                        f"{split_name}: "
                        f"No matching states."
                    )

                    continue

                print(
                    f"{scenario_name} / "
                    f"{split_name}: "
                    f"{len(difficult_data):,} states"
                )

                metrics = evaluate_scenario(
                    scenario_name,
                    split_name,
                    difficult_data,
                    models,
                    similarity_reference,
                )

                results.append(metrics)

    # ----------------------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_header(
        "ROBUSTNESS EVALUATION COMPLETE"
    )

    if len(results_df) > 0:

        display_columns = [
            "scenario",
            "split",
            "total_states",
            "baseline_accuracy",
            "final_accuracy",
            "coverage",
            "selective_risk",
            "error_detection_rate",
            "error_recovery_rate",
            "harmful_correction_rate",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in results_df.columns
        ]

        print(
            results_df[
                available_columns
            ].to_string(
                index=False
            )
        )

    print(
        f"\nResults saved to:\n{OUTPUT_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "No models were retrained and no "
        "thresholds were tuned during robustness testing."
    )


if __name__ == "__main__":
    main()