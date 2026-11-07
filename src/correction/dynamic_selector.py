"""
PRISM - DYNAMIC ALTERNATIVE MODEL SELECTOR

Member 8: Using Alternative Models

This module compares the frozen baseline models on the validation set
and generates evidence that can later be used by the PRISM verifier.

The module does NOT automatically replace the primary baseline prediction.
Instead, it records:
    - Predictions from all models
    - Model confidences
    - Model disagreement
    - Alternative consensus
    - Evidence supporting an alternative prediction

IMPORTANT:
    The 2025 and 2026 test datasets are not used for tuning or analysis.
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


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "match_states.parquet"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
HGB_MODEL_PATH = MODEL_DIR / "hist_gradient_boosting.joblib"


# ======================================================================
# CONFIGURATION
# ======================================================================

PRIMARY_MODEL = "random_forest"

NUMERICAL_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "venue",
    "batting_team",
    "bowling_team",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN = "chasing_team_won"


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_model(path, name):
    """Load a saved model and verify that it exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required model not found:\n{path}\n"
            f"Please run train_baselines.py first."
        )

    print(f"Loading {name}...")
    return joblib.load(path)


def get_predictions(model, X):
    """
    Generate class predictions and probabilities.

    Returns:
        predictions: predicted class labels
        probabilities: probability of chasing team winning
        confidence: confidence in the predicted class
    """
    predictions = model.predict(X)

    probabilities_all = model.predict_proba(X)

    # Probability corresponding to class 1 (chasing team wins).
    class_indices = list(model.classes_)
    positive_index = class_indices.index(1)

    probabilities = probabilities_all[:, positive_index]

    # Confidence is the probability assigned to the predicted class.
    confidence = np.maximum(
        probabilities,
        1.0 - probabilities
    )

    return predictions, probabilities, confidence


def calculate_metrics(y_true, predictions, probabilities):
    """Calculate standard classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities
        )
    except ValueError:
        metrics["roc_auc"] = np.nan

    return metrics


# ======================================================================
# MAIN
# ======================================================================

def main():

    print_header(
        "PRISM - DYNAMIC ALTERNATIVE MODEL SELECTOR"
    )

    # ------------------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # ------------------------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    print("\nLoading match-state dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print(f"Total states : {len(df):,}")
    print(
        f"Total matches: "
        f"{df['match_id'].nunique():,}"
    )

    # ------------------------------------------------------------------
    # SELECT VALIDATION DATA ONLY
    # ------------------------------------------------------------------

    validation_df = df[
        df["split"] == "validation"
    ].copy()

    test_2025_count = len(
        df[df["split"] == "test_2025"]
    )

    test_2026_count = len(
        df[df["split"] == "test_2026"]
    )

    print_header("DATA SPLITS")

    print(
        f"Validation states : "
        f"{len(validation_df):,}"
    )

    print(
        f"Validation matches: "
        f"{validation_df['match_id'].nunique():,}"
    )

    print()
    print(
        f"Reserved 2025 test states: "
        f"{test_2025_count:,}"
    )

    print(
        f"Reserved 2026 test states: "
        f"{test_2026_count:,}"
    )

    if len(validation_df) == 0:
        raise RuntimeError(
            "No validation data found."
        )

    # ------------------------------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------------------------------

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in validation_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required feature columns:\n"
            + "\n".join(missing_features)
        )

    X_validation = validation_df[
        FEATURE_COLUMNS
    ].copy()

    y_validation = validation_df[
        TARGET_COLUMN
    ].astype(int).copy()

    # ------------------------------------------------------------------
    # LOAD FROZEN MODELS
    # ------------------------------------------------------------------

    print_header("LOADING FROZEN BASELINE MODELS")

    logistic_model = load_model(
        LOGISTIC_MODEL_PATH,
        "Logistic Regression"
    )

    random_forest_model = load_model(
        RANDOM_FOREST_MODEL_PATH,
        "Random Forest"
    )

    hgb_model = load_model(
        HGB_MODEL_PATH,
        "HistGradientBoosting"
    )

    models = {
        "logistic_regression": logistic_model,
        "random_forest": random_forest_model,
        "hist_gradient_boosting": hgb_model,
    }

    # ------------------------------------------------------------------
    # GENERATE PREDICTIONS
    # ------------------------------------------------------------------

    print_header("GENERATING VALIDATION PREDICTIONS")

    results = validation_df[
        [
            "match_id",
            "date",
            "season",
            "over_completed",
            TARGET_COLUMN,
        ]
    ].copy()

    model_metrics = []
    prediction_data = {}

    total_start_time = time.perf_counter()

    for model_name, model in models.items():

        print(f"\nRunning {model_name}...")

        start_time = time.perf_counter()

        predictions, probabilities, confidence = (
            get_predictions(
                model,
                X_validation
            )
        )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        metrics = calculate_metrics(
            y_validation,
            predictions,
            probabilities
        )

        metrics["model"] = model_name
        metrics["inference_seconds"] = elapsed_time

        model_metrics.append(metrics)

        prediction_data[model_name] = {
            "predictions": predictions,
            "probabilities": probabilities,
            "confidence": confidence,
        }

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"ROC-AUC  : {metrics['roc_auc']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1_score']:.4f}"
        )

        print(
            f"Inference: {elapsed_time:.4f} seconds"
        )

    total_runtime = (
        time.perf_counter() - total_start_time
    )

    # ------------------------------------------------------------------
    # STORE MODEL PREDICTIONS
    # ------------------------------------------------------------------

    for model_name, data in prediction_data.items():

        results[f"{model_name}_prediction"] = (
            data["predictions"]
        )

        results[f"{model_name}_probability"] = (
            data["probabilities"]
        )

        results[f"{model_name}_confidence"] = (
            data["confidence"]
        )

    # ------------------------------------------------------------------
    # PRIMARY BASELINE INFORMATION
    # ------------------------------------------------------------------

    primary_prediction = prediction_data[
        PRIMARY_MODEL
    ]["predictions"]

    primary_probability = prediction_data[
        PRIMARY_MODEL
    ]["probabilities"]

    primary_confidence = prediction_data[
        PRIMARY_MODEL
    ]["confidence"]

    results["baseline_prediction"] = primary_prediction
    results["baseline_probability"] = primary_probability
    results["baseline_confidence"] = primary_confidence

    results["baseline_correct"] = (
        results["baseline_prediction"]
        == results[TARGET_COLUMN]
    ).astype(int)

    results["baseline_error"] = (
        1 - results["baseline_correct"]
    )

    # ------------------------------------------------------------------
    # MODEL AGREEMENT AND DISAGREEMENT
    # ------------------------------------------------------------------

    print_header("MODEL DISAGREEMENT ANALYSIS")

    prediction_columns = [
        f"{name}_prediction"
        for name in models.keys()
    ]

    prediction_matrix = results[
        prediction_columns
    ].to_numpy()

    # Number of models predicting each possible outcome.
    votes_for_zero = (
        prediction_matrix == 0
    ).sum(axis=1)

    votes_for_one = (
        prediction_matrix == 1
    ).sum(axis=1)

    results["votes_for_0"] = votes_for_zero
    results["votes_for_1"] = votes_for_one

    results["max_model_agreement"] = np.maximum(
        votes_for_zero,
        votes_for_one
    )

    # All models agree when all 3 make the same prediction.
    results["all_models_agree"] = (
        results["max_model_agreement"] == 3
    ).astype(int)

    results["model_disagreement"] = (
        1 - results["all_models_agree"]
    )

    disagreement_rate = (
        results["model_disagreement"].mean()
    )

    print(
        f"Total validation states : "
        f"{len(results):,}"
    )

    print(
        f"Full model agreement    : "
        f"{results['all_models_agree'].sum():,} "
        f"({results['all_models_agree'].mean():.2%})"
    )

    print(
        f"Model disagreement      : "
        f"{results['model_disagreement'].sum():,} "
        f"({disagreement_rate:.2%})"
    )

    # ------------------------------------------------------------------
    # DISAGREEMENT VS BASELINE ERROR
    # ------------------------------------------------------------------

    disagreement_table = (
        results.groupby("model_disagreement")
        ["baseline_error"]
        .agg(
            predictions="count",
            baseline_errors="sum",
            baseline_error_rate="mean"
        )
        .reset_index()
    )

    disagreement_table["group"] = np.where(
        disagreement_table["model_disagreement"] == 1,
        "Models disagree",
        "Models agree"
    )

    print("\nBaseline error rate by model agreement:")

    print(
        disagreement_table[
            [
                "group",
                "predictions",
                "baseline_errors",
                "baseline_error_rate",
            ]
        ].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # BUILD ALTERNATIVE MODEL EVIDENCE
    # ------------------------------------------------------------------

    print_header(
        "ALTERNATIVE PREDICTION EVIDENCE"
    )

    alternative_models = [
        name
        for name in models.keys()
        if name != PRIMARY_MODEL
    ]

    # Count how many alternative models disagree with baseline.
    alternative_disagreement_count = np.zeros(
        len(results),
        dtype=int
    )

    alternative_predictions = []

    alternative_confidences = []

    for index in range(len(results)):

        baseline_pred = int(
            primary_prediction[index]
        )

        alternatives = []

        confidences = []

        for model_name in alternative_models:

            alt_prediction = int(
                prediction_data[model_name]
                ["predictions"][index]
            )

            alt_confidence = float(
                prediction_data[model_name]
                ["confidence"][index]
            )

            if alt_prediction != baseline_pred:
                alternatives.append(
                    alt_prediction
                )
                confidences.append(
                    alt_confidence
                )

        alternative_disagreement_count[index] = (
            len(alternatives)
        )

        # A suggested alternative is only produced when
        # BOTH alternative models disagree with the baseline
        # AND agree with each other.
        if len(alternatives) == 2:

            if alternatives[0] == alternatives[1]:

                alternative_predictions.append(
                    alternatives[0]
                )

                alternative_confidences.append(
                    float(np.mean(confidences))
                )

            else:

                alternative_predictions.append(
                    np.nan
                )

                alternative_confidences.append(
                    np.nan
                )

        else:

            alternative_predictions.append(
                np.nan
            )

            alternative_confidences.append(
                np.nan
            )

    results[
        "alternative_models_disagree_with_baseline"
    ] = alternative_disagreement_count

    results["alternative_prediction"] = (
        alternative_predictions
    )

    results["alternative_confidence"] = (
        alternative_confidences
    )

    results["alternative_consensus"] = (
        results["alternative_prediction"]
        .notna()
        .astype(int)
    )

    # Alternative evidence exists only when both secondary
    # models support the same alternative outcome.
    consensus_count = (
        results["alternative_consensus"].sum()
    )

    print(
        f"States with alternative consensus: "
        f"{consensus_count:,} "
        f"({consensus_count / len(results):.2%})"
    )

    # ------------------------------------------------------------------
    # ANALYZE ALTERNATIVE CORRECTION POTENTIAL
    # ------------------------------------------------------------------

    consensus_states = results[
        results["alternative_consensus"] == 1
    ].copy()

    if len(consensus_states) > 0:

        consensus_states[
            "alternative_correct"
        ] = (
            consensus_states[
                "alternative_prediction"
            ].astype(int)
            == consensus_states[TARGET_COLUMN]
        ).astype(int)

        baseline_wrong_alt_correct = (
            (
                consensus_states[
                    "baseline_correct"
                ] == 0
            )
            &
            (
                consensus_states[
                    "alternative_correct"
                ] == 1
            )
        ).sum()

        baseline_correct_alt_wrong = (
            (
                consensus_states[
                    "baseline_correct"
                ] == 1
            )
            &
            (
                consensus_states[
                    "alternative_correct"
                ] == 0
            )
        ).sum()

        print(
            "\nAlternative consensus analysis:"
        )

        print(
            f"Baseline wrong -> "
            f"alternative correct: "
            f"{baseline_wrong_alt_correct:,}"
        )

        print(
            f"Baseline correct -> "
            f"alternative wrong: "
            f"{baseline_correct_alt_wrong:,}"
        )

    else:

        baseline_wrong_alt_correct = 0
        baseline_correct_alt_wrong = 0

        print(
            "\nNo states produced a full "
            "alternative-model consensus."
        )

    # ------------------------------------------------------------------
    # MODEL METRICS TABLE
    # ------------------------------------------------------------------

    metrics_df = pd.DataFrame(
        model_metrics
    )

    metrics_df = metrics_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "inference_seconds",
        ]
    ].sort_values(
        "roc_auc",
        ascending=False
    )

    print_header(
        "MODEL PERFORMANCE COMPARISON"
    )

    print(
        metrics_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------

    disagreement_path = (
        RESULTS_DIR
        / "model_disagreement.csv"
    )

    evidence_path = (
        RESULTS_DIR
        / "dynamic_selector_predictions.parquet"
    )

    metrics_path = (
        RESULTS_DIR
        / "alternative_model_metrics.csv"
    )

    disagreement_table.to_csv(
        disagreement_path,
        index=False
    )

    results.to_parquet(
        evidence_path,
        index=False
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ------------------------------------------------------------------
    # CREATE REPORT
    # ------------------------------------------------------------------

    report_path = (
        DOCS_DIR
        / "selection_policy_report.md"
    )

    baseline_accuracy = accuracy_score(
        y_validation,
        primary_prediction
    )

    report = f"""# PRISM Alternative Model Selection Report

## Purpose

This experiment evaluates whether alternative baseline models can provide
useful evidence when the primary PRISM prediction may be unreliable.

The primary model is the frozen Random Forest baseline. Logistic Regression
and HistGradientBoosting are treated as alternative evidence sources.

## Dataset

- Validation states: {len(results):,}
- Validation matches: {results['match_id'].nunique():,}
- Reserved 2025 test states: {test_2025_count:,}
- Reserved 2026 test states: {test_2026_count:,}

The 2025 and 2026 test datasets were not used for tuning or policy selection.

## Model Agreement

- Full agreement: {results['all_models_agree'].sum():,} ({results['all_models_agree'].mean():.2%})
- Disagreement: {results['model_disagreement'].sum():,} ({results['model_disagreement'].mean():.2%})

## Primary Baseline

- Model: {PRIMARY_MODEL}
- Validation accuracy: {baseline_accuracy:.4f}

## Alternative Consensus

An alternative prediction is proposed only when both alternative models
disagree with the Random Forest baseline and agree with each other.

- Alternative consensus states: {consensus_count:,}
- Baseline wrong and alternative correct: {baseline_wrong_alt_correct:,}
- Baseline correct and alternative wrong: {baseline_correct_alt_wrong:,}

## Selection Policy

The dynamic selector does not automatically replace the primary prediction.

Instead, it provides evidence to the PRISM verifier, including:

1. Predictions from all baseline models.
2. Model confidence values.
3. Whether all models agree.
4. The number of models supporting each outcome.
5. An alternative prediction when both secondary models support the same outcome.

The final ACCEPT, CORRECT, or ABSTAIN decision will be made by the
verification module using this evidence together with reliability signals
and historical similarity evidence.

## Reproducibility

- Primary model: Random Forest
- Alternative models: Logistic Regression and HistGradientBoosting
- Validation data used for analysis only
- 2025 and 2026 test datasets remain untouched
- Total prediction runtime: {total_runtime:.4f} seconds
"""

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_header(
        "DYNAMIC SELECTOR COMPLETE"
    )

    print(
        f"\nPrimary baseline model: "
        f"{PRIMARY_MODEL}"
    )

    print(
        f"Alternative consensus states: "
        f"{consensus_count:,}"
    )

    print(
        f"\nFiles saved:"
    )

    print(f"  {disagreement_path}")
    print(f"  {evidence_path}")
    print(f"  {metrics_path}")
    print(f"  {report_path}")

    print(
        "\nIMPORTANT:"
    )

    print(
        "Alternative models currently provide evidence only."
    )

    print(
        "No automatic correction policy has been applied."
    )

    print(
        "The 2025 and 2026 test datasets remain untouched."
    )


if __name__ == "__main__":
    main()