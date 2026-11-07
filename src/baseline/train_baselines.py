"""
PRISM - Baseline Model Training

This script trains baseline machine-learning models for predicting
whether the chasing team will win an IPL match.

Dataset split:
    Training   : 2020/21, 2021, 2022, 2023
    Validation : 2024
    Test       : 2025 (reserved)
    Test       : 2026 (reserved)

The test seasons are NOT used for model selection or tuning.
"""

from pathlib import Path
import time
import platform

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "match_states.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# FEATURE CONFIGURATION
# ======================================================================

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
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }

    return metrics


def print_metrics(model_name, metrics, cm, inference_time):
    """
    Print model evaluation results.
    """

    print(f"\nModel: {model_name}")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"Log Loss  : {metrics['log_loss']:.4f}")
    print(f"Inference : {inference_time:.4f} seconds")

    print("\nConfusion Matrix:")
    print(cm)


def create_preprocessor():
    """
    Create preprocessing pipeline.

    sparse_output=False is important because
    HistGradientBoostingClassifier requires dense input.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ======================================================================
# MAIN PROGRAM
# ======================================================================

def main():

    print_header("PRISM - BASELINE MODEL TRAINING")

    # ------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------

    print("\nLoading match-state dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}\n\n"
            "Run build_states.py first."
        )

    df = pd.read_parquet(DATA_PATH)

    print(f"Total states : {len(df):,}")
    print(f"Total matches: {df['match_id'].nunique():,}")

    # ------------------------------------------------------------------
    # VERIFY REQUIRED COLUMNS
    # ------------------------------------------------------------------

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "split"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing:\n"
            f"{missing_columns}"
        )

    # ------------------------------------------------------------------
    # CREATE DATA SPLITS
    # ------------------------------------------------------------------

    train_df = df[df["split"] == "train"].copy()
    validation_df = df[df["split"] == "validation"].copy()
    test_2025_df = df[df["split"] == "test_2025"].copy()
    test_2026_df = df[df["split"] == "test_2026"].copy()

    print_header("DATA SPLITS")

    print(
        f"Training states   : {len(train_df):,} "
        f"({train_df['match_id'].nunique()} matches)"
    )

    print(
        f"Validation states : {len(validation_df):,} "
        f"({validation_df['match_id'].nunique()} matches)"
    )

    print(
        f"Reserved test states (2025): {len(test_2025_df):,}"
    )

    print(
        f"Reserved test states (2026): {len(test_2026_df):,}"
    )

    if len(train_df) == 0 or len(validation_df) == 0:
        raise RuntimeError(
            "Training or validation split is empty."
        )

    # ------------------------------------------------------------------
    # FEATURE CONFIGURATION
    # ------------------------------------------------------------------

    print_header("FEATURE CONFIGURATION")

    print("\nNumerical features:")
    for feature in NUMERICAL_FEATURES:
        print(f"  - {feature}")

    print("\nCategorical features:")
    for feature in CATEGORICAL_FEATURES:
        print(f"  - {feature}")

    # ------------------------------------------------------------------
    # PREPARE DATA
    # ------------------------------------------------------------------

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_validation = validation_df[FEATURE_COLUMNS].copy()
    y_validation = validation_df[TARGET_COLUMN].copy()

    # ------------------------------------------------------------------
    # DEFINE MODELS
    # ------------------------------------------------------------------

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
    }

    # ------------------------------------------------------------------
    # TRAIN AND EVALUATE MODELS
    # ------------------------------------------------------------------

    all_results = []
    trained_models = {}

    for model_name, model in models.items():

        print_header(
            f"TRAINING: {model_name.upper()}"
        )

        # Create a fresh preprocessor for every model.
        preprocessor = create_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        print("\nTraining model...")

        training_start = time.perf_counter()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        print(
            f"Training completed in "
            f"{training_time:.2f} seconds."
        )

        # --------------------------------------------------------------
        # VALIDATION PREDICTIONS
        # --------------------------------------------------------------

        inference_start = time.perf_counter()

        validation_predictions = pipeline.predict(
            X_validation
        )

        validation_probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        # --------------------------------------------------------------
        # METRICS
        # --------------------------------------------------------------

        metrics = calculate_metrics(
            y_validation,
            validation_predictions,
            validation_probabilities,
        )

        cm = confusion_matrix(
            y_validation,
            validation_predictions,
        )

        print_metrics(
            model_name,
            metrics,
            cm,
            inference_time,
        )

        # --------------------------------------------------------------
        # SAVE MODEL
        # --------------------------------------------------------------

        model_path = (
            MODELS_DIR /
            f"{model_name}.joblib"
        )

        joblib.dump(
            pipeline,
            model_path,
        )

        print("\nModel saved to:")
        print(model_path)

        # Store model
        trained_models[model_name] = pipeline

        # Store metrics
        result_row = {
            "model": model_name,
            **metrics,
            "training_time_seconds": training_time,
            "inference_time_seconds": inference_time,
        }

        all_results.append(result_row)

    # ------------------------------------------------------------------
    # SAVE VALIDATION METRICS
    # ------------------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    metrics_path = (
        RESULTS_DIR /
        "baseline_metrics.csv"
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # DISPLAY MODEL COMPARISON
    # ------------------------------------------------------------------

    print_header("VALIDATION MODEL COMPARISON")

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "log_loss",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------
    # SELECT BEST MODEL
    # ------------------------------------------------------------------

    best_model_name = results_df.iloc[0]["model"]

    print_header("BEST BASELINE MODEL")

    print(
        f"Selected model: {best_model_name}"
    )

    print(
        "\nSelection criterion: "
        "highest validation ROC-AUC."
    )

    # ------------------------------------------------------------------
    # SAVE VALIDATION PREDICTIONS
    # ------------------------------------------------------------------

    best_pipeline = trained_models[
        best_model_name
    ]

    validation_predictions = best_pipeline.predict(
        X_validation
    )

    validation_probabilities = (
        best_pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    prediction_output = validation_df[
        [
            "match_id",
            "date",
            "season",
            "over_completed",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_output[
        "predicted_class"
    ] = validation_predictions

    prediction_output[
        "predicted_probability"
    ] = validation_probabilities

    prediction_output[
        "model"
    ] = best_model_name

    predictions_path = (
        RESULTS_DIR /
        "baseline_validation_predictions.parquet"
    )

    prediction_output.to_parquet(
        predictions_path,
        index=False,
    )

    print("\nValidation predictions saved to:")
    print(predictions_path)

    # ------------------------------------------------------------------
    # SAVE EXPERIMENT INFORMATION
    # ------------------------------------------------------------------

    experiment_info = [
        "PRISM Baseline Model Experiment",
        "=" * 50,
        "",
        f"Dataset: {DATA_PATH}",
        f"Training states: {len(train_df)}",
        f"Validation states: {len(validation_df)}",
        "",
        "Training seasons: 2020/21–2023",
        "Validation season: 2024",
        "Test seasons: 2025 and 2026 (NOT USED HERE)",
        "",
        f"Selected baseline model: {best_model_name}",
        "Selection criterion: Highest validation ROC-AUC",
        "",
        "Numerical features:",
        *[f"- {feature}" for feature in NUMERICAL_FEATURES],
        "",
        "Categorical features:",
        *[f"- {feature}" for feature in CATEGORICAL_FEATURES],
        "",
        "Software information:",
        f"Python platform: {platform.platform()}",
        f"Pandas version: {pd.__version__}",
        f"NumPy version: {np.__version__}",
    ]

    report_path = (
        RESULTS_DIR /
        "baseline_experiment_report.txt"
    )

    report_path.write_text(
        "\n".join(experiment_info),
        encoding="utf-8",
    )

    print("\nExperiment report saved to:")
    print(report_path)

    # ------------------------------------------------------------------
    # IMPORTANT NOTE
    # ------------------------------------------------------------------

    print_header("BASELINE TRAINING COMPLETE")

    print(
        "\nIMPORTANT:"
    )

    print(
        "The 2025 and 2026 test sets have not been used "
        "for model selection or tuning."
    )

    print(
        "\nNext step: Review the validation results, freeze the "
        "baseline configuration, and then use the saved predictions "
        "to build PRISM's reliability and error-detection modules."
    )


if __name__ == "__main__":
    main()