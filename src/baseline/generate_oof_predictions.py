"""
PRISM - Out-of-Fold Baseline Predictions

This script generates leakage-safe predictions for the training data.

For each fold:
    1. Hold out complete matches.
    2. Train the baseline model on the remaining matches.
    3. Predict the held-out matches.

Every training state therefore receives a prediction from a model that
was not trained on its match.

Output:
    results/baseline_oof_predictions.parquet
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

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

TARGET = "chasing_team_won"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "match_states.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "baseline_oof_predictions.parquet"
)


# ============================================================
# MODEL CREATION
# ============================================================

def create_baseline_pipeline():
    """
    Create the frozen Random Forest baseline configuration.

    This should match the configuration selected during
    baseline model training.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                "passthrough",
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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PRISM - OUT-OF-FOLD BASELINE PREDICTIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading match-state dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    train_df = (
        df[df["split"] == "train"]
        .copy()
        .reset_index(drop=True)
    )

    print(f"Training states : {len(train_df):,}")
    print(
        f"Training matches: "
        f"{train_df['match_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    feature_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = train_df[feature_columns]
    y = train_df[TARGET].astype(int)

    groups = train_df["match_id"]

    # --------------------------------------------------------
    # Group K-Fold
    # --------------------------------------------------------

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    oof_probability = np.full(
        shape=len(train_df),
        fill_value=np.nan,
        dtype=float,
    )

    oof_class = np.full(
        shape=len(train_df),
        fill_value=-1,
        dtype=int,
    )

    print("\nGenerating OOF predictions...")
    print(
        f"Number of folds: {N_SPLITS}"
    )

    total_start_time = time.perf_counter()

    # --------------------------------------------------------
    # Train and predict each fold
    # --------------------------------------------------------

    for fold, (
        train_index,
        validation_index,
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        print("\n" + "-" * 70)
        print(
            f"FOLD {fold}/{N_SPLITS}"
        )
        print("-" * 70)

        X_fold_train = (
            X.iloc[train_index]
        )

        y_fold_train = (
            y.iloc[train_index]
        )

        X_fold_validation = (
            X.iloc[validation_index]
        )

        train_matches = (
            groups.iloc[train_index]
            .nunique()
        )

        validation_matches = (
            groups.iloc[validation_index]
            .nunique()
        )

        print(
            f"Training states : "
            f"{len(train_index):,}"
        )

        print(
            f"Held-out states : "
            f"{len(validation_index):,}"
        )

        print(
            f"Training matches: "
            f"{train_matches}"
        )

        print(
            f"Held-out matches: "
            f"{validation_matches}"
        )

        # Check match overlap
        train_match_ids = set(
            groups.iloc[train_index]
        )

        validation_match_ids = set(
            groups.iloc[validation_index]
        )

        overlap = (
            train_match_ids
            & validation_match_ids
        )

        if overlap:
            raise RuntimeError(
                f"Match leakage detected in "
                f"fold {fold}: "
                f"{len(overlap)} overlapping matches."
            )

        # Create model
        pipeline = (
            create_baseline_pipeline()
        )

        # Train
        print("\nTraining model...")

        fold_start_time = (
            time.perf_counter()
        )

        pipeline.fit(
            X_fold_train,
            y_fold_train,
        )

        # Predict
        probability = (
            pipeline.predict_proba(
                X_fold_validation
            )[:, 1]
        )

        predicted_class = (
            (probability >= 0.5)
            .astype(int)
        )

        # Store predictions
        oof_probability[
            validation_index
        ] = probability

        oof_class[
            validation_index
        ] = predicted_class

        # Fold evaluation
        fold_accuracy = (
            predicted_class
            == y.iloc[validation_index]
        ).mean()

        fold_auc = roc_auc_score(
            y.iloc[validation_index],
            probability,
        )

        fold_time = (
            time.perf_counter()
            - fold_start_time
        )

        print(
            f"\nFold accuracy: "
            f"{fold_accuracy:.4f}"
        )

        print(
            f"Fold ROC-AUC : "
            f"{fold_auc:.4f}"
        )

        print(
            f"Fold time    : "
            f"{fold_time:.2f} seconds"
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if np.isnan(
        oof_probability
    ).any():

        missing = np.isnan(
            oof_probability
        ).sum()

        raise RuntimeError(
            f"{missing} states did not receive "
            "an OOF prediction."
        )

    if (oof_class == -1).any():

        raise RuntimeError(
            "Some states did not receive "
            "an OOF class prediction."
        )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output_columns = [
        "match_id",
        "date",
        "season",
        "over_completed",
        TARGET,
    ]

    output_df = (
        train_df[output_columns]
        .copy()
    )

    output_df[
        "predicted_class"
    ] = oof_class

    output_df[
        "predicted_probability"
    ] = oof_probability

    output_df[
        "model"
    ] = "random_forest_oof"

    output_df[
        "prediction_correct"
    ] = (
        output_df[
            "predicted_class"
        ]
        == output_df[TARGET]
    ).astype(int)

    output_df[
        "prediction_error"
    ] = (
        1
        - output_df[
            "prediction_correct"
        ]
    )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    overall_accuracy = (
        output_df[
            "prediction_correct"
        ].mean()
    )

    overall_auc = roc_auc_score(
        output_df[TARGET],
        output_df[
            "predicted_probability"
        ],
    )

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    print("\n" + "=" * 70)
    print("OOF PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Total predictions : "
        f"{len(output_df):,}"
    )

    print(
        f"Total matches     : "
        f"{output_df['match_id'].nunique():,}"
    )

    print(
        f"Overall accuracy  : "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Overall ROC-AUC   : "
        f"{overall_auc:.4f}"
    )

    print(
        f"Total runtime     : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Prediction errors : "
        f"{output_df['prediction_error'].sum():,}"
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\nOOF predictions saved to:")
    print(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("OOF PREDICTION GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()