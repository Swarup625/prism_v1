"""
PRISM - Match State Dataset Audit

This script validates the IPL match-state dataset before it is used
for baseline model training.

Checks performed:
1. Missing values
2. Split consistency
3. Feature validity
4. Mathematical consistency
5. Match progression
6. Target distribution
7. States per match
8. Sample states
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "match_states.parquet"


# ============================================================
# HELPER FUNCTION
# ============================================================

def print_check(name, error_count):
    """Print PASS or FAIL for a validation check."""
    if error_count == 0:
        print(f"PASS: {name} = 0 errors")
    else:
        print(f"FAIL : {name} = {error_count} errors")


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 70)
    print("PRISM - MATCH STATE DATASET AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print(f"\nTotal states  : {len(df)}")
    print(f"Total matches : {df['match_id'].nunique()}")

    # ========================================================
    # 1. MISSING VALUES
    # ========================================================

    print("\n1. MISSING VALUES")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print("Missing values detected:")
        print(missing.to_string())

    # ========================================================
    # 2. SPLIT CONSISTENCY
    # ========================================================

    print("\n2. SPLIT CONSISTENCY")

    match_split_counts = df.groupby("match_id")["split"].nunique()

    inconsistent_matches = (match_split_counts > 1).sum()

    if inconsistent_matches == 0:
        print("PASS: Every match belongs to exactly one split.")
    else:
        print(
            f"FAIL: {inconsistent_matches} matches appear "
            "in multiple splits."
        )

    # ========================================================
    # 3. FEATURE VALIDITY
    # ========================================================

    print("\n3. FEATURE VALIDITY")

    negative_score = (df["current_score"] < 0).sum()
    print_check("Negative current score", negative_score)

    negative_wickets = (df["wickets_lost"] < 0).sum()
    print_check("Negative wickets", negative_wickets)

    too_many_wickets = (df["wickets_lost"] > 10).sum()
    print_check("More than 10 wickets", too_many_wickets)

    negative_runs_required = (df["runs_required"] < 0).sum()
    print_check("Negative runs required", negative_runs_required)

    negative_balls = (df["balls_remaining"] < 0).sum()
    print_check("Negative balls remaining", negative_balls)

    invalid_target = (df["target"] <= 0).sum()
    print_check("Invalid target", invalid_target)

    invalid_over = (
        (df["over_completed"] < 1)
        | (df["over_completed"] > 20)
    ).sum()
    print_check("Invalid over", invalid_over)

    # ========================================================
    # 4. MATHEMATICAL CONSISTENCY
    # ========================================================

    print("\n4. MATHEMATICAL CONSISTENCY")

    # Runs required:
    #
    # The dataset defines:
    # runs_required = max(0, target - current_score)
    #
    expected_runs_required = (
        df["target"] - df["current_score"]
    ).clip(lower=0)

    runs_required_errors = (
        df["runs_required"]
        != expected_runs_required
    ).sum()

    print_check(
        "Runs required consistency",
        runs_required_errors
    )

    # Balls remaining:
    #
    # Prediction states are created after completed overs.
    # A T20 innings contains 120 legal balls.
    #
    expected_balls_remaining = (
        120 - df["over_completed"] * 6
    ).clip(lower=0)

    balls_remaining_errors = (
        df["balls_remaining"]
        != expected_balls_remaining
    ).sum()

    print_check(
        "Balls remaining consistency",
        balls_remaining_errors
    )

    # ========================================================
    # 5. MATCH PROGRESSION
    # ========================================================

    print("\n5. MATCH PROGRESSION")

    decreasing_score_matches = 0
    decreasing_wicket_matches = 0
    duplicate_over_matches = 0

    for match_id, group in df.groupby("match_id"):

        group = group.sort_values("over_completed")

        # Score should never decrease.
        if (group["current_score"].diff().dropna() < 0).any():
            decreasing_score_matches += 1

        # Wickets lost should never decrease.
        if (group["wickets_lost"].diff().dropna() < 0).any():
            decreasing_wicket_matches += 1

        # Only one prediction state per completed over.
        if group["over_completed"].duplicated().any():
            duplicate_over_matches += 1

    print_check(
        "Matches with decreasing score",
        decreasing_score_matches
    )

    print_check(
        "Matches with decreasing wickets",
        decreasing_wicket_matches
    )

    print_check(
        "Matches with duplicate overs",
        duplicate_over_matches
    )

    # ========================================================
    # 6. TARGET DISTRIBUTION BY SPLIT
    # ========================================================

    print("\n6. TARGET DISTRIBUTION BY SPLIT")

    target_distribution = (
        df.groupby("split")["chasing_team_won"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(3)
    )

    print(target_distribution)

    # ========================================================
    # 7. STATES PER MATCH
    # ========================================================

    print("\n7. STATES PER MATCH")

    states_per_match = (
        df.groupby("match_id")
        .size()
    )

    print(states_per_match.describe())

    matches_with_fewer_states = (
        states_per_match < 19
    ).sum()

    print(
        f"\nMatches with fewer than 19 states: "
        f"{matches_with_fewer_states}"
    )

    # ========================================================
    # 8. SAMPLE STATES
    # ========================================================

    print("\n8. SAMPLE STATES")

    sample_columns = [
        "match_id",
        "season",
        "over_completed",
        "current_score",
        "wickets_lost",
        "target",
        "runs_required",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "chasing_team_won",
    ]

    available_columns = [
        column
        for column in sample_columns
        if column in df.columns
    ]

    print(
        df[available_columns]
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()