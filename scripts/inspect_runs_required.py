import pandas as pd

DATA_PATH = "../data/match_states.parquet"

df = pd.read_parquet(DATA_PATH)

# Expected relationship:
# runs_required = target - current_score
#
# Depending on your dataset definition, runs_required may be clipped
# at zero once the target has been reached.

df["expected_runs_required"] = (
    df["target"] - df["current_score"]
).clip(lower=0)

errors = df[
    df["runs_required"] != df["expected_runs_required"]
].copy()

print("=" * 70)
print("PRISM - RUNS REQUIRED INSPECTION")
print("=" * 70)

print(f"\nTotal states          : {len(df)}")
print(f"Inconsistent states   : {len(errors)}")

if len(errors) > 0:
    print("\nSample inconsistent rows:\n")

    columns = [
        "match_id",
        "season",
        "over_completed",
        "current_score",
        "target",
        "runs_required",
        "expected_runs_required",
        "balls_remaining",
        "chasing_team_won"
    ]

    print(errors[columns].head(30).to_string(index=False))

    print("\nDifference distribution:")
    errors["difference"] = (
        errors["runs_required"]
        - errors["expected_runs_required"]
    )

    print(errors["difference"].value_counts().sort_index())

    print("\nAffected matches:")
    print(errors["match_id"].nunique())

print("\n" + "=" * 70)