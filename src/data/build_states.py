"""
PRISM - IPL Match State Dataset Builder

Creates machine learning prediction states from cleaned IPL
ball-by-ball data.

Prediction task:
    After each completed over of the second innings, predict
    whether the chasing team will eventually win the match.

Dataset split:
    Training   : 2020/21, 2021, 2022, 2023
    Validation : 2024
    Test       : 2025, 2026
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
DELIVERIES_PATH = PROJECT_ROOT / "data" / "processed" / "deliveries.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "match_states.parquet"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "split_manifest.csv"


# ---------------------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------------------

SELECTED_SEASONS = [
    "2020/21",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
]

TRAIN_SEASONS = ["2020/21", "2021", "2022", "2023"]
VALIDATION_SEASONS = ["2024"]
TEST_SEASONS = ["2025", "2026"]

MIN_OVER = 1
MAX_OVER = 19


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def assign_split(season):
    """Assign a dataset split based on the IPL season."""

    season = str(season)

    if season in TRAIN_SEASONS:
        return "train"

    if season in VALIDATION_SEASONS:
        return "validation"

    if season == "2025":
        return "test_2025"

    if season == "2026":
        return "test_2026"

    return None


def get_over_summary(innings_data):
    """
    Create cumulative over-level information for one innings.

    Returns one row per completed over.
    """

    over_rows = []

    cumulative_score = 0
    cumulative_wickets = 0

    for over_number in sorted(innings_data["over"].unique()):

        over_data = innings_data[
            innings_data["over"] == over_number
        ]

        runs = over_data["runs_total"].sum()
        wickets = over_data["is_wicket"].sum()

        cumulative_score += runs
        cumulative_wickets += wickets

        over_rows.append(
            {
                "over": int(over_number),
                "runs_this_over": runs,
                "wickets_this_over": wickets,
                "current_score": cumulative_score,
                "wickets_lost": cumulative_wickets,
            }
        )

    return pd.DataFrame(over_rows)


# ---------------------------------------------------------------------
# MAIN DATASET BUILDER
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PRISM - IPL MATCH STATE DATASET BUILDER")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    print("\nLoading processed data...")

    matches = pd.read_csv(MATCHES_PATH)
    deliveries = pd.read_parquet(DELIVERIES_PATH)

    # Normalize identifiers so both datasets use the same format.
    #    match_id is an identifier, so we store it as a string.
    matches["match_id"] = matches["match_id"].astype(str)
    deliveries["match_id"] = deliveries["match_id"].astype(str)

    # Normalize season representation.
    matches["season"] = matches["season"].astype(str)

    print(f"Matches loaded    : {len(matches)}")
    print(f"Deliveries loaded : {len(deliveries)}")

    # -------------------------------------------------------------
    # Select experiment seasons
    # -------------------------------------------------------------

    matches = matches[
        matches["season"].isin(SELECTED_SEASONS)
    ].copy()

    print(f"\nMatches in selected seasons: {len(matches)}")

    # -------------------------------------------------------------
    # Keep only normal-result matches with a known winner
    # -------------------------------------------------------------

    matches = matches[
        (matches["result_type"] == "normal")
        & (matches["winner"].notna())
    ].copy()

    print(f"After normal-result filter: {len(matches)}")

    # -------------------------------------------------------------
    # Keep matches with exactly two innings
    # -------------------------------------------------------------

    innings_counts = (
        deliveries[
            deliveries["match_id"].isin(matches["match_id"])
        ]
        .groupby("match_id")["innings"]
        .nunique()
    )

    valid_match_ids = innings_counts[
        innings_counts == 2
    ].index

    matches = matches[
        matches["match_id"].isin(valid_match_ids)
    ].copy()

    print(f"After two-innings filter: {len(matches)}")

    # -------------------------------------------------------------
    # Build match states
    # -------------------------------------------------------------

    all_states = []
    excluded_matches = []

    print("\nBuilding over-by-over prediction states...")

    for match_index, (_, match) in enumerate(
        matches.iterrows(),
        start=1
    ):

        match_id = match["match_id"]

        match_deliveries = deliveries[
            deliveries["match_id"] == match_id
        ].copy()

        # Sort deliveries to preserve match order
        match_deliveries = match_deliveries.sort_values(
            ["innings", "over", "delivery_index"]
        )

        innings_numbers = sorted(
            match_deliveries["innings"].unique()
        )

        if len(innings_numbers) != 2:
            excluded_matches.append(
                (match_id, "not_exactly_two_innings")
            )
            continue

        first_innings_number = innings_numbers[0]
        second_innings_number = innings_numbers[1]

        first_innings = match_deliveries[
            match_deliveries["innings"] == first_innings_number
        ]

        second_innings = match_deliveries[
            match_deliveries["innings"] == second_innings_number
        ]

        if first_innings.empty or second_innings.empty:
            excluded_matches.append(
                (match_id, "empty_innings")
            )
            continue

        # ---------------------------------------------------------
        # Identify innings teams
        # ---------------------------------------------------------

        first_innings_team = first_innings[
            "batting_team"
        ].iloc[0]

        chasing_team = second_innings[
            "batting_team"
        ].iloc[0]

        # The bowling team in the second innings is the first
        # innings batting team for a normal two-team match.
        bowling_team = first_innings_team

        # Safety check: winner must be one of the match teams.
        if match["winner"] not in {
            match["team1"],
            match["team2"],
        }:
            excluded_matches.append(
                (match_id, "winner_not_match_team")
            )
            continue

        # ---------------------------------------------------------
        # Calculate target
        # ---------------------------------------------------------

        first_innings_total = first_innings[
            "runs_total"
        ].sum()

        target = int(first_innings_total + 1)

        # ---------------------------------------------------------
        # Create target label
        # ---------------------------------------------------------

        chasing_team_won = int(
            match["winner"] == chasing_team
        )

        # ---------------------------------------------------------
        # Build over summaries
        # ---------------------------------------------------------

        over_summary = get_over_summary(second_innings)

        if over_summary.empty:
            excluded_matches.append(
                (match_id, "no_over_data")
            )
            continue

        # ---------------------------------------------------------
        # Create states after completed overs 1–19
        # ---------------------------------------------------------

        for _, current in over_summary.iterrows():

            over_completed = int(current["over"]) + 1

            # Cricsheet uses zero-based over numbering.
            if not (
                MIN_OVER
                <= over_completed
                <= MAX_OVER
            ):
                continue

            current_score = int(current["current_score"])
            wickets_lost = int(current["wickets_lost"])

            runs_required = max(
                0,
                target - current_score
            )

            balls_completed = over_completed * 6
            balls_remaining = max(
                0,
                120 - balls_completed
            )

            current_run_rate = (
                current_score / over_completed
                if over_completed > 0
                else 0.0
            )

            required_run_rate = (
                (runs_required * 6 / balls_remaining)
                if balls_remaining > 0
                else 0.0
            )

            # -----------------------------------------------------
            # Recent-form features
            # -----------------------------------------------------

            recent_start = max(
                0,
                int(current["over"]) - 2
            )

            recent_overs = over_summary[
                (
                    over_summary["over"]
                    >= recent_start
                )
                &
                (
                    over_summary["over"]
                    <= current["over"]
                )
            ]

            runs_last_3_overs = int(
                recent_overs["runs_this_over"].sum()
            )

            wickets_last_3_overs = int(
                recent_overs["wickets_this_over"].sum()
            )

            # -----------------------------------------------------
            # Create state
            # -----------------------------------------------------

            state = {
                "match_id": match_id,
                "date": match["date"],
                "season": match["season"],
                "venue": match["venue"],

                "batting_team": chasing_team,
                "bowling_team": bowling_team,

                "over_completed": over_completed,
                "current_score": current_score,
                "wickets_lost": wickets_lost,

                "target": target,
                "runs_required": runs_required,
                "balls_remaining": balls_remaining,

                "current_run_rate": round(
                    current_run_rate, 4
                ),
                "required_run_rate": round(
                    required_run_rate, 4
                ),

                "runs_last_3_overs": (
                    runs_last_3_overs
                ),
                "wickets_last_3_overs": (
                    wickets_last_3_overs
                ),

                "chasing_team_won": chasing_team_won,
            }

            all_states.append(state)

        if match_index % 50 == 0:
            print(
                f"Processed {match_index}/{len(matches)} matches..."
            )

    # -------------------------------------------------------------
    # Create final dataframe
    # -------------------------------------------------------------

    states = pd.DataFrame(all_states)

    if states.empty:
        raise RuntimeError(
            "No match states were created. "
            "Check the input data and innings structure."
        )

    # -------------------------------------------------------------
    # Assign train / validation / test splits
    # -------------------------------------------------------------

    states["split"] = states["season"].apply(
        assign_split
    )

    states = states[
        states["split"].notna()
    ].copy()

    # -------------------------------------------------------------
    # Save dataset
    # -------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    states.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    # -------------------------------------------------------------
    # Create split manifest
    # -------------------------------------------------------------

    split_manifest = (
        states[
            [
                "match_id",
                "season",
                "split",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["season", "match_id"]
        )
    )

    split_manifest.to_csv(
        SPLIT_MANIFEST_PATH,
        index=False
    )

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nValid matches used : "
        f"{states['match_id'].nunique()}"
    )

    print(
        f"Prediction states  : {len(states)}"
    )

    print("\nStates by split:")

    print(
        states.groupby("split")
        .agg(
            matches=("match_id", "nunique"),
            states=("match_id", "size"),
        )
        .to_string()
    )

    print("\nTarget distribution:")

    print(
        states["chasing_team_won"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nDataset saved to:\n{OUTPUT_PATH}"
    )

    print(
        f"\nSplit manifest saved to:\n"
        f"{SPLIT_MANIFEST_PATH}"
    )

    if excluded_matches:
        print(
            f"\nExcluded matches: "
            f"{len(excluded_matches)}"
        )

        for reason in pd.Series(
            [reason for _, reason in excluded_matches]
        ).value_counts().items():

            print(
                f"  {reason[0]}: {reason[1]}"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()