"""
PRISM - IPL Cricsheet Data Parser

Reads IPL Cricsheet JSON files and creates:
    1. data/processed/matches.csv
    2. data/processed/deliveries.parquet

Each row in deliveries.parquet represents one delivery event.
"""

import json
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ipl_json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MATCHES_OUTPUT = PROCESSED_DIR / "matches.csv"
DELIVERIES_OUTPUT = PROCESSED_DIR / "deliveries.parquet"


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def get_match_id(json_path: Path) -> str:
    """
    Use the JSON filename as a stable match identifier.
    Example: 1234567.json -> 1234567
    """
    return json_path.stem


def get_result_type(outcome: dict) -> str:
    """
    Determine the general match result type.
    """
    if not outcome:
        return "unknown"

    if "winner" in outcome:
        return "normal"

    if outcome.get("result"):
        return str(outcome["result"])

    return "unknown"


def get_extras_type(extras: dict) -> str:
    """
    Convert the extras dictionary into a readable string.

    Example:
        {"wides": 1} -> "wides"
        {"noballs": 1, "byes": 2} -> "byes,noballs"
    """
    if not extras:
        return ""

    return ",".join(sorted(extras.keys()))


# ------------------------------------------------------------
# PARSING FUNCTIONS
# ------------------------------------------------------------

def parse_match(json_path: Path):
    """
    Parse one Cricsheet JSON file.

    Returns
    -------
    match_row : dict
        One row of match-level information.

    delivery_rows : list[dict]
        Ball-by-ball delivery records.
    """

    with open(json_path, "r", encoding="utf-8") as file:
        match = json.load(file)

    match_id = get_match_id(json_path)

    info = match.get("info", {})
    outcome = info.get("outcome", {})

    # -------------------------
    # MATCH METADATA
    # -------------------------

    dates = info.get("dates", [])
    match_date = dates[0] if dates else None

    teams = info.get("teams", [])

    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    match_row = {
        "match_id": match_id,
        "date": match_date,
        "season": info.get("season"),
        "team1": team1,
        "team2": team2,
        "venue": info.get("venue"),
        "winner": outcome.get("winner"),
        "result_type": get_result_type(outcome),
    }

    # -------------------------
    # DELIVERY DATA
    # -------------------------

    delivery_rows = []

    innings_list = match.get("innings", [])

    for innings_number, innings in enumerate(innings_list, start=1):

        batting_team = innings.get("team")

        overs = innings.get("overs", [])

        for over_data in overs:

            over_number = over_data.get("over")

            deliveries = over_data.get("deliveries", [])

            for delivery_index, delivery in enumerate(deliveries):

                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                wicket_count = len(wickets)

                row = {
                    # Match identification
                    "match_id": match_id,
                    "innings": innings_number,

                    # Delivery position
                    "over": over_number,
                    "delivery_index": delivery_index,
                    "actual_delivery": delivery.get("ball"),

                    # Teams and players
                    "batting_team": batting_team,
                    "batter": delivery.get("batter"),
                    "non_striker": delivery.get("non_striker"),
                    "bowler": delivery.get("bowler"),

                    # Runs
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),

                    # Extras
                    "extras_type": get_extras_type(extras),

                    # Wickets
                    "is_wicket": int(wicket_count > 0),
                    "wicket_count": wicket_count,
                }

                delivery_rows.append(row)

    return match_row, delivery_rows


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("PRISM - IPL Cricsheet Data Parser")
    print("=" * 60)

    # Create output directory if needed
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"\nRaw data directory not found:\n{RAW_DATA_DIR}\n\n"
            "Please extract the IPL JSON files into:\n"
            "data/raw/ipl_json/"
        )

    json_files = sorted(RAW_DATA_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in:\n{RAW_DATA_DIR}"
        )

    print(f"\nFound {len(json_files)} JSON files.")
    print("Parsing matches...\n")

    match_rows = []
    all_delivery_rows = []

    successful_matches = 0
    failed_matches = []

    for index, json_path in enumerate(json_files, start=1):

        try:
            match_row, delivery_rows = parse_match(json_path)

            match_rows.append(match_row)
            all_delivery_rows.extend(delivery_rows)

            successful_matches += 1

        except Exception as error:

            failed_matches.append({
                "file": json_path.name,
                "error": str(error),
            })

            print(
                f"[WARNING] Failed to parse "
                f"{json_path.name}: {error}"
            )

        if index % 100 == 0 or index == len(json_files):
            print(
                f"Processed {index}/{len(json_files)} files "
                f"({successful_matches} successful)"
            )

    # --------------------------------------------------------
    # CREATE DATAFRAMES
    # --------------------------------------------------------

    matches_df = pd.DataFrame(match_rows)
    deliveries_df = pd.DataFrame(all_delivery_rows)

    # Sort for reproducibility
    if not matches_df.empty:
        matches_df = matches_df.sort_values(
            by=["date", "match_id"],
            na_position="last"
        )

    if not deliveries_df.empty:
        deliveries_df = deliveries_df.sort_values(
            by=[
                "match_id",
                "innings",
                "over",
                "delivery_index",
            ]
        )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    matches_df.to_csv(
        MATCHES_OUTPUT,
        index=False
    )

    deliveries_df.to_parquet(
        DELIVERIES_OUTPUT,
        index=False
    )

    # Save failed file information if necessary
    if failed_matches:

        failed_df = pd.DataFrame(failed_matches)

        failed_output = PROCESSED_DIR / "failed_matches.csv"

        failed_df.to_csv(
            failed_output,
            index=False
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PARSING COMPLETE")
    print("=" * 60)

    print(f"Total JSON files found : {len(json_files)}")
    print(f"Successfully parsed    : {successful_matches}")
    print(f"Failed                 : {len(failed_matches)}")

    print(f"\nMatches saved to:")
    print(f"  {MATCHES_OUTPUT}")

    print(f"\nDeliveries saved to:")
    print(f"  {DELIVERIES_OUTPUT}")

    print("\nDataset summary:")

    print(f"  Matches    : {len(matches_df):,}")
    print(f"  Deliveries : {len(deliveries_df):,}")

    if not matches_df.empty:

        print("\nSeasons found:")

        # Convert seasons to strings before sorting because the
        # Cricsheet dataset may contain both numeric and string seasons.
        season_counts = (
            matches_df["season"]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
        )

        for season, count in season_counts.items():
            print(f"  {season}: {count}")

    if failed_matches:
        print(
            f"\nWarning: {len(failed_matches)} files failed to parse."
        )


if __name__ == "__main__":
    main()