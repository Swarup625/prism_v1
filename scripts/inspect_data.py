from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.csv"
DELIVERIES_PATH = PROJECT_ROOT / "data" / "processed" / "deliveries.parquet"


def main():
    print("=" * 65)
    print("PRISM - DATA INSPECTION")
    print("=" * 65)

    matches = pd.read_csv(MATCHES_PATH)
    deliveries = pd.read_parquet(DELIVERIES_PATH)

    print("\n1. DATASET SHAPES")
    print(f"Matches:    {matches.shape}")
    print(f"Deliveries: {deliveries.shape}")

    print("\n2. MATCHES COLUMNS")
    print(matches.columns.tolist())

    print("\n3. DELIVERIES COLUMNS")
    print(deliveries.columns.tolist())

    print("\n4. AVAILABLE SEASONS")
    print(
        matches.groupby("season")
        .size()
        .to_string()
    )

    print("\n5. MATCH OUTCOMES")
    print(
        matches["result_type"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n6. MISSING VALUES - MATCHES")
    print(matches.isnull().sum().to_string())

    print("\n7. INNINGS PER MATCH")
    innings_per_match = (
        deliveries.groupby("match_id")["innings"]
        .nunique()
    )

    print(innings_per_match.value_counts().sort_index().to_string())

    print("\n8. SAMPLE MATCHES FROM PRISM PERIOD")

    prism_seasons = [
        "2020/21", "2021", "2022",
        "2023", "2024", "2025", "2026"
    ]

    prism_matches = matches[
        matches["season"].astype(str).isin(prism_seasons)
    ]

    print(f"\nMatches from 2020/21–2026: {len(prism_matches)}")

    print(
        prism_matches[
            ["match_id", "date", "season", "team1", "team2", "winner"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()