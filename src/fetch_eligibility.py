"""
Build the position architecture for the fantasy basketball pipeline.

This file deliberately separates NBA position information from ESPN
fantasy eligibility.

NBA source:
    data/raw/player_positions.csv

Output:
    data/raw/position_eligibility.csv

IMPORTANT:
    NBA broad positions are NOT treated as authoritative ESPN fantasy
    eligibility.

    In particular:
        G   does not automatically mean PG or SG
        F   does not automatically mean SF or PF
        G-F does not automatically establish PG/SG eligibility
        F-G does not automatically establish SF/PF eligibility

    ESPN_PRIMARY_POSITION and ESPN_ELIGIBLE_POSITIONS remain blank
    until authoritative ESPN fantasy data is available.
"""

import pandas as pd

from common import RAW_DIR, ensure_directories


INPUT_PATH = f"{RAW_DIR}/player_positions.csv"
OUTPUT_PATH = f"{RAW_DIR}/position_eligibility.csv"


def main():
    ensure_directories()

    print("Loading NBA position data...")
    df = pd.read_csv(INPUT_PATH)

    required = [
        "PERSON_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "POSITION",
        "POSITION_GROUP",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "player_positions.csv is missing columns: "
            + ", ".join(missing)
        )

    output = df[required].copy()

    output = output.rename(
        columns={
            "POSITION": "NBA_POSITION",
            "POSITION_GROUP": "NBA_POSITION_GROUP",
        }
    )

    # ------------------------------------------------------------
    # ESPN fields
    # ------------------------------------------------------------
    #
    # Do NOT infer PG/SG/SF/PF from broad NBA G/F labels.
    #
    # These fields remain blank until authoritative ESPN data
    # is available.
    #

    output["ESPN_PRIMARY_POSITION"] = pd.NA
    output["ESPN_ELIGIBLE_POSITIONS"] = pd.NA
    output["ESPN_SOURCE"] = "NOT_YET_RETRIEVED"
    output["ESPN_CONFIDENCE"] = "UNRESOLVED"

    # ------------------------------------------------------------
    # Safe NBA-derived fallback information
    # ------------------------------------------------------------
    #
    # This is NOT ESPN eligibility.
    #
    # It exists so the pipeline can retain useful positional
    # information without contaminating the ESPN fields.
    #

    def nba_derived_fallback(position):
        position = str(position).strip().upper()

        if position in {"C", "F-C", "C-F"}:
            return "C"

        return pd.NA

    output["NBA_DERIVED_PRIMARY_FALLBACK"] = output[
        "NBA_POSITION"
    ].apply(nba_derived_fallback)

    output["NBA_DERIVED_FALLBACK_CONFIDENCE"] = output[
        "NBA_DERIVED_PRIMARY_FALLBACK"
    ].apply(
        lambda value: (
            "SAFE"
            if pd.notna(value)
            else "NOT_DETERMINABLE"
        )
    )

    # One record per NBA player ID.
    output = output.drop_duplicates(
        subset=["PERSON_ID"],
        keep="first",
    )

    output = output.sort_values(
        ["PLAYER_NAME", "PERSON_ID"]
    ).reset_index(drop=True)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("POSITION ARCHITECTURE COMPLETE")
    print("=" * 60)
    print()
    print(f"Rows: {len(output)}")
    print(
        "ESPN primary positions populated: "
        f"{output['ESPN_PRIMARY_POSITION'].notna().sum()}"
    )
    print(
        "ESPN eligible-position sets populated: "
        f"{output['ESPN_ELIGIBLE_POSITIONS'].notna().sum()}"
    )
    print(
        "Safe NBA-derived C fallback: "
        f"{output['NBA_DERIVED_PRIMARY_FALLBACK'].notna().sum()}"
    )
    print()
    print("NBA position distribution:")
    print(
        output["NBA_POSITION"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("ESPN status:")
    print(
        output["ESPN_CONFIDENCE"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()
    print("Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()