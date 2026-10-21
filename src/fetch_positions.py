"""
Download NBA player position information.

Source:
    NBA API PlayerIndex

Output:
    data/raw/player_positions.csv

This captures NBA's broad position labels. It does NOT attempt
to infer ESPN fantasy eligibility.
"""

import pandas as pd
from nba_api.stats.endpoints import playerindex

from common import (
    RAW_DIR,
    SLEEP_SECONDS,
    TIMEOUT,
    ensure_directories,
    with_retries,
)

OUTPUT_PATH = f"{RAW_DIR}/player_positions.csv"


def fetch_positions(season):
    response = playerindex.PlayerIndex(
        season=season,
        timeout=TIMEOUT,
    )

    frames = response.get_data_frames()

    if not frames:
        raise RuntimeError(
            f"PlayerIndex returned no data for {season}."
        )

    df = frames[0].copy()

    if df.empty:
        raise RuntimeError(
            f"PlayerIndex returned zero rows for {season}."
        )

    return df


def clean_positions(df, season):
    required = [
        "PERSON_ID",
        "PLAYER_LAST_NAME",
        "PLAYER_FIRST_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "POSITION",
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{season} PlayerIndex is missing columns: "
            + ", ".join(missing)
        )

    output = df[required].copy()

    output["PLAYER_NAME"] = (
        output["PLAYER_FIRST_NAME"].fillna("").astype(str).str.strip()
        + " "
        + output["PLAYER_LAST_NAME"].fillna("").astype(str).str.strip()
    ).str.strip()

    output["POSITION"] = (
        output["POSITION"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    def position_group(position):
        position = position.upper()

        if "C" in position and "F" in position:
            return "F/C"

        if "C" in position:
            return "C"

        if "G" in position and "F" in position:
            return "G/F"

        if "G" in position:
            return "G"

        if "F" in position:
            return "F"

        return ""

    output["POSITION_GROUP"] = output["POSITION"].apply(
        position_group
    )

    output["SEASON"] = season
    output["SEASON_START_YEAR"] = int(season[:4])

    return output


def main():
    ensure_directories()

    # The 2026-27 season has not begun yet, so use the most recent
    # completed season for the player-position reference table.
    season = "2025-26"

    print(f"Player positions: {season}")

    raw = with_retries(
        lambda: fetch_positions(season),
        f"PlayerIndex {season}",
    )

    print(f"    Rows downloaded: {len(raw)}")

    output = clean_positions(raw, season)

    output = output.drop_duplicates(
        subset=["PERSON_ID"],
        keep="first",
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("PLAYER POSITIONS COMPLETE")
    print("=" * 60)
    print()
    print(f"Rows: {len(output)}")
    print(f"Unique players: {output['PERSON_ID'].nunique()}")
    print()
    print("NBA position distribution:")
    print(output["POSITION"].value_counts(dropna=False).to_string())
    print()
    print(f"Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()