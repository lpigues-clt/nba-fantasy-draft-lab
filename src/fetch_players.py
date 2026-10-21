"""
Download and build historical NBA player-season data.

Historical seasons:
    2022-23
    2023-24
    2024-25
    2025-26

For each player-season this collects:
    - per-game statistics
    - total statistics
    - advanced statistics

Games Started (GS) is added separately after the base statistics
are collected.

Final output:

data/processed/fantasy_player_seasons.csv
"""

import os
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from nba_api.stats.library.parameters import (
    MeasureTypeDetailedDefense,
    PerModeDetailed,
    SeasonTypeAllStar,
)

from common import (
    MASTER_PATH,
    RAW_SEASON_DIR,
    SLEEP_SECONDS,
    TIMEOUT,
    all_seasons,
    ensure_directories,
    with_retries,
)


# ============================================================
# IDENTIFICATION COLUMNS
# ============================================================

KEYS = [
    "PLAYER_ID",
    "TEAM_ID",
]


# ============================================================
# COUNTING STATISTICS
# ============================================================

COUNTING_COLS = [
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "BLKA",
    "PF",
    "PFD",
    "PTS",
    "PLUS_MINUS",
    "NBA_FANTASY_PTS",
    "DD2",
    "TD3",
]


# ============================================================
# NBA API
# ============================================================

def fetch_table(season, measure, per_mode):
    """
    Download one LeagueDashPlayerStats table.
    """

    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=SeasonTypeAllStar.regular,
        per_mode_detailed=per_mode,
        measure_type_detailed_defense=measure,
        timeout=TIMEOUT,
    )

    frames = response.get_data_frames()

    if not frames:
        raise RuntimeError(
            f"NBA API returned no data frames for {season}."
        )

    df = frames[0].copy()

    if df.empty:
        raise RuntimeError(
            f"NBA API returned zero rows for {season}."
        )

    return df


# ============================================================
# CACHE
# ============================================================

def cached(view, season, measure, per_mode):
    """
    Load a cached NBA API response when available.
    Otherwise download and save it.
    """

    ensure_directories()

    path = os.path.join(
        RAW_SEASON_DIR,
        f"{view}_{season}.csv",
    )

    if os.path.exists(path):
        print(f"    Using cached {view}")
        return pd.read_csv(path)

    print(f"    Downloading {view}")

    df = with_retries(
        lambda: fetch_table(
            season,
            measure,
            per_mode,
        ),
        f"{view} {season}",
    )

    if df.empty:
        raise RuntimeError(
            f"{view} {season} returned zero rows."
        )

    df.to_csv(
        path,
        index=False,
    )

    time.sleep(SLEEP_SECONDS)

    return df


# ============================================================
# CLEANING
# ============================================================

def drop_noise(df):
    """
    Remove ranking/helper fields that are not needed.
    """

    drop = [
        column
        for column in df.columns
        if column.endswith("_RANK")
    ]

    drop += [
        column
        for column in [
            "CFID",
            "CFPARAMS",
        ]
        if column in df.columns
    ]

    return df.drop(
        columns=drop,
        errors="ignore",
    )


# ============================================================
# BUILD ONE SEASON
# ============================================================

def build_season(season):
    """
    Build one historical player-season table.
    """

    print()
    print(f"Season {season}")

    base = MeasureTypeDetailedDefense.base
    advanced = MeasureTypeDetailedDefense.advanced

    # --------------------------------------------------------
    # Per-game statistics
    # --------------------------------------------------------

    per_game = drop_noise(
        cached(
            "base_pergame",
            season,
            base,
            PerModeDetailed.per_game,
        )
    )

    # --------------------------------------------------------
    # Total statistics
    # --------------------------------------------------------

    totals = drop_noise(
        cached(
            "base_totals",
            season,
            base,
            PerModeDetailed.totals,
        )
    )

    # --------------------------------------------------------
    # Advanced statistics
    # --------------------------------------------------------

    advanced_df = drop_noise(
        cached(
            "advanced_pergame",
            season,
            advanced,
            PerModeDetailed.per_game,
        )
    )

    # --------------------------------------------------------
    # Merge totals.
    # --------------------------------------------------------

    available_counting = [
        column
        for column in COUNTING_COLS
        if column in totals.columns
        and column in per_game.columns
    ]

    total_table = totals[
        KEYS + available_counting
    ].copy()

    total_table = total_table.rename(
        columns={
            column: f"{column}_TOTAL"
            for column in available_counting
        }
    )

    df = per_game.merge(
        total_table,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Merge advanced statistics.
    # --------------------------------------------------------

    advanced_columns = [
        column
        for column in advanced_df.columns
        if column not in df.columns
    ]

    df = df.merge(
        advanced_df[
            KEYS + advanced_columns
        ],
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Season identifiers.
    # --------------------------------------------------------

    df.insert(
        0,
        "SEASON_START_YEAR",
        int(season[:4]),
    )

    df.insert(
        0,
        "SEASON",
        season,
    )

    print(f"    Rows: {len(df)}")

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    frames = []

    for season in all_seasons():

        df = build_season(season)

        frames.append(df)

    # --------------------------------------------------------
    # Combine all seasons.
    # --------------------------------------------------------

    master = pd.concat(
        frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Put identifying fields first.
    # --------------------------------------------------------

    lead = [
        "SEASON",
        "SEASON_START_YEAR",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
    ]

    remaining = [
        column
        for column in master.columns
        if column not in lead
    ]

    master = master[
        lead + remaining
    ]

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    master.to_csv(
        MASTER_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("HISTORICAL PLAYER DATA COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Player-season rows: {len(master)}"
    )
    print(
        f"Unique players: {master['PLAYER_ID'].nunique()}"
    )
    print(
        f"Seasons: {master['SEASON'].nunique()}"
    )
    print()
    print(
        f"Saved to:\n{MASTER_PATH}"
    )


if __name__ == "__main__":
    main()