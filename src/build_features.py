"""
Build fantasy basketball features from the historical NBA player dataset.

Input:
    data/processed/fantasy_player_seasons.csv

Output:
    data/processed/fantasy_scored.csv

This script:

1. Applies the exact ESPN H2H Points scoring system.
2. Calculates fantasy points per game correctly.
3. Calculates per-36 fantasy production.
4. Calculates custom efficiency and production metrics.
5. Calculates availability and opportunity metrics.
6. Calculates role stability without using Games Started.
7. Adds the 2026-27 fantasy playoff schedule.
8. Builds historical 4-year / 3-year / 2-year / latest metrics.
9. Consolidates traded players into one player-season before
   calculating historical features.
"""

import os
import numpy as np
import pandas as pd

from common import (
    FEATURES_PATH,
    MASTER_PATH,
    SCHEDULE_PATH,
    PROCESSED_DIR,
    ensure_directories,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Minimum minutes used when calculating certain rate metrics.
# This does NOT remove players from the dataset.
MIN_MINUTES_FOR_RATE = 100


# ============================================================
# LOAD DATA
# ============================================================

ensure_directories()

print("Loading historical player data...")

raw = pd.read_csv(MASTER_PATH)

print(f"Rows loaded: {len(raw)}")
print(f"Columns loaded: {len(raw.columns)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "SEASON_START_YEAR",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PTS",
]

missing = [
    column
    for column in required_columns
    if column not in raw.columns
]

if missing:
    raise RuntimeError(
        "The historical player dataset is missing required columns:\n"
        + "\n".join(f"    {column}" for column in missing)
    )


# ============================================================
# NUMERIC CLEANUP
# ============================================================

numeric_columns = [
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PTS",
]

for column in numeric_columns:
    raw[column] = pd.to_numeric(
        raw[column],
        errors="coerce",
    )


# Remove rows that cannot be used as player-season observations.

raw = raw[
    raw["PLAYER_NAME"].notna()
    & raw["SEASON_START_YEAR"].notna()
    & raw["GP"].notna()
].copy()


raw["SEASON_START_YEAR"] = (
    raw["SEASON_START_YEAR"]
    .astype(int)
)


# ============================================================
# PLAYER-SEASON CONSOLIDATION
# ============================================================
#
# Important:
#
# The NBA dataset can contain multiple rows for a player in one
# season when that player changed teams.
#
# Example:
#
#     Player X   Team A   30 GP
#     Player X   Team B   40 GP
#
# Those must become one 70-game player-season.
#
# We use GP-weighted averages for per-game statistics.
# This prevents traded players from being counted multiple times
# in historical calculations.
# ============================================================

print()
print("Consolidating player-team rows into player-seasons...")


# Stats that are reported as per-game averages.

per_game_columns = [
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PTS",
]


def weighted_average(group, column):
    """
    GP-weighted average for a per-game statistic.
    """

    values = group[column]
    weights = group["GP"]

    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid],
    )


player_season_rows = []


for (player_name, season_year), group in raw.groupby(
    ["PLAYER_NAME", "SEASON_START_YEAR"],
    sort=False,
):

    row = {
        "PLAYER_NAME": player_name,
        "SEASON_START_YEAR": season_year,
    }

    # Total games across teams.
    row["GP"] = group["GP"].sum()

    # Use the team where the player accumulated the most games
    # as the representative team for that historical season.
    team_games = (
        group.groupby("TEAM_ABBREVIATION")["GP"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(team_games) > 0:
        row["TEAM_ABBREVIATION"] = team_games.index[0]
    else:
        row["TEAM_ABBREVIATION"] = np.nan

    for column in per_game_columns:
        row[column] = weighted_average(
            group,
            column,
        )

    player_season_rows.append(row)


df = pd.DataFrame(player_season_rows)


print(
    f"Player-season rows after consolidation: {len(df)}"
)


# ============================================================
# SEASON LABEL
# ============================================================

def season_label(start_year):
    return f"{int(start_year)}-{str(int(start_year) + 1)[-2:]}"


df["SEASON"] = (
    df["SEASON_START_YEAR"]
    .apply(season_label)
)


# ============================================================
# EXACT FANTASY SCORING
# ============================================================
#
# IMPORTANT:
#
# The NBA statistics above are PER-GAME statistics.
#
# Therefore this calculation already produces fantasy points
# PER GAME.
#
# We must NOT divide this value by GP.
# ============================================================

print()
print("Calculating fantasy scoring...")


df["FANTASY_PTS"] = (
    (df["FGM"] * 2)
    - df["FGA"]
    + df["FTM"]
    - df["FTA"]
    + df["FG3M"]
    + df["REB"]
    + (df["AST"] * 2)
    + (df["STL"] * 4)
    + (df["BLK"] * 4)
    - (df["TOV"] * 2)
    + df["PTS"]
)


df["FANTASY_PTS_PER_GAME"] = (
    df["FANTASY_PTS"]
)


# ============================================================
# PER-36 METRICS
# ============================================================

print("Calculating per-36 metrics...")


def per36(stat, minutes):
    """
    Convert a per-game statistic to a per-36-minute rate.
    """

    return np.where(
        minutes >= MIN_MINUTES_FOR_RATE / 100,
        stat / minutes * 36,
        np.nan,
    )


df["FGM_PER36"] = per36(
    df["FGM"],
    df["MIN"],
)

df["FGA_PER36"] = per36(
    df["FGA"],
    df["MIN"],
)

df["FG3M_PER36"] = per36(
    df["FG3M"],
    df["MIN"],
)

df["FTM_PER36"] = per36(
    df["FTM"],
    df["MIN"],
)

df["FTA_PER36"] = per36(
    df["FTA"],
    df["MIN"],
)

df["REB_PER36"] = per36(
    df["REB"],
    df["MIN"],
)

df["AST_PER36"] = per36(
    df["AST"],
    df["MIN"],
)

df["STL_PER36"] = per36(
    df["STL"],
    df["MIN"],
)

df["BLK_PER36"] = per36(
    df["BLK"],
    df["MIN"],
)

df["TOV_PER36"] = per36(
    df["TOV"],
    df["MIN"],
)

df["PTS_PER36"] = per36(
    df["PTS"],
    df["MIN"],
)

df["FANTASY_PTS_PER36"] = per36(
    df["FANTASY_PTS"],
    df["MIN"],
)


# ============================================================
# SCORING COMPONENT VALUES
# ============================================================

print("Calculating scoring components...")


df["FG_SCORING_VALUE"] = (
    (df["FGM"] * 2)
    - df["FGA"]
)


df["FT_SCORING_VALUE"] = (
    df["FTM"]
    - df["FTA"]
)


df["THREE_POINT_VALUE"] = (
    df["FG3M"]
)


df["REBOUND_VALUE"] = (
    df["REB"]
)


df["PLAYMAKING_VALUE"] = (
    df["AST"] * 2
)


df["STOCK_VALUE"] = (
    (df["STL"] * 4)
    + (df["BLK"] * 4)
)


df["TURNOVER_PENALTY"] = (
    df["TOV"] * -2
)


df["POINT_VALUE"] = (
    df["PTS"]
)


# ============================================================
# CUSTOM PRODUCTION / EFFICIENCY METRICS
# ============================================================

print("Calculating custom metrics...")


# Shot Value Index:
#
# Positive values indicate that field-goal and free-throw
# shooting contributes positively after accounting for attempts.

df["SHOT_VALUE_INDEX"] = (
    df["FG_SCORING_VALUE"]
    + df["FT_SCORING_VALUE"]
)


# Playmaking Margin:
#
# AST value minus turnover penalty.
#
# Since turnovers are negative fantasy events, this measures
# net playmaking contribution.

df["PLAYMAKING_MARGIN"] = (
    df["PLAYMAKING_VALUE"]
    + df["TURNOVER_PENALTY"]
)


# Stocks rate:
#
# Combined steals + blocks fantasy value per 36 minutes.

df["STOCKS_RATE_PER36"] = (
    (df["STL_PER36"] * 4)
    + (df["BLK_PER36"] * 4)
)


# Fantasy points per minute.

df["FANTASY_PTS_PER_MINUTE"] = np.where(
    df["MIN"] > 0,
    df["FANTASY_PTS"] / df["MIN"],
    np.nan,
)


# Net fantasy efficiency:
#
# Fantasy production relative to shot volume.
#
# This is intentionally a descriptive metric rather than a
# replacement for fantasy PPG.

df["NET_FANTASY_EFFICIENCY"] = np.where(
    (df["FGA"] + df["FTA"]) > 0,
    df["FANTASY_PTS"]
    / (df["FGA"] + df["FTA"]),
    np.nan,
)


# ============================================================
# AVAILABILITY
# ============================================================

print("Calculating availability...")


# 82 games is used as the historical full-season reference.
#
# This is not a prediction of future games played.

df["AVAILABILITY_PCT"] = (
    df["GP"] / 82 * 100
)


df["AVAILABILITY_SCORE"] = (
    df["AVAILABILITY_PCT"]
    .clip(lower=0, upper=100)
)


# ============================================================
# OPPORTUNITY
# ============================================================

print("Calculating opportunity...")


# Opportunity events represent possessions / actions that can
# generate fantasy production.
#
# We deliberately do not use Games Started.

df["OPPORTUNITY_EVENTS_PER_GAME"] = (
    df["FGA"]
    + df["FTA"]
    + df["AST"]
    + df["REB"]
)


df["MINUTES_PER_GAME"] = (
    df["MIN"]
)


# ============================================================
# ROLE STABILITY
# ============================================================
#
# No Games Started.
#
# The role signal is based on:
#
#     50% availability
#     50% minutes
#
# This rewards players who reliably remain on the floor without
# assuming that "starter" automatically means "better."
# ============================================================

df["ROLE_STABILITY_SCORE"] = (
    0.50 * df["AVAILABILITY_SCORE"]
    + 0.50 * (
        df["MINUTES_PER_GAME"]
        / df["MINUTES_PER_GAME"].max()
        * 100
    )
)


# ============================================================
# 2026-27 PLAYOFF SCHEDULE
# ============================================================

print()
print("Loading 2026-27 playoff schedule...")


schedule = pd.read_csv(
    SCHEDULE_PATH
)


required_schedule_columns = [
    "TEAM_ABBREVIATION",
    "IS_ROUND_1",
    "IS_CHAMPIONSHIP",
    "IS_FANTASY_PLAYOFF",
]


missing_schedule = [
    column
    for column in required_schedule_columns
    if column not in schedule.columns
]


if missing_schedule:
    raise RuntimeError(
        "Schedule is missing required columns:\n"
        + "\n".join(
            f"    {column}"
            for column in missing_schedule
        )
    )


round_1 = schedule[
    schedule["IS_ROUND_1"].astype(bool)
].copy()


championship = schedule[
    schedule["IS_CHAMPIONSHIP"].astype(bool)
].copy()


round_1_games = (
    round_1
    .groupby("TEAM_ABBREVIATION")
    .size()
    .rename("PLAYOFF_ROUND_1_GAMES")
)


championship_games = (
    championship
    .groupby("TEAM_ABBREVIATION")
    .size()
    .rename("CHAMPIONSHIP_GAMES")
)


playoff_games = (
    schedule[
        schedule["IS_FANTASY_PLAYOFF"].astype(bool)
    ]
    .groupby("TEAM_ABBREVIATION")
    .size()
    .rename("TOTAL_PLAYOFF_GAMES")
)


team_schedule = pd.concat(
    [
        round_1_games,
        championship_games,
        playoff_games,
    ],
    axis=1,
).reset_index()


df = df.merge(
    team_schedule,
    on="TEAM_ABBREVIATION",
    how="left",
)


schedule_columns = [
    "PLAYOFF_ROUND_1_GAMES",
    "CHAMPIONSHIP_GAMES",
    "TOTAL_PLAYOFF_GAMES",
]


for column in schedule_columns:
    df[column] = (
        df[column]
        .fillna(0)
        .astype(int)
    )


# ============================================================
# PLAYOFF SCHEDULE SCORE
# ============================================================

# Both fantasy playoff rounds are two weeks long, so they receive
# equal weight.

round_1_max = max(
    df["PLAYOFF_ROUND_1_GAMES"].max(),
    1,
)

championship_max = max(
    df["CHAMPIONSHIP_GAMES"].max(),
    1,
)


df["PLAYOFF_SCHEDULE_SCORE"] = (
    50 * (
        df["PLAYOFF_ROUND_1_GAMES"]
        / round_1_max
    )
    +
    50 * (
        df["CHAMPIONSHIP_GAMES"]
        / championship_max
    )
)


# ============================================================
# HISTORICAL FEATURES
# ============================================================
#
# These are calculated from the consolidated player-season table.
#
# This is important because traded players now contribute exactly
# one observation per season.
# ============================================================

print()
print("Calculating historical features...")


historical_metrics = [
    "FANTASY_PTS_PER_GAME",
    "FANTASY_PTS_PER36",
    "MINUTES_PER_GAME",
    "AVAILABILITY_PCT",
    "ROLE_STABILITY_SCORE",
]


# Sort chronologically for each player.

df = df.sort_values(
    [
        "PLAYER_NAME",
        "SEASON_START_YEAR",
    ]
).reset_index(drop=True)


for metric in historical_metrics:

    # --------------------------------------------------------
    # Four-year average
    # --------------------------------------------------------

    df[f"{metric}_4YR"] = (
        df.groupby("PLAYER_NAME")[metric]
        .transform(
            lambda s: s.rolling(
                window=4,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Three-year average
    # --------------------------------------------------------

    df[f"{metric}_3YR"] = (
        df.groupby("PLAYER_NAME")[metric]
        .transform(
            lambda s: s.rolling(
                window=3,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Two-year average
    # --------------------------------------------------------

    df[f"{metric}_2YR"] = (
        df.groupby("PLAYER_NAME")[metric]
        .transform(
            lambda s: s.rolling(
                window=2,
                min_periods=1,
            ).mean()
        )
    )

    # --------------------------------------------------------
    # Latest season
    # --------------------------------------------------------

    df[f"{metric}_LATEST"] = (
        df.groupby("PLAYER_NAME")[metric]
        .transform("last")
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    def calculate_trend(values):
        values = values.dropna()

        if len(values) < 2:
            return np.nan

        x = np.arange(len(values))

        slope = np.polyfit(
            x,
            values.to_numpy(),
            1,
        )[0]

        return slope

    trend_by_player = (
        df.groupby("PLAYER_NAME")[metric]
        .apply(calculate_trend)
        .rename(f"{metric}_TREND")
    )

    df = df.drop(
        columns=[
            f"{metric}_TREND"
        ],
        errors="ignore",
    )

    df = df.merge(
        trend_by_player,
        left_on="PLAYER_NAME",
        right_index=True,
        how="left",
    )

    # --------------------------------------------------------
    # Standard deviation
    # --------------------------------------------------------

    df[f"{metric}_STD"] = (
        df.groupby("PLAYER_NAME")[metric]
        .transform("std")
    )

    # --------------------------------------------------------
    # Coefficient of variation
    # --------------------------------------------------------

    df[f"{metric}_CV"] = np.where(
        df[f"{metric}_4YR"].abs() > 0,
        df[f"{metric}_STD"].abs()
        / df[f"{metric}_4YR"].abs(),
        np.nan,
    )


# ============================================================
# RESTORE CHRONOLOGICAL ORDER
# ============================================================

df = df.sort_values(
    [
        "SEASON_START_YEAR",
        "PLAYER_NAME",
    ]
).reset_index(drop=True)


# ============================================================
# COLUMN ORDER
# ============================================================

preferred_columns = [
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "SEASON",
    "SEASON_START_YEAR",
    "GP",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FTM",
    "FTA",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PTS",

    # Fantasy scoring
    "FANTASY_PTS",
    "FANTASY_PTS_PER_GAME",
    "FANTASY_PTS_PER36",

    # Per-36 statistics
    "FGM_PER36",
    "FGA_PER36",
    "FG3M_PER36",
    "FTM_PER36",
    "FTA_PER36",
    "REB_PER36",
    "AST_PER36",
    "STL_PER36",
    "BLK_PER36",
    "TOV_PER36",
    "PTS_PER36",

    # Scoring components
    "FG_SCORING_VALUE",
    "FT_SCORING_VALUE",
    "THREE_POINT_VALUE",
    "REBOUND_VALUE",
    "PLAYMAKING_VALUE",
    "STOCK_VALUE",
    "TURNOVER_PENALTY",
    "POINT_VALUE",

    # Custom metrics
    "SHOT_VALUE_INDEX",
    "PLAYMAKING_MARGIN",
    "STOCKS_RATE_PER36",
    "NET_FANTASY_EFFICIENCY",
    "FANTASY_PTS_PER_MINUTE",

    # Availability / opportunity / role
    "AVAILABILITY_PCT",
    "AVAILABILITY_SCORE",
    "OPPORTUNITY_EVENTS_PER_GAME",
    "MINUTES_PER_GAME",
    "ROLE_STABILITY_SCORE",

    # Playoff schedule
    "PLAYOFF_ROUND_1_GAMES",
    "CHAMPIONSHIP_GAMES",
    "TOTAL_PLAYOFF_GAMES",
    "PLAYOFF_SCHEDULE_SCORE",
]


historical_columns = [
    column
    for column in df.columns
    if any(
        column.startswith(metric + "_")
        for metric in historical_metrics
    )
]


ordered_columns = []

for column in preferred_columns:
    if column in df.columns:
        ordered_columns.append(column)

for column in historical_columns:
    if column not in ordered_columns:
        ordered_columns.append(column)

for column in df.columns:
    if column not in ordered_columns:
        ordered_columns.append(column)


df = df[ordered_columns]


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    FEATURES_PATH,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

print()
print("=" * 60)
print("FEATURE BUILD COMPLETE")
print("=" * 60)
print()

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(
    f"Unique players: {df['PLAYER_NAME'].nunique()}"
)
print(
    f"Seasons: {df['SEASON_START_YEAR'].nunique()}"
)

print()
print("Fantasy scoring sanity check:")

sanity_players = [
    "Nikola Jokić",
    "Shai Gilgeous-Alexander",
    "Luka Dončić",
    "Stephen Curry",
]


sanity = df[
    df["PLAYER_NAME"].isin(sanity_players)
    & (
        df["SEASON_START_YEAR"]
        == df["SEASON_START_YEAR"].max()
    )
][
    [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "GP",
        "MIN",
        "FANTASY_PTS",
        "FANTASY_PTS_PER_GAME",
        "FANTASY_PTS_PER36",
    ]
]


print(
    sanity.to_string(
        index=False
    )
)


print()
print("2026-27 playoff schedule:")
print(
    df[
        [
            "TEAM_ABBREVIATION",
            "PLAYOFF_ROUND_1_GAMES",
            "CHAMPIONSHIP_GAMES",
            "TOTAL_PLAYOFF_GAMES",
        ]
    ]
    .drop_duplicates()
    .sort_values("TEAM_ABBREVIATION")
    .to_string(index=False)
)


print()
print("Saved to:")
print(FEATURES_PATH)