"""
Build the fantasy basketball draft value model.

The model is intentionally modular so each component has a distinct job:

    Production          50%
    Historical          20%
    Efficiency          10%
    Opportunity / Role  10%
    Availability         5%
    Playoff Schedule     5%

Production:
    80% observed fantasy PPG
    20% reliability-adjusted per-36

Per-36 reliability is based on estimated season minutes:
    season minutes = GP * MIN

Reliability:
    season_minutes / (season_minutes + 500)

This shrinks small-sample per-36 performance toward the player's
observed fantasy PPG percentile.

IMPORTANT:
    The playoff schedule currently uses the player's representative
    2025-26 team as a temporary mapping to the 2026-27 schedule.
    This is a provisional component until 2026-27 player/team mapping
    is available.
"""

import numpy as np
import pandas as pd

from common import (
    FEATURES_PATH,
    SCHEDULE_PATH,
    PROCESSED_DIR,
    ensure_directories,
)


# ============================================================
# MODEL WEIGHTS
# ============================================================

WEIGHTS = {
    "PRODUCTION": 0.50,
    "HISTORICAL": 0.20,
    "EFFICIENCY": 0.10,
    "OPPORTUNITY_ROLE": 0.10,
    "AVAILABILITY": 0.05,
    "PLAYOFF_SCHEDULE": 0.05,
}


# Production sub-weights
PRODUCTION_PPG_WEIGHT = 0.80
PRODUCTION_PER36_WEIGHT = 0.20


# Reliability parameters
RELIABILITY_HALF_LIFE_MINUTES = 500


# Output
OUTPUT_PATH = f"{PROCESSED_DIR}/fantasy_value_model.csv"


# ============================================================
# HELPERS
# ============================================================

def percentile_score(series):
    """
    Convert a metric into a 0-100 percentile score.

    Higher raw values are better.
    """
    series = pd.to_numeric(series, errors="coerce")

    valid = series.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    if valid.sum() == 0:
        return result

    result.loc[valid] = (
        series.loc[valid]
        .rank(method="average", pct=True)
        * 100
    )

    return result


def inverse_percentile_score(series):
    """
    Convert a metric into a 0-100 percentile score where
    lower raw values are better.
    """
    series = pd.to_numeric(series, errors="coerce")

    valid = series.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float,
    )

    if valid.sum() == 0:
        return result

    result.loc[valid] = (
        1
        - series.loc[valid]
        .rank(method="average", pct=True)
    ) * 100

    return result


def weighted_average(df, columns, weights):
    """
    Weighted average that tolerates missing values.

    Missing components are excluded and the remaining weights
    are renormalized.
    """
    values = df[columns].astype(float)

    weight_array = np.array(weights, dtype=float)

    weighted = values.mul(weight_array, axis=1)

    numerator = weighted.sum(axis=1, skipna=True)

    available_weights = values.notna().mul(
        weight_array,
        axis=1,
    ).sum(axis=1)

    return numerator / available_weights.replace(0, np.nan)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading scored player data...")

    df = pd.read_csv(FEATURES_PATH)

    required = [
        "PLAYER_NAME",
        "SEASON_START_YEAR",
        "GP",
        "MIN",
        "FANTASY_PTS_PER_GAME",
        "FANTASY_PTS_PER36",
        "NET_FANTASY_EFFICIENCY",
        "FANTASY_PTS_PER_MINUTE",
        "SHOT_VALUE_INDEX",
        "PLAYMAKING_MARGIN",
        "OPPORTUNITY_EVENTS_PER_GAME",
        "MINUTES_PER_GAME",
        "ROLE_STABILITY_SCORE",
        "AVAILABILITY_PCT",
        "PLAYOFF_ROUND_1_GAMES",
        "CHAMPIONSHIP_GAMES",
        "TOTAL_PLAYOFF_GAMES",
        "FANTASY_PTS_PER_GAME_4YR",
        "FANTASY_PTS_PER_GAME_3YR",
        "FANTASY_PTS_PER_GAME_2YR",
        "FANTASY_PTS_PER_GAME_LATEST",
        "FANTASY_PTS_PER_GAME_CV",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "fantasy_scored.csv is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    return df


def load_schedule():
    print("Loading 2026-27 playoff schedule...")

    schedule = pd.read_csv(SCHEDULE_PATH)

    required = [
        "TEAM_ABBREVIATION",
        "PLAYOFF_WINDOW",
        "IS_ROUND_1",
        "IS_CHAMPIONSHIP",
        "IS_FANTASY_PLAYOFF",
    ]

    missing = [
        column
        for column in required
        if column not in schedule.columns
    ]

    if missing:
        raise RuntimeError(
            "schedule_2026_27.csv is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )

    return schedule


# ============================================================
# PLAYOFF SCHEDULE
# ============================================================

def build_schedule_lookup(schedule):
    """
    Build team-level playoff schedule totals.

    The schedule itself is authoritative.

    Player-to-team assignment remains provisional because the
    current player dataset represents the player's 2025-26 team.
    """

    team_schedule = (
        schedule
        .groupby("TEAM_ABBREVIATION")
        .agg(
            PLAYOFF_ROUND_1_GAMES=(
                "IS_ROUND_1",
                "sum",
            ),
            CHAMPIONSHIP_GAMES=(
                "IS_CHAMPIONSHIP",
                "sum",
            ),
            TOTAL_PLAYOFF_GAMES=(
                "IS_FANTASY_PLAYOFF",
                "sum",
            ),
        )
        .reset_index()
    )

    return team_schedule


# ============================================================
# PRODUCTION
# ============================================================

def build_production_scores(df):
    print("Building production scores...")

    # --------------------------------------------------------
    # Observed fantasy PPG
    # --------------------------------------------------------

    df["PROD_FPPG_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_GAME"]
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    # MIN is minutes per game.
    # Therefore estimated season minutes are:
    #
    #     GP * MIN
    #
    # This is intentionally calculated here rather than using
    # MIN alone.

    df["ESTIMATED_SEASON_MINUTES"] = (
        pd.to_numeric(df["GP"], errors="coerce")
        * pd.to_numeric(df["MIN"], errors="coerce")
    )

    df["PER36_RELIABILITY"] = (
        df["ESTIMATED_SEASON_MINUTES"]
        / (
            df["ESTIMATED_SEASON_MINUTES"]
            + RELIABILITY_HALF_LIFE_MINUTES
        )
    )

    df["PER36_RELIABILITY"] = (
        df["PER36_RELIABILITY"]
        .clip(lower=0, upper=1)
    )

    # --------------------------------------------------------
    # Raw per-36 percentile
    # --------------------------------------------------------

    df["RAW_PER36_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER36"]
    )

    # --------------------------------------------------------
    # Reliability-adjusted per-36
    # --------------------------------------------------------
    #
    # A player with strong per-36 production but a tiny sample
    # is pulled toward their actual fantasy PPG percentile.
    #
    # A player with a large sample receives more of the raw
    # per-36 signal.

    df["RELIABLE_PER36_SCORE"] = (
        (
            df["PER36_RELIABILITY"]
            * df["RAW_PER36_SCORE"]
        )
        +
        (
            (1 - df["PER36_RELIABILITY"])
            * df["PROD_FPPG_SCORE"]
        )
    )

    # --------------------------------------------------------
    # Final production score
    # --------------------------------------------------------

    df["PRODUCTION_SCORE"] = (
        (
            df["PROD_FPPG_SCORE"]
            * PRODUCTION_PPG_WEIGHT
        )
        +
        (
            df["RELIABLE_PER36_SCORE"]
            * PRODUCTION_PER36_WEIGHT
        )
    )

    return df


# ============================================================
# HISTORICAL CONSISTENCY
# ============================================================

def build_historical_scores(df):
    print("Building historical consistency scores...")

    df["HIST_4YR_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_GAME_4YR"]
    )

    df["HIST_3YR_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_GAME_3YR"]
    )

    df["HIST_2YR_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_GAME_2YR"]
    )

    df["HIST_LATEST_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_GAME_LATEST"]
    )

    historical_base = weighted_average(
        df,
        [
            "HIST_4YR_SCORE",
            "HIST_3YR_SCORE",
            "HIST_2YR_SCORE",
            "HIST_LATEST_SCORE",
        ],
        [
            0.35,
            0.25,
            0.20,
            0.20,
        ],
    )

    # Lower coefficient of variation = greater consistency.
    df["CONSISTENCY_SCORE"] = inverse_percentile_score(
        df["FANTASY_PTS_PER_GAME_CV"]
    )

    df["HISTORICAL_SCORE"] = (
        (historical_base * 0.80)
        + (df["CONSISTENCY_SCORE"] * 0.20)
    )

    return df


# ============================================================
# EFFICIENCY
# ============================================================

def build_efficiency_scores(df):
    print("Building efficiency scores...")

    df["EFF_NET_FANTASY_SCORE"] = percentile_score(
        df["NET_FANTASY_EFFICIENCY"]
    )

    df["EFF_FANTASY_PER_MINUTE_SCORE"] = percentile_score(
        df["FANTASY_PTS_PER_MINUTE"]
    )

    df["EFF_SHOT_VALUE_SCORE"] = percentile_score(
        df["SHOT_VALUE_INDEX"]
    )

    df["EFF_PLAYMAKING_SCORE"] = percentile_score(
        df["PLAYMAKING_MARGIN"]
    )

    df["EFFICIENCY_SCORE"] = weighted_average(
        df,
        [
            "EFF_NET_FANTASY_SCORE",
            "EFF_FANTASY_PER_MINUTE_SCORE",
            "EFF_SHOT_VALUE_SCORE",
            "EFF_PLAYMAKING_SCORE",
        ],
        [
            0.40,
            0.30,
            0.15,
            0.15,
        ],
    )

    return df


# ============================================================
# OPPORTUNITY / ROLE
# ============================================================

def build_opportunity_role_scores(df):
    print("Building opportunity / role scores...")

    df["OPPORTUNITY_EVENTS_SCORE"] = percentile_score(
        df["OPPORTUNITY_EVENTS_PER_GAME"]
    )

    df["MINUTES_SCORE"] = percentile_score(
        df["MINUTES_PER_GAME"]
    )

    df["ROLE_STABILITY_SCORE_NORMALIZED"] = percentile_score(
        df["ROLE_STABILITY_SCORE"]
    )

    df["OPPORTUNITY_ROLE_SCORE"] = weighted_average(
        df,
        [
            "OPPORTUNITY_EVENTS_SCORE",
            "MINUTES_SCORE",
            "ROLE_STABILITY_SCORE_NORMALIZED",
        ],
        [
            0.35,
            0.35,
            0.30,
        ],
    )

    return df


# ============================================================
# AVAILABILITY
# ============================================================

def build_availability_scores(df):
    print("Building availability scores...")

    df["AVAILABILITY_SCORE"] = percentile_score(
        df["AVAILABILITY_PCT"]
    )

    return df


# ============================================================
# PLAYOFF SCHEDULE
# ============================================================

def build_playoff_scores(df, schedule):
    print("Building playoff schedule scores...")

    team_schedule = build_schedule_lookup(schedule)

    # Merge against the player's representative team from the
    # latest season. This is explicitly provisional until actual
    # 2026-27 player/team mapping is available.

    if "TEAM_ABBREVIATION" not in df.columns:
        raise RuntimeError(
            "TEAM_ABBREVIATION is required for the provisional "
            "playoff schedule mapping."
        )

    latest_schedule = team_schedule.copy()

    df = df.merge(
        latest_schedule,
        on="TEAM_ABBREVIATION",
        how="left",
        suffixes=("", "_SCHEDULE"),
    )

    # Prefer schedule-derived values when available.
    df["PLAYOFF_ROUND_1_GAMES_MODEL"] = (
        df["PLAYOFF_ROUND_1_GAMES_SCHEDULE"]
    )

    df["CHAMPIONSHIP_GAMES_MODEL"] = (
        df["CHAMPIONSHIP_GAMES_SCHEDULE"]
    )

    df["TOTAL_PLAYOFF_GAMES_MODEL"] = (
        df["TOTAL_PLAYOFF_GAMES_SCHEDULE"]
    )

    df["PLAYOFF_ROUND_1_SCORE"] = percentile_score(
        df["PLAYOFF_ROUND_1_GAMES_MODEL"]
    )

    df["CHAMPIONSHIP_SCORE"] = percentile_score(
        df["CHAMPIONSHIP_GAMES_MODEL"]
    )

    df["PLAYOFF_SCHEDULE_SCORE"] = weighted_average(
        df,
        [
            "PLAYOFF_ROUND_1_SCORE",
            "CHAMPIONSHIP_SCORE",
        ],
        [
            0.50,
            0.50,
        ],
    )

    return df


# ============================================================
# FINAL MODEL
# ============================================================

def build_final_score(df):
    print("Building final draft value score...")

    df["DRAFT_VALUE_SCORE"] = weighted_average(
        df,
        [
            "PRODUCTION_SCORE",
            "HISTORICAL_SCORE",
            "EFFICIENCY_SCORE",
            "OPPORTUNITY_ROLE_SCORE",
            "AVAILABILITY_SCORE",
            "PLAYOFF_SCHEDULE_SCORE",
        ],
        [
            WEIGHTS["PRODUCTION"],
            WEIGHTS["HISTORICAL"],
            WEIGHTS["EFFICIENCY"],
            WEIGHTS["OPPORTUNITY_ROLE"],
            WEIGHTS["AVAILABILITY"],
            WEIGHTS["PLAYOFF_SCHEDULE"],
        ],
    )

    df = df.sort_values(
        "DRAFT_VALUE_SCORE",
        ascending=False,
    ).reset_index(drop=True)

    df["OVERALL_RANK"] = np.arange(1, len(df) + 1)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_directories()

    df = load_data()
    schedule = load_schedule()

    # Only latest completed season is ranked.
    latest_season = df["SEASON_START_YEAR"].max()

    df = df[
        df["SEASON_START_YEAR"] == latest_season
    ].copy()

    print()
    print(f"Latest season: {latest_season}")
    print(f"Players ranked: {len(df)}")
    print()

    df = build_production_scores(df)
    df = build_historical_scores(df)
    df = build_efficiency_scores(df)
    df = build_opportunity_role_scores(df)
    df = build_availability_scores(df)
    df = build_playoff_scores(df, schedule)
    df = build_final_score(df)

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FANTASY VALUE MODEL COMPLETE")
    print("=" * 70)
    print()

    print("MODEL WEIGHTS:")
    print(
        f"    PRODUCTION           "
        f"{WEIGHTS['PRODUCTION']:.0%}"
    )
    print(
        f"    HISTORICAL           "
        f"{WEIGHTS['HISTORICAL']:.0%}"
    )
    print(
        f"    EFFICIENCY           "
        f"{WEIGHTS['EFFICIENCY']:.0%}"
    )
    print(
        f"    OPPORTUNITY / ROLE   "
        f"{WEIGHTS['OPPORTUNITY_ROLE']:.0%}"
    )
    print(
        f"    AVAILABILITY         "
        f"{WEIGHTS['AVAILABILITY']:.0%}"
    )
    print(
        f"    PLAYOFF SCHEDULE     "
        f"{WEIGHTS['PLAYOFF_SCHEDULE']:.0%}"
    )
    print()

    print("PRODUCTION:")
    print(
        f"    Fantasy PPG          "
        f"{PRODUCTION_PPG_WEIGHT:.0%}"
    )
    print(
        f"    Reliable per-36      "
        f"{PRODUCTION_PER36_WEIGHT:.0%}"
    )
    print()

    print("TOP 30:")
    print(
        df[
            [
                "OVERALL_RANK",
                "PLAYER_NAME",
                "GP",
                "MIN",
                "FANTASY_PTS_PER_GAME",
                "FANTASY_PTS_PER36",
                "PER36_RELIABILITY",
                "PRODUCTION_SCORE",
                "HISTORICAL_SCORE",
                "EFFICIENCY_SCORE",
                "OPPORTUNITY_ROLE_SCORE",
                "AVAILABILITY_SCORE",
                "PLAYOFF_SCHEDULE_SCORE",
                "DRAFT_VALUE_SCORE",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print()
    print("Saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()