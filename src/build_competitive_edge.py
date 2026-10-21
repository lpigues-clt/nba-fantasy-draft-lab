
"""
Competitive Edge: reliability and context overlay.

Uses independent annual observations from fantasy_scored.csv.

Does NOT replace Draft Value rankings or modify Draft Value files.

Competitive Edge:
    35% Availability Reliability
    30% Role Stability
    20% Playoff Schedule
    15% Historical Confidence

Optional draft overlay:
    Draft Value * (1 + bounded Competitive Edge adjustment)

The adjustment is capped at +/- 5%.

2026-27 playoff team assignments are provisional and currently use
the player's 2025-26 representative team.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    FEATURES_PATH,
    PROCESSED_DIR,
    SCHEDULE_PATH,
)


EDGE_PATH = Path(PROCESSED_DIR) / "competitive_edge.csv"

DRAFT_PATH = (
    Path(PROCESSED_DIR)
    / "fantasy_value_model.csv"
)

AVAILABILITY_WEIGHT = 0.35
ROLE_WEIGHT = 0.30
PLAYOFF_WEIGHT = 0.20
HISTORY_WEIGHT = 0.15

MAX_DRAFT_ADJUSTMENT = 0.05
NEUTRAL_SCORE = 50.0


def percentile(values, higher_is_better=True):
    """Percentile scores from 0 to 100."""

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    if not higher_is_better:
        values = -values

    return values.rank(
        pct=True,
        method="average",
    ) * 100


def neutralize(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(NEUTRAL_SCORE)


def shrink_to_neutral(score, season_count):
    """
    Reduce confidence in stability estimates from short histories.

    1 season: neutral stability
    2 seasons: 1/3 observed, 2/3 neutral
    3 seasons: 2/3 observed, 1/3 neutral
    4 seasons: fully observed
    """

    evidence = (
        (season_count - 1) / 3
    ).clip(0, 1)

    return (
        evidence * score
        + (1 - evidence) * NEUTRAL_SCORE
    )


def weighted_recent_average(values):
    """
    Chronological annual values, oldest to newest.

    More recent seasons receive greater weight.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    weights = np.arange(
        1,
        len(values) + 1,
        dtype=float,
    )

    return float(
        np.average(
            values,
            weights=weights,
        )
    )


def build_annual_history(full):
    """
    Calculate player reliability from actual annual observations,
    not overlapping 2YR/3YR/4YR rolling averages.
    """

    required = [
        "PLAYER_NAME",
        "SEASON_START_YEAR",
        "GP",
        "MINUTES_PER_GAME",
    ]

    missing = [
        col for col in required
        if col not in full.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required feature columns: "
            + ", ".join(missing)
        )

    full = full.copy()

    for col in [
        "SEASON_START_YEAR",
        "GP",
        "MINUTES_PER_GAME",
    ]:
        full[col] = pd.to_numeric(
            full[col],
            errors="coerce",
        )

    if full[
        ["PLAYER_NAME", "SEASON_START_YEAR"]
    ].duplicated().any():
        raise RuntimeError(
            "Expected one row per player-season. "
            "Rebuild fantasy_scored.csv before proceeding."
        )

    latest_year = int(
        full["SEASON_START_YEAR"].max()
    )

    latest = full.loc[
        full["SEASON_START_YEAR"] == latest_year
    ].copy()

    if latest["PLAYER_NAME"].duplicated().any():
        raise RuntimeError(
            "Duplicate latest-season players detected."
        )

    records = []

    for player, history in full.groupby(
        "PLAYER_NAME",
        sort=False,
    ):
        history = history.sort_values(
            "SEASON_START_YEAR"
        ).tail(4)

        seasons = len(history)

        gp = history["GP"].to_numpy(
            dtype=float
        )

        minutes = history[
            "MINUTES_PER_GAME"
        ].to_numpy(dtype=float)

        # GP is historical NBA games played.
        # Clamp to 82 for a conventional availability scale.
        availability = (
            np.clip(gp, 0, 82)
            / 82
            * 100
        )

        valid_availability = availability[
            np.isfinite(availability)
        ]

        valid_minutes = minutes[
            np.isfinite(minutes)
        ]

        if len(valid_availability):
            availability_base = (
                weighted_recent_average(
                    valid_availability
                )
            )

            availability_floor = float(
                np.min(valid_availability)
            )
        else:
            availability_base = np.nan
            availability_floor = np.nan

        if len(valid_availability) >= 2:
            availability_std = float(
                np.std(
                    valid_availability,
                    ddof=0,
                )
            )
        else:
            availability_std = np.nan

        if len(valid_minutes) >= 2:
            mean_minutes = float(
                np.mean(valid_minutes)
            )

            if mean_minutes > 0:
                minutes_cv = float(
                    np.std(
                        valid_minutes,
                        ddof=0,
                    )
                    / mean_minutes
                )
            else:
                minutes_cv = np.nan

            prior_mean = float(
                np.mean(valid_minutes[:-1])
            )

            current_minutes = float(
                valid_minutes[-1]
            )

            if prior_mean > 0:
                minutes_continuity_gap = abs(
                    current_minutes / prior_mean - 1
                )
            else:
                minutes_continuity_gap = np.nan

            # Year-to-year changes, not overlapping averages.
            minutes_change = np.diff(
                valid_minutes
            )

            minutes_change_magnitude = float(
                np.mean(
                    np.abs(minutes_change)
                )
            )
        else:
            minutes_cv = np.nan
            minutes_continuity_gap = np.nan
            minutes_change_magnitude = np.nan

        records.append(
            {
                "PLAYER_NAME": player,
                "EDGE_ACTUAL_SEASONS": seasons,
                "EDGE_AVAILABILITY_BASE":
                    availability_base,
                "EDGE_AVAILABILITY_FLOOR":
                    availability_floor,
                "EDGE_AVAILABILITY_STD":
                    availability_std,
                "EDGE_MINUTES_CV":
                    minutes_cv,
                "EDGE_MINUTES_CONTINUITY_GAP":
                    minutes_continuity_gap,
                "EDGE_MINUTES_CHANGE_MAGNITUDE":
                    minutes_change_magnitude,
            }
        )

    history_df = pd.DataFrame(records)

    result = latest.merge(
        history_df,
        on="PLAYER_NAME",
        how="left",
        validate="one_to_one",
    )

    return result


def build_availability(df):
    """
    Historical availability and consistency.

    50% recent-weighted availability
    30% year-to-year stability
    20% historical availability floor
    """

    df = df.copy()

    df["EDGE_AVAILABILITY_BASE_SCORE"] = (
        neutralize(
            percentile(
                df["EDGE_AVAILABILITY_BASE"]
            )
        )
    )

    raw_stability = neutralize(
        percentile(
            df["EDGE_AVAILABILITY_STD"],
            higher_is_better=False,
        )
    )

    df["EDGE_AVAILABILITY_STABILITY"] = (
        shrink_to_neutral(
            raw_stability,
            df["EDGE_ACTUAL_SEASONS"],
        )
    )

    df["EDGE_AVAILABILITY_FLOOR_SCORE"] = (
        neutralize(
            percentile(
                df["EDGE_AVAILABILITY_FLOOR"]
            )
        )
    )

    df["EDGE_AVAILABILITY_SCORE"] = (
        0.50
        * df["EDGE_AVAILABILITY_BASE_SCORE"]
        + 0.30
        * df["EDGE_AVAILABILITY_STABILITY"]
        + 0.20
        * df["EDGE_AVAILABILITY_FLOOR_SCORE"]
    )

    return df


def build_role(df):
    """
    Stability of minutes, not absolute playing time.

    50% minute variation
    30% latest-to-prior continuity
    20% year-to-year minute change magnitude

    All three are based on independent annual observations.
    """

    df = df.copy()

    raw_minute_stability = neutralize(
        percentile(
            df["EDGE_MINUTES_CV"],
            higher_is_better=False,
        )
    )

    raw_continuity = neutralize(
        percentile(
            df["EDGE_MINUTES_CONTINUITY_GAP"],
            higher_is_better=False,
        )
    )

    raw_change_stability = neutralize(
        percentile(
            df["EDGE_MINUTES_CHANGE_MAGNITUDE"],
            higher_is_better=False,
        )
    )

    season_count = df[
        "EDGE_ACTUAL_SEASONS"
    ]

    df["EDGE_MINUTE_STABILITY_SCORE"] = (
        shrink_to_neutral(
            raw_minute_stability,
            season_count,
        )
    )

    df["EDGE_ROLE_CONTINUITY_SCORE"] = (
        shrink_to_neutral(
            raw_continuity,
            season_count,
        )
    )

    df["EDGE_MINUTE_TREND_STABILITY_SCORE"] = (
        shrink_to_neutral(
            raw_change_stability,
            season_count,
        )
    )

    df["EDGE_ROLE_STABILITY_SCORE"] = (
        0.50
        * df["EDGE_MINUTE_STABILITY_SCORE"]
        + 0.30
        * df["EDGE_ROLE_CONTINUITY_SCORE"]
        + 0.20
        * df["EDGE_MINUTE_TREND_STABILITY_SCORE"]
    )

    return df


def build_schedule(df):
    """
    Use actual 2026-27 NBA schedule.

    Player-to-team assignment remains provisional.
    """

    schedule = pd.read_csv(
        SCHEDULE_PATH
    )

    required = [
        "TEAM_ABBREVIATION",
        "PLAYOFF_WINDOW",
    ]

    missing = [
        col for col in required
        if col not in schedule.columns
    ]

    if missing:
        raise RuntimeError(
            "Schedule missing columns: "
            + ", ".join(missing)
        )

    counts = (
        schedule.groupby(
            [
                "TEAM_ABBREVIATION",
                "PLAYOFF_WINDOW",
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for period in [
        "ROUND_1",
        "CHAMPIONSHIP",
    ]:
        if period not in counts.columns:
            counts[period] = 0

    counts = counts[
        [
            "TEAM_ABBREVIATION",
            "ROUND_1",
            "CHAMPIONSHIP",
        ]
    ].rename(
        columns={
            "ROUND_1":
                "EDGE_ROUND_1_GAMES",
            "CHAMPIONSHIP":
                "EDGE_CHAMPIONSHIP_GAMES",
        }
    )

    df = df.merge(
        counts,
        on="TEAM_ABBREVIATION",
        how="left",
        validate="many_to_one",
    )

    valid_schedule = (
        df["EDGE_ROUND_1_GAMES"].notna()
        & df["EDGE_CHAMPIONSHIP_GAMES"].notna()
    )

    df["EDGE_TOTAL_PLAYOFF_GAMES"] = (
        df["EDGE_ROUND_1_GAMES"]
        + df["EDGE_CHAMPIONSHIP_GAMES"]
    )

    # Keep equal weight for both two-week playoff rounds.
    round_score = percentile(
        df["EDGE_ROUND_1_GAMES"]
    )

    championship_score = percentile(
        df["EDGE_CHAMPIONSHIP_GAMES"]
    )

    df["EDGE_PLAYOFF_SCHEDULE_SCORE"] = (
        0.50 * round_score
        + 0.50 * championship_score
    ).fillna(NEUTRAL_SCORE)

    df["EDGE_PLAYOFF_TEAM_MAPPING"] = np.where(
        valid_schedule,
        "2025-26_TEAM_PROXY",
        "UNKNOWN_TEAM_NEUTRAL_SCORE",
    )

    return df


def build_final(df):
    """
    Competitive Edge is a standalone reliability/context score.
    """

    df = df.copy()

    df["EDGE_HISTORY_CONFIDENCE"] = (
        df["EDGE_ACTUAL_SEASONS"]
        .clip(0, 4)
        / 4
        * 100
    )

    df["COMPETITIVE_EDGE_SCORE"] = (
        AVAILABILITY_WEIGHT
        * df["EDGE_AVAILABILITY_SCORE"]
        + ROLE_WEIGHT
        * df["EDGE_ROLE_STABILITY_SCORE"]
        + PLAYOFF_WEIGHT
        * df["EDGE_PLAYOFF_SCHEDULE_SCORE"]
        + HISTORY_WEIGHT
        * df["EDGE_HISTORY_CONFIDENCE"]
    )

    df["COMPETITIVE_EDGE_RANK"] = (
        df["COMPETITIVE_EDGE_SCORE"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    return df


def add_draft_overlay(df):
    """
    Optional small Competitive Edge adjustment.

    Does not modify fantasy_value_model.csv.

    A score of 50 produces no adjustment.
    A score of 100 produces +5%.
    A score of 0 produces -5%.

    This is an advisory sensitivity score, not a projection
    of expected fantasy points.
    """

    if not DRAFT_PATH.exists():
        raise FileNotFoundError(
            f"Draft Value file not found: {DRAFT_PATH}"
        )

    draft = pd.read_csv(
        DRAFT_PATH
    )

    required = [
        "PLAYER_NAME",
        "DRAFT_VALUE_SCORE",
    ]

    missing = [
        col for col in required
        if col not in draft.columns
    ]

    if missing:
        raise RuntimeError(
            "Draft Value missing columns: "
            + ", ".join(missing)
        )

    if draft["PLAYER_NAME"].duplicated().any():
        raise RuntimeError(
            "Duplicate Draft Value player names detected."
        )

    draft = draft[required].copy()

    df = df.merge(
        draft,
        on="PLAYER_NAME",
        how="left",
        validate="one_to_one",
    )

    if df["DRAFT_VALUE_SCORE"].isna().any():
        raise RuntimeError(
            "Some Competitive Edge players have no Draft Value. "
            "Overlay calculation stopped."
        )

    df["EDGE_DRAFT_ADJUSTMENT_PCT"] = (
        (
            df["COMPETITIVE_EDGE_SCORE"]
            - NEUTRAL_SCORE
        )
        / 50
        * MAX_DRAFT_ADJUSTMENT
        * 100
    ).clip(
        -MAX_DRAFT_ADJUSTMENT * 100,
        MAX_DRAFT_ADJUSTMENT * 100,
    )

    df["EDGE_ADJUSTED_DRAFT_SCORE"] = (
        df["DRAFT_VALUE_SCORE"]
        * (
            1
            + df["EDGE_DRAFT_ADJUSTMENT_PCT"]
            / 100
        )
    )

    df["DRAFT_VALUE_RANK"] = (
        df["DRAFT_VALUE_SCORE"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    df["EDGE_ADJUSTED_DRAFT_RANK"] = (
        df["EDGE_ADJUSTED_DRAFT_SCORE"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    df["EDGE_DRAFT_RANK_CHANGE"] = (
        df["DRAFT_VALUE_RANK"]
        - df["EDGE_ADJUSTED_DRAFT_RANK"]
    )

    return df


def main():
    print("=" * 70)
    print("COMPETITIVE EDGE — FINAL RELIABILITY MODEL")
    print("=" * 70)

    full = pd.read_csv(
        FEATURES_PATH
    )

    print(
        f"Historical player-season rows: {len(full):,}"
    )

    print(
        "Building independent annual history..."
    )
    df = build_annual_history(full)

    print(
        "Building availability reliability..."
    )
    df = build_availability(df)

    print(
        "Building minute-based role stability..."
    )
    df = build_role(df)

    print(
        "Building playoff schedule context..."
    )
    df = build_schedule(df)

    print(
        "Building Competitive Edge..."
    )
    df = build_final(df)

    print(
        "Applying capped draft overlay..."
    )
    df = add_draft_overlay(df)

    preferred = [
        "PLAYER_NAME",
        "PERSON_ID",
        "TEAM_ABBREVIATION",
        "SEASON_START_YEAR",
        "EDGE_ACTUAL_SEASONS",
        "EDGE_HISTORY_CONFIDENCE",
        "EDGE_AVAILABILITY_BASE",
        "EDGE_AVAILABILITY_STD",
        "EDGE_AVAILABILITY_FLOOR",
        "EDGE_AVAILABILITY_BASE_SCORE",
        "EDGE_AVAILABILITY_STABILITY",
        "EDGE_AVAILABILITY_FLOOR_SCORE",
        "EDGE_AVAILABILITY_SCORE",
        "EDGE_MINUTES_CV",
        "EDGE_MINUTES_CONTINUITY_GAP",
        "EDGE_MINUTES_CHANGE_MAGNITUDE",
        "EDGE_MINUTE_STABILITY_SCORE",
        "EDGE_ROLE_CONTINUITY_SCORE",
        "EDGE_MINUTE_TREND_STABILITY_SCORE",
        "EDGE_ROLE_STABILITY_SCORE",
        "EDGE_ROUND_1_GAMES",
        "EDGE_CHAMPIONSHIP_GAMES",
        "EDGE_TOTAL_PLAYOFF_GAMES",
        "EDGE_PLAYOFF_SCHEDULE_SCORE",
        "EDGE_PLAYOFF_TEAM_MAPPING",
        "COMPETITIVE_EDGE_SCORE",
        "COMPETITIVE_EDGE_RANK",
        "DRAFT_VALUE_SCORE",
        "DRAFT_VALUE_RANK",
        "EDGE_DRAFT_ADJUSTMENT_PCT",
        "EDGE_ADJUSTED_DRAFT_SCORE",
        "EDGE_ADJUSTED_DRAFT_RANK",
        "EDGE_DRAFT_RANK_CHANGE",
    ]

    output = df[
        [
            col for col in preferred
            if col in df.columns
        ]
    ].copy()

    if output["PLAYER_NAME"].duplicated().any():
        raise RuntimeError(
            "Duplicate players in Competitive Edge output."
        )

    if output["COMPETITIVE_EDGE_SCORE"].isna().any():
        raise RuntimeError(
            "Missing Competitive Edge scores."
        )

    if (
        output["EDGE_DRAFT_ADJUSTMENT_PCT"]
        .abs()
        > MAX_DRAFT_ADJUSTMENT * 100 + 1e-9
    ).any():
        raise RuntimeError(
            "Draft adjustment exceeded its configured cap."
        )

    output = output.sort_values(
        "COMPETITIVE_EDGE_RANK"
    ).reset_index(drop=True)

    output.to_csv(
        EDGE_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(f"Players: {len(output):,}")
    print(
        "Unique players: "
        f"{output['PLAYER_NAME'].nunique():,}"
    )

    print(
        "Missing Competitive Edge scores: "
        f"{output['COMPETITIVE_EDGE_SCORE'].isna().sum()}"
    )

    print()
    print("Actual historical seasons:")
    print(
        output["EDGE_ACTUAL_SEASONS"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Competitive Edge range: "
        f"{output['COMPETITIVE_EDGE_SCORE'].min():.2f} "
        "to "
        f"{output['COMPETITIVE_EDGE_SCORE'].max():.2f}"
    )

    print(
        "Draft adjustment range: "
        f"{output['EDGE_DRAFT_ADJUSTMENT_PCT'].min():+.2f}% "
        "to "
        f"{output['EDGE_DRAFT_ADJUSTMENT_PCT'].max():+.2f}%"
    )

    print()
    print("TOP 15 BY EDGE-ADJUSTED DRAFT SCORE")
    print()

    columns = [
        "PLAYER_NAME",
        "DRAFT_VALUE_RANK",
        "EDGE_ADJUSTED_DRAFT_RANK",
        "DRAFT_VALUE_SCORE",
        "COMPETITIVE_EDGE_SCORE",
        "EDGE_DRAFT_ADJUSTMENT_PCT",
        "EDGE_ADJUSTED_DRAFT_SCORE",
    ]

    print(
        output.sort_values(
            "EDGE_ADJUSTED_DRAFT_RANK"
        )[columns]
        .head(15)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print("GIANNIS VS ROYCE SANITY CHECK")
    print()

    names = [
        "Giannis Antetokounmpo",
        "Royce O'Neale",
    ]

    print(
        output[
            output["PLAYER_NAME"].isin(names)
        ][columns]
        .sort_values(
            "EDGE_ADJUSTED_DRAFT_RANK"
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print()
    print(f"Saved: {EDGE_PATH}")
    print("Original Draft Value file: unchanged")
    print("COMPLETE")


if __name__ == "__main__":
    main()
