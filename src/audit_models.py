
"""
Audit Draft Value against Competitive Edge.

This script is read-only with respect to the two models.

Inputs:
    data/processed/fantasy_value_model.csv
    data/processed/competitive_edge.csv

Output:
    data/processed/model_audit.csv

Checks:
    - Player counts and uniqueness
    - Player overlap
    - Model score correlation
    - Component correlations
    - Largest rank differences
    - Biggest Competitive Edge risers and fallers
    - Top 25 from each model
"""

import os

import numpy as np
import pandas as pd

from common import PROCESSED_DIR


DRAFT_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_value_model.csv",
)

EDGE_PATH = os.path.join(
    PROCESSED_DIR,
    "competitive_edge.csv",
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "model_audit.csv",
)


def require_columns(df, columns, label):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{label} is missing required columns: "
            + ", ".join(missing)
        )


def correlation(df, first, second):
    if first not in df.columns or second not in df.columns:
        return np.nan

    values = df[[first, second]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if len(values) < 3:
        return np.nan

    return values[first].corr(values[second])


def print_table(df, columns, count=25):
    available = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        df[available]
        .head(count)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


def main():
    print("=" * 72)
    print("DRAFT VALUE VS COMPETITIVE EDGE AUDIT")
    print("=" * 72)

    draft = pd.read_csv(DRAFT_PATH)
    edge = pd.read_csv(EDGE_PATH)

    require_columns(
        draft,
        ["PLAYER_NAME", "DRAFT_VALUE_SCORE"],
        "Draft Value",
    )

    require_columns(
        edge,
        [
            "PLAYER_NAME",
            "COMPETITIVE_EDGE_SCORE",
            "EDGE_ACTUAL_SEASONS",
            "EDGE_HISTORY_CONFIDENCE",
        ],
        "Competitive Edge",
    )

    print()
    print(f"Draft Value rows: {len(draft):,}")
    print(f"Competitive Edge rows: {len(edge):,}")

    for label, df in [
        ("Draft Value", draft),
        ("Competitive Edge", edge),
    ]:
        duplicate_count = df["PLAYER_NAME"].duplicated().sum()

        print(f"{label} duplicate player names: {duplicate_count:,}")

        if duplicate_count:
            raise RuntimeError(
                f"{label} contains duplicate player names. "
                "Audit stopped to avoid an invalid many-to-many merge."
            )

    # Preserve all model columns without collisions.
    # PLAYER_NAME is the current project-level join key.
    draft = draft.rename(
        columns={
            column: f"DRAFT_{column}"
            for column in draft.columns
            if column.startswith("EDGE_")
        }
    )

    draft_columns = [
        "PLAYER_NAME",
        "DRAFT_VALUE_SCORE",
        "PRODUCTION_SCORE",
        "HISTORICAL_SCORE",
        "EFFICIENCY_SCORE",
        "OPPORTUNITY_ROLE_SCORE",
        "AVAILABILITY_SCORE",
        "PLAYOFF_SCHEDULE_SCORE",
    ]

    draft_columns = [
        column
        for column in draft_columns
        if column in draft.columns
    ]

    edge_columns = [
        "PLAYER_NAME",
        "COMPETITIVE_EDGE_SCORE",
        "COMPETITIVE_EDGE_RANK",
        "EDGE_ACTUAL_SEASONS",
        "EDGE_HISTORY_CONFIDENCE",
        "EDGE_AVAILABILITY_SCORE",
        "EDGE_AVAILABILITY_BASE",
        "EDGE_AVAILABILITY_STABILITY",
        "EDGE_ROLE_STABILITY_SCORE",
        "EDGE_MINUTE_STABILITY_SCORE",
        "EDGE_ROLE_CONTINUITY_SCORE",
        "EDGE_MINUTE_TREND_STABILITY_SCORE",
        "EDGE_PLAYOFF_SCHEDULE_SCORE",
        "EDGE_ROUND_1_GAMES",
        "EDGE_CHAMPIONSHIP_GAMES",
        "EDGE_TOTAL_PLAYOFF_GAMES",
        "EDGE_PLAYOFF_TEAM_MAPPING",
    ]

    edge_columns = [
        column
        for column in edge_columns
        if column in edge.columns
    ]

    merged = draft[draft_columns].merge(
        edge[edge_columns],
        on="PLAYER_NAME",
        how="inner",
        validate="one_to_one",
    )

    print(f"Players in both models: {len(merged):,}")

    missing_from_edge = (
        set(draft["PLAYER_NAME"])
        - set(edge["PLAYER_NAME"])
    )

    missing_from_draft = (
        set(edge["PLAYER_NAME"])
        - set(draft["PLAYER_NAME"])
    )

    print(f"Draft-only players: {len(missing_from_edge):,}")
    print(f"Edge-only players: {len(missing_from_draft):,}")

    if merged.empty:
        raise RuntimeError("No overlapping players found.")

    # Calculate ranks directly from model scores.
    # This avoids relying on a pre-existing rank column.
    merged["DRAFT_VALUE_RANK"] = (
        merged["DRAFT_VALUE_SCORE"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    merged["COMPETITIVE_EDGE_RANK"] = (
        merged["COMPETITIVE_EDGE_SCORE"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Positive = player ranks better in Competitive Edge.
    merged["EDGE_RANK_ADVANTAGE"] = (
        merged["DRAFT_VALUE_RANK"]
        - merged["COMPETITIVE_EDGE_RANK"]
    )

    merged["ABS_RANK_DIFFERENCE"] = (
        merged["EDGE_RANK_ADVANTAGE"]
        .abs()
    )

    print()
    print("=" * 72)
    print("OVERALL MODEL RELATIONSHIP")
    print("=" * 72)

    overall_correlation = correlation(
        merged,
        "DRAFT_VALUE_SCORE",
        "COMPETITIVE_EDGE_SCORE",
    )

    print(
        "Draft Value vs Competitive Edge correlation: "
        f"{overall_correlation:.3f}"
    )

    print()
    print(
        "Interpretation: correlation measures overlap, "
        "not whether either model is accurate."
    )

    print()
    print("=" * 72)
    print("COMPONENT RELATIONSHIPS")
    print("=" * 72)

    comparisons = [
        (
            "Availability",
            "AVAILABILITY_SCORE",
            "EDGE_AVAILABILITY_SCORE",
        ),
        (
            "Role vs Opportunity/Role",
            "OPPORTUNITY_ROLE_SCORE",
            "EDGE_ROLE_STABILITY_SCORE",
        ),
        (
            "Playoff Schedule",
            "PLAYOFF_SCHEDULE_SCORE",
            "EDGE_PLAYOFF_SCHEDULE_SCORE",
        ),
        (
            "Historical vs Confidence",
            "HISTORICAL_SCORE",
            "EDGE_HISTORY_CONFIDENCE",
        ),
    ]

    for label, first, second in comparisons:
        value = correlation(merged, first, second)

        if pd.isna(value):
            print(f"{label}: unavailable")
        else:
            print(f"{label}: {value:.3f}")

    comparison_columns = [
        "PLAYER_NAME",
        "DRAFT_VALUE_RANK",
        "COMPETITIVE_EDGE_RANK",
        "EDGE_RANK_ADVANTAGE",
        "DRAFT_VALUE_SCORE",
        "COMPETITIVE_EDGE_SCORE",
        "EDGE_ACTUAL_SEASONS",
        "EDGE_HISTORY_CONFIDENCE",
    ]

    print()
    print("=" * 72)
    print("LARGEST ABSOLUTE RANK DIFFERENCES")
    print("=" * 72)

    print_table(
        merged.sort_values(
            "ABS_RANK_DIFFERENCE",
            ascending=False,
        ),
        comparison_columns,
    )

    print()
    print("=" * 72)
    print("BIGGEST COMPETITIVE EDGE RISERS")
    print("=" * 72)

    print_table(
        merged.sort_values(
            "EDGE_RANK_ADVANTAGE",
            ascending=False,
        ),
        comparison_columns,
    )

    print()
    print("=" * 72)
    print("BIGGEST COMPETITIVE EDGE FALLERS")
    print("=" * 72)

    print_table(
        merged.sort_values(
            "EDGE_RANK_ADVANTAGE",
            ascending=True,
        ),
        comparison_columns,
    )

    top_columns = [
        "PLAYER_NAME",
        "DRAFT_VALUE_RANK",
        "COMPETITIVE_EDGE_RANK",
        "DRAFT_VALUE_SCORE",
        "COMPETITIVE_EDGE_SCORE",
        "EDGE_ACTUAL_SEASONS",
        "EDGE_HISTORY_CONFIDENCE",
    ]

    print()
    print("=" * 72)
    print("TOP 25 BY DRAFT VALUE")
    print("=" * 72)

    print_table(
        merged.sort_values(
            "DRAFT_VALUE_RANK"
        ),
        top_columns,
    )

    print()
    print("=" * 72)
    print("TOP 25 BY COMPETITIVE EDGE")
    print("=" * 72)

    print_table(
        merged.sort_values(
            "COMPETITIVE_EDGE_RANK"
        ),
        top_columns,
    )

    print()
    print("=" * 72)
    print("COMPETITIVE EDGE HISTORY DEPTH")
    print("=" * 72)

    print(
        merged.groupby("EDGE_ACTUAL_SEASONS")
        .agg(
            PLAYERS=("PLAYER_NAME", "count"),
            MEAN_EDGE_SCORE=("COMPETITIVE_EDGE_SCORE", "mean"),
            MEDIAN_EDGE_RANK=("COMPETITIVE_EDGE_RANK", "median"),
        )
        .to_string(
            float_format=lambda value: f"{value:.2f}"
        )
    )

    # Save the full merged audit for later visualization.
    merged = merged.sort_values(
        "DRAFT_VALUE_RANK"
    ).reset_index(drop=True)

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved audit to: {OUTPUT_PATH}")
    print()
    print("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
