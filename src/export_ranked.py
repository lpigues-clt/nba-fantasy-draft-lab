"""
Export clean ranked fantasy basketball lists from the value model.

Default:
    Sort by DRAFT_VALUE_SCORE and export the full player pool.

Examples:
    python src/export_ranked.py

    python src/export_ranked.py --sort PRODUCTION_SCORE --top 100

    python src/export_ranked.py --sort EFFICIENCY_SCORE --top 50
"""

import argparse
import os
from datetime import datetime

import pandas as pd

from common import PROCESSED_DIR, PROJECT_ROOT


MODEL_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_value_model.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "rankings",
)


EXPORT_COLUMNS = [
    "OVERALL_RANK",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
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


SORT_OPTIONS = [
    "DRAFT_VALUE_SCORE",
    "PRODUCTION_SCORE",
    "HISTORICAL_SCORE",
    "EFFICIENCY_SCORE",
    "OPPORTUNITY_ROLE_SCORE",
    "AVAILABILITY_SCORE",
    "PLAYOFF_SCHEDULE_SCORE",
    "FANTASY_PTS_PER_GAME",
    "FANTASY_PTS_PER36",
    "MIN",
    "GP",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Export a ranked fantasy basketball player list."
    )

    parser.add_argument(
        "--sort",
        default="DRAFT_VALUE_SCORE",
        choices=SORT_OPTIONS,
        help="Column used to rank players.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optional number of players to export.",
    )

    return parser.parse_args()


def validate_model(df):
    required_columns = set(EXPORT_COLUMNS)

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Model output is missing required columns:\n"
            + "\n".join(sorted(missing))
        )

    if df.empty:
        raise RuntimeError(
            "The fantasy value model contains no players."
        )


def build_export(df, sort_column, top_n):
    output = df.copy()

    output = output.sort_values(
        by=sort_column,
        ascending=False,
        na_position="last",
    )

    if top_n is not None:
        if top_n < 1:
            raise ValueError(
                "--top must be at least 1."
            )

        output = output.head(top_n)

    output = output[
        EXPORT_COLUMNS
    ].copy()

    # Re-rank based on the selected metric.
    output["RANK"] = range(1, len(output) + 1)

    output = output[
        [
            "RANK",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
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

    return output


def main():
    args = parse_arguments()

    print()
    print("=" * 70)
    print("FANTASY BASKETBALL RANKING EXPORT")
    print("=" * 70)
    print()

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            "Could not find fantasy value model:\n"
            f"{MODEL_PATH}"
        )

    print("Loading model:")
    print(MODEL_PATH)

    df = pd.read_csv(MODEL_PATH)

    print()
    print(f"Players in model: {len(df)}")

    validate_model(df)

    output = build_export(
        df=df,
        sort_column=args.sort,
        top_n=args.top,
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    suffix = (
        f"top_{args.top}"
        if args.top is not None
        else "all"
    )

    filename = (
        f"ranking_{args.sort.lower()}_{suffix}_"
        f"{timestamp}.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Ranking:")
    print(f"  Sort metric: {args.sort}")
    print(f"  Players exported: {len(output)}")

    print()
    print("Top 20:")
    print(
        output[
            [
                "RANK",
                "PLAYER_NAME",
                "TEAM_ABBREVIATION",
                "DRAFT_VALUE_SCORE",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print()
    print("Saved to:")
    print(output_path)

    print()
    print("=" * 70)
    print("EXPORT COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()