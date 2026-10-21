"""
Build positional ranks and replacement-level value.

IMPORTANT:
    ESPN eligibility is NOT inferred from NBA position.

    Until authoritative ESPN eligibility is loaded, players with
    unresolved ESPN eligibility are kept separate and are NOT
    assigned artificial PG/SG/SF/PF/C eligibility.

This script builds the mathematical framework for:

    Overall Draft Value
        ↓
    Position Pool
        ↓
    Positional Rank
        ↓
    Replacement Level
        ↓
    Marginal Roster Value
"""

import os

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, RAW_DIR, ensure_directories


MODEL_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_value_model.csv",
)

ELIGIBILITY_PATH = os.path.join(
    RAW_DIR,
    "position_eligibility.csv",
)

OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_position_value.csv",
)


# ============================================================
# LEAGUE STRUCTURE
# ============================================================

NUM_TEAMS = 10

STARTER_SLOTS = {
    "PG": 1,
    "SG": 1,
    "SF": 1,
    "PF": 1,
    "C": 1,
    "G": 1,
    "F": 1,
    "UTIL": 3,
}

# Maximum rostered centers in the league.
MAX_CENTERS = 4


# ============================================================
# POSITION DEFINITIONS
# ============================================================

POSITION_ORDER = [
    "PG",
    "SG",
    "SF",
    "PF",
    "C",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find model file:\n{MODEL_PATH}"
        )

    if not os.path.isfile(ELIGIBILITY_PATH):
        raise FileNotFoundError(
            f"Could not find eligibility file:\n{ELIGIBILITY_PATH}"
        )

    model = pd.read_csv(MODEL_PATH)
    eligibility = pd.read_csv(ELIGIBILITY_PATH)

    return model, eligibility


# ============================================================
# MERGE POSITION INFORMATION
# ============================================================

def merge_position_data(model, eligibility):

    position_columns = [
        "PERSON_ID",
        "PLAYER_NAME",
        "NBA_POSITION",
        "NBA_POSITION_GROUP",
        "ESPN_PRIMARY_POSITION",
        "ESPN_ELIGIBLE_POSITIONS",
        "ESPN_SOURCE",
        "ESPN_CONFIDENCE",
    ]

    available_columns = [
        column
        for column in position_columns
        if column in eligibility.columns
    ]

    eligibility_subset = eligibility[
        available_columns
    ].copy()

    # Player name is currently the safest common key because
    # the historical model does not yet contain PERSON_ID.
    output = model.merge(
        eligibility_subset,
        on="PLAYER_NAME",
        how="left",
        suffixes=("", "_POSITION"),
    )

    return output


# ============================================================
# PARSE ESPN ELIGIBILITY
# ============================================================

def parse_eligibility(value):

    if pd.isna(value):
        return []

    value = str(value).strip().upper()

    if not value:
        return []

    # Expected future format:
    #
    # PG
    # PG,SG
    # C,PF
    # PG,SG,SF
    #
    # This intentionally does NOT infer positions from NBA data.

    positions = [
        item.strip()
        for item in value.replace("/", ",").split(",")
        if item.strip()
    ]

    valid_positions = {
        "PG",
        "SG",
        "SF",
        "PF",
        "C",
    }

    return [
        position
        for position in positions
        if position in valid_positions
    ]


# ============================================================
# BUILD POSITION POOLS
# ============================================================

def build_position_pools(df):

    df = df.copy()

    df["ESPN_POSITION_LIST"] = (
        df["ESPN_ELIGIBLE_POSITIONS"]
        .apply(parse_eligibility)
    )

    for position in POSITION_ORDER:

        column = f"ELIGIBLE_{position}"

        df[column] = df[
            "ESPN_POSITION_LIST"
        ].apply(
            lambda positions: position in positions
        )

    return df


# ============================================================
# POSITIONAL RANK
# ============================================================

def calculate_position_ranks(df):

    df = df.copy()

    for position in POSITION_ORDER:

        eligible_mask = df[
            f"ELIGIBLE_{position}"
        ]

        eligible = df.loc[
            eligible_mask
        ].sort_values(
            "DRAFT_VALUE_SCORE",
            ascending=False,
        )

        rank_column = (
            f"{position}_RANK"
        )

        df[rank_column] = np.nan

        if not eligible.empty:

            df.loc[
                eligible.index,
                rank_column,
            ] = range(
                1,
                len(eligible) + 1,
            )

    return df


# ============================================================
# REPLACEMENT LEVEL
# ============================================================

def calculate_replacement_levels(df):

    df = df.copy()

    # This is deliberately conservative.
    #
    # A dedicated position has one starting slot per team.
    # Therefore the first 10 players at a position represent
    # the basic starter pool.
    #
    # G/F/UTIL scarcity will be modeled separately later.

    for position in POSITION_ORDER:

        eligible = df[
            df[f"ELIGIBLE_{position}"]
        ].sort_values(
            "DRAFT_VALUE_SCORE",
            ascending=False,
        )

        replacement_index = NUM_TEAMS

        replacement_column = (
            f"{position}_REPLACEMENT_SCORE"
        )

        if len(eligible) > replacement_index:

            replacement_score = eligible.iloc[
                replacement_index
            ]["DRAFT_VALUE_SCORE"]

        elif not eligible.empty:

            replacement_score = eligible.iloc[-1][
                "DRAFT_VALUE_SCORE"
            ]

        else:

            replacement_score = np.nan

        df[replacement_column] = replacement_score

    return df


# ============================================================
# MARGINAL ROSTER VALUE
# ============================================================

def calculate_marginal_value(df):

    df = df.copy()

    marginal_columns = []

    for position in POSITION_ORDER:

        column = (
            f"{position}_MARGINAL_VALUE"
        )

        df[column] = np.where(
            df[f"ELIGIBLE_{position}"],
            (
                df["DRAFT_VALUE_SCORE"]
                - df[
                    f"{position}_REPLACEMENT_SCORE"
                ]
            ),
            np.nan,
        )

        marginal_columns.append(column)

    # Best positional marginal value available
    # to the player.
    df["BEST_MARGINAL_ROSTER_VALUE"] = (
        df[marginal_columns]
        .max(axis=1, skipna=True)
    )

    return df


# ============================================================
# POSITIONAL SCARCITY
# ============================================================

def calculate_scarcity(df):

    df = df.copy()

    scarcity_columns = []

    for position in POSITION_ORDER:

        column = (
            f"{position}_SCARCITY_SCORE"
        )

        rank_column = (
            f"{position}_RANK"
        )

        # Simple first-pass scarcity signal.
        #
        # Higher score = player is more valuable relative
        # to the available pool at that position.

        df[column] = np.where(
            df[rank_column].notna(),
            100
            * (
                1
                - (
                    df[rank_column] - 1
                )
                / (
                    df[
                        rank_column
                    ].notna().sum()
                    - 1
                )
            ),
            np.nan,
        )

        scarcity_columns.append(column)

    df["BEST_POSITIONAL_SCARCITY"] = (
        df[scarcity_columns]
        .max(axis=1, skipna=True)
    )

    return df


# ============================================================
# FINAL OUTPUT
# ============================================================

def build_output(df):

    columns = [
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "DRAFT_VALUE_SCORE",

        "NBA_POSITION",
        "NBA_POSITION_GROUP",

        "ESPN_PRIMARY_POSITION",
        "ESPN_ELIGIBLE_POSITIONS",
        "ESPN_SOURCE",
        "ESPN_CONFIDENCE",

        "BEST_MARGINAL_ROSTER_VALUE",
        "BEST_POSITIONAL_SCARCITY",
    ]

    for position in POSITION_ORDER:

        columns.extend(
            [
                f"ELIGIBLE_{position}",
                f"{position}_RANK",
                f"{position}_REPLACEMENT_SCORE",
                f"{position}_MARGINAL_VALUE",
                f"{position}_SCARCITY_SCORE",
            ]
        )

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    return df[
        available_columns
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print()
    print("=" * 70)
    print("POSITIONAL VALUE MODEL")
    print("=" * 70)
    print()

    model, eligibility = load_data()

    print(
        f"Players in model: {len(model)}"
    )

    df = merge_position_data(
        model,
        eligibility,
    )

    df = build_position_pools(df)

    df = calculate_position_ranks(df)

    df = calculate_replacement_levels(df)

    df = calculate_marginal_value(df)

    df = calculate_scarcity(df)

    output = build_output(df)

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "ESPN eligibility currently populated:"
    )

    print(
        output[
            "ESPN_ELIGIBLE_POSITIONS"
        ]
        .notna()
        .sum()
    )

    print()
    print(
        "NBA position distribution:"
    )

    print(
        output[
            "NBA_POSITION"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Players with positional eligibility:"
    )

    for position in POSITION_ORDER:

        count = int(
            output[
                f"ELIGIBLE_{position}"
            ].sum()
        )

        print(
            f"  {position}: {count}"
        )

    print()
    print(
        "Saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 70)
    print("POSITIONAL VALUE MODEL COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()