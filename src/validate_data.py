"""Step: integrity checks on the player-season master file. FAIL = must fix. WARN = worth a look."""

import os
import sys

import pandas as pd

from common import MASTER_PATH, all_seasons

ANCHORS = [("Joel Embiid", "2022-23", 33.1), ("Shai Gilgeous-Alexander", "2024-25", 32.7)]
FAILS, WARNS = [], []


def flag(name, bad, df, cols, allow_frac=0.0):
    n_bad = int(bad.sum())
    if n_bad == 0:
        print(f"  PASS  {name}")
        return
    frac = n_bad / len(df)
    level = "FAIL" if frac > allow_frac else "WARN"
    (FAILS if level == "FAIL" else WARNS).append(name)
    print(f"  {level}  {name}: {n_bad} rows ({frac:.3%})")
    print(df.loc[bad, cols].head(5).to_string(index=False))


def main():
    if not os.path.exists(MASTER_PATH):
        print(f"Missing {MASTER_PATH}. Run fetch_players.py first.")
        sys.exit(1)
    df = pd.read_csv(MASTER_PATH)
    print(f"Loaded {len(df)} player-season rows")
    ident = ["SEASON", "PLAYER_NAME", "TEAM_ABBREVIATION"]

    missing = [s for s in all_seasons() if s not in set(df["SEASON"])]
    if missing:
        FAILS.append("missing seasons")
        print(f"  FAIL  missing seasons: {missing}")
    else:
        print(f"  PASS  all {len(all_seasons())} seasons present")

    flag("no duplicate player+season+team rows",
         df.duplicated(["PLAYER_ID", "SEASON", "TEAM_ID"], keep=False), df, ident)
    flag("GP is positive and reasonable", (df["GP"] < 1) | (df["GP"] > 100), df, ident + ["GP"])
    flag("per-game PTS at most 45", df["PTS"] > 45, df, ident + ["PTS"])

    for col in ["FG_PCT", "FG3_PCT", "FT_PCT"]:
        if col in df.columns:
            flag(f"{col} between 0 and 1", (df[col] < 0) | (df[col] > 1), df, ident + [col])

    for name, season, expected in ANCHORS:
        rows = df[(df["PLAYER_NAME"] == name) & (df["SEASON"] == season)]
        if rows.empty:
            print(f"  NOTE  {name} {season}: not found (name spelling may differ)")
            continue
        got = float(rows["PTS"].iloc[0])
        if abs(got - expected) <= 0.15:
            print(f"  PASS  {name} {season}: {got:.1f} PPG")
        else:
            WARNS.append(f"anchor {name} {season}")
            print(f"  WARN  {name} {season}: data says {got:.1f}, expected {expected}")

    print(f"\nSUMMARY: {len(FAILS)} FAIL, {len(WARNS)} WARN")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()