"""
Build the 2026-27 NBA schedule from the official NBA schedule PDF.

Input:
    data/raw/2026-27 NBA Schedule by Date.pdf

Output:
    data/raw/schedule_2026_27.csv

The PDF contains one row per NBA game. This script converts those rows
into one row per team-game so that later analysis can calculate:
    - total games
    - playoff-round games
    - championship-round games
    - total playoff-week games
"""

import os
import re

import pandas as pd
from pypdf import PdfReader

from common import (
    PLAYOFF_ROUND_1_END,
    PLAYOFF_ROUND_1_START,
    CHAMPIONSHIP_END,
    CHAMPIONSHIP_START,
    SCHEDULE_PATH,
    ensure_directories,
)


PDF_PATH = os.path.join(
    os.path.dirname(SCHEDULE_PATH),
    "2026-27 NBA Schedule by Date.pdf",
)


TEAM_NAME_TO_ABBREVIATION = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Oklahoma City": "OKC",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS",
}


GAME_ROW_PATTERN = re.compile(
    r"^(\d+)\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\.\s+"
    r"(\d{1,2}/\d{1,2}/\d{2})\s+"
    r"(.+?)\s+(?:at|vs)\s+(.+?)\s+"
    r"(\d{1,2}:\d{2}\s+[AP]M)\s+"
    r"(\d{1,2}:\d{2}\s+[AP]M)"
    r"(?:\s+(.*))?$"
)


def classify_playoff_window(game_date):
    """Return the fantasy playoff classification for a game date."""
    date_string = game_date.strftime("%Y-%m-%d")

    if PLAYOFF_ROUND_1_START <= date_string <= PLAYOFF_ROUND_1_END:
        return "ROUND_1"

    if CHAMPIONSHIP_START <= date_string <= CHAMPIONSHIP_END:
        return "CHAMPIONSHIP"

    return "REGULAR_SEASON"


def extract_pdf_text():
    """Extract all text from the official NBA schedule PDF."""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Could not find schedule PDF:\n{PDF_PATH}"
        )

    print("Reading official NBA schedule PDF...")
    print(f"PDF: {PDF_PATH}")

    reader = PdfReader(PDF_PATH)

    print(f"Pages: {len(reader.pages)}")

    page_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            page_text.append(text)

        print(
            f"    Page {page_number:2d}: "
            f"{len(text or ''):,} characters"
        )

    return "\n".join(page_text)


def parse_game_rows(text):
    """Parse individual NBA games from extracted PDF text."""
    games = []
    unmatched = []

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()

        if not line:
            continue

        match = GAME_ROW_PATTERN.match(line)

        if not match:
            continue

        (
            game_number,
            day_name,
            date_text,
            away_team,
            home_team,
            local_time,
            eastern_time,
            broadcast,
        ) = match.groups()

        away_team = away_team.strip()
        home_team = home_team.strip()

        if away_team not in TEAM_NAME_TO_ABBREVIATION:
            unmatched.append(
                f"Unknown away team: {away_team} | {line}"
            )
            continue

        if home_team not in TEAM_NAME_TO_ABBREVIATION:
            unmatched.append(
                f"Unknown home team: {home_team} | {line}"
            )
            continue

        game_date = pd.to_datetime(
            date_text,
            format="%m/%d/%y",
        )

        games.append(
            {
                "GAME_NUMBER": int(game_number),
                "GAME_DATE": game_date,
                "DAY": day_name,
                "AWAY_TEAM": away_team,
                "AWAY_ABBREVIATION": TEAM_NAME_TO_ABBREVIATION[
                    away_team
                ],
                "HOME_TEAM": home_team,
                "HOME_ABBREVIATION": TEAM_NAME_TO_ABBREVIATION[
                    home_team
                ],
                "LOCAL_TIME": local_time,
                "EASTERN_TIME": eastern_time,
                "BROADCAST": broadcast or "",
            }
        )

    if unmatched:
        print()
        print(
            f"WARNING: {len(unmatched)} parsed rows had "
            "unrecognized team names."
        )

        for item in unmatched[:20]:
            print(f"    {item}")

    return pd.DataFrame(games)


def expand_to_team_games(games):
    """
    Convert one row per NBA game into one row per team-game.

    Each actual game therefore produces exactly two rows:
        one for the away team
        one for the home team
    """
    rows = []

    for _, game in games.iterrows():
        playoff_window = classify_playoff_window(
            game["GAME_DATE"]
        )

        game_id = (
            f"2026-27_"
            f"{game['GAME_DATE'].strftime('%Y%m%d')}_"
            f"{game['AWAY_ABBREVIATION']}_"
            f"{game['HOME_ABBREVIATION']}_"
            f"{game['GAME_NUMBER']}"
        )

        rows.append(
            {
                "SEASON": "2026-27",
                "GAME_ID": game_id,
                "GAME_NUMBER": game["GAME_NUMBER"],
                "GAME_DATE": game["GAME_DATE"],
                "DAY": game["DAY"],
                "TEAM_ABBREVIATION": game["AWAY_ABBREVIATION"],
                "OPPONENT_ABBREVIATION": game[
                    "HOME_ABBREVIATION"
                ],
                "HOME_AWAY": "AWAY",
                "MATCHUP": (
                    f"{game['AWAY_ABBREVIATION']} at "
                    f"{game['HOME_ABBREVIATION']}"
                ),
                "EASTERN_TIME": game["EASTERN_TIME"],
                "BROADCAST": game["BROADCAST"],
                "PLAYOFF_WINDOW": playoff_window,
                "IS_ROUND_1": playoff_window == "ROUND_1",
                "IS_CHAMPIONSHIP": playoff_window == "CHAMPIONSHIP",
                "IS_FANTASY_PLAYOFF": (
                    playoff_window
                    in {"ROUND_1", "CHAMPIONSHIP"}
                ),
            }
        )

        rows.append(
            {
                "SEASON": "2026-27",
                "GAME_ID": game_id,
                "GAME_NUMBER": game["GAME_NUMBER"],
                "GAME_DATE": game["GAME_DATE"],
                "DAY": game["DAY"],
                "TEAM_ABBREVIATION": game["HOME_ABBREVIATION"],
                "OPPONENT_ABBREVIATION": game[
                    "AWAY_ABBREVIATION"
                ],
                "HOME_AWAY": "HOME",
                "MATCHUP": (
                    f"{game['HOME_ABBREVIATION']} vs "
                    f"{game['AWAY_ABBREVIATION']}"
                ),
                "EASTERN_TIME": game["EASTERN_TIME"],
                "BROADCAST": game["BROADCAST"],
                "PLAYOFF_WINDOW": playoff_window,
                "IS_ROUND_1": playoff_window == "ROUND_1",
                "IS_CHAMPIONSHIP": playoff_window == "CHAMPIONSHIP",
                "IS_FANTASY_PLAYOFF": (
                    playoff_window
                    in {"ROUND_1", "CHAMPIONSHIP"}
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_schedule(games, team_games):
    """Run structural checks before saving the schedule."""
    print()
    print("=" * 60)
    print("SCHEDULE VALIDATION")
    print("=" * 60)

    print()
    print(f"Actual games parsed: {len(games):,}")
    print(f"Team-game rows:       {len(team_games):,}")

    if len(team_games) != len(games) * 2:
        raise RuntimeError(
            "Expected exactly two team-game rows per NBA game."
        )

    duplicate_games = games.duplicated(
        subset=[
            "GAME_DATE",
            "AWAY_ABBREVIATION",
            "HOME_ABBREVIATION",
        ]
    ).sum()

    if duplicate_games:
        raise RuntimeError(
            f"Found {duplicate_games} duplicate NBA games."
        )

    team_counts = team_games[
        "TEAM_ABBREVIATION"
    ].value_counts()

    print()
    print("Games by team:")

    print(team_counts.sort_index().to_string())

    if len(team_counts) != 30:
        raise RuntimeError(
            f"Expected 30 NBA teams, found {len(team_counts)}."
        )

    print()
    print("Playoff-window games by team:")

    playoff_counts = (
        team_games[
            team_games["IS_FANTASY_PLAYOFF"]
        ]
        .groupby(
            ["TEAM_ABBREVIATION", "PLAYOFF_WINDOW"]
        )
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    print(playoff_counts.to_string())

    print()
    print("Date range:")
    print(
        f"    {games['GAME_DATE'].min().strftime('%Y-%m-%d')}"
        f" through "
        f"{games['GAME_DATE'].max().strftime('%Y-%m-%d')}"
    )

    playoff_games = games[
        (
            games["GAME_DATE"]
            >= pd.Timestamp(PLAYOFF_ROUND_1_START)
        )
        & (
            games["GAME_DATE"]
            <= pd.Timestamp(CHAMPIONSHIP_END)
        )
    ]

    print()
    print(
        "NBA games during the fantasy playoff window: "
        f"{len(playoff_games):,}"
    )


def main():
    ensure_directories()

    text = extract_pdf_text()

    print()
    print("Parsing NBA game rows...")

    games = parse_game_rows(text)

    if games.empty:
        raise RuntimeError(
            "No NBA game rows were parsed from the PDF."
        )

    games = games.sort_values(
        ["GAME_DATE", "GAME_NUMBER"]
    ).reset_index(drop=True)

    team_games = expand_to_team_games(games)

    team_games = team_games.sort_values(
        [
            "GAME_DATE",
            "GAME_NUMBER",
            "TEAM_ABBREVIATION",
        ]
    ).reset_index(drop=True)

    validate_schedule(
        games,
        team_games,
    )

    team_games["GAME_DATE"] = team_games[
        "GAME_DATE"
    ].dt.strftime("%Y-%m-%d")

    team_games.to_csv(
        SCHEDULE_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("2026-27 NBA SCHEDULE COMPLETE")
    print("=" * 60)
    print()
    print(f"Saved {len(team_games):,} team-game rows to:")
    print(SCHEDULE_PATH)
    print()
    print("First 10 rows:")
    print(
        team_games.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()