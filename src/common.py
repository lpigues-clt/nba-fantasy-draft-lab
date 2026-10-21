"""Shared settings, paths, and helpers for the fantasy basketball data pipeline."""

import os
import time


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root = Fantasy Draft 2026
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_SEASON_DIR = os.path.join(RAW_DIR, "seasons")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


# ============================================================
# HISTORICAL NBA DATA
# ============================================================

# 2022 means the 2022-23 NBA season.
# 2025 means the 2025-26 NBA season.
FIRST_YEAR = 2022
LAST_YEAR = 2025


# ============================================================
# 2026-27 FANTASY SEASON
# ============================================================

DRAFT_SEASON = "2026-27"

PLAYOFF_ROUND_1_START = "2027-03-01"
PLAYOFF_ROUND_1_END = "2027-03-14"

CHAMPIONSHIP_START = "2027-03-15"
CHAMPIONSHIP_END = "2027-03-28"

PLAYOFF_START = PLAYOFF_ROUND_1_START
PLAYOFF_END = CHAMPIONSHIP_END


# ============================================================
# API SETTINGS
# ============================================================

SLEEP_SECONDS = 1.0
MAX_RETRIES = 4
TIMEOUT = 60


# ============================================================
# FILE PATHS
# ============================================================

MASTER_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_player_seasons.csv",
)

FEATURES_PATH = os.path.join(
    PROCESSED_DIR,
    "fantasy_scored.csv",
)

TEAM_RECORDS_PATH = os.path.join(
    RAW_DIR,
    "team_records.csv",
)

POSITION_PATH = os.path.join(
    RAW_DIR,
    "player_positions.csv",
)

SCHEDULE_PATH = os.path.join(
    RAW_DIR,
    "schedule_2026_27.csv",
)

CROSSWALK_PATH = os.path.join(
    RAW_DIR,
    "position_crosswalk.csv",
)


# ============================================================
# HELPERS
# ============================================================

def season_label(start_year):
    """Convert 2022 -> 2022-23."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def all_seasons():
    """Return all historical seasons used by the project."""
    return [
        season_label(year)
        for year in range(FIRST_YEAR, LAST_YEAR + 1)
    ]


def ensure_directories():
    """Create required data directories if they do not exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RAW_SEASON_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def with_retries(fn, label):
    """Run an API request with retries and exponential backoff."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()

        except Exception as exc:
            print(
                f"    {label}: attempt {attempt} failed "
                f"({type(exc).__name__}: {exc})"
            )

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Giving up on {label} after {MAX_RETRIES} attempts."
                ) from exc

            wait_seconds = 3 * (2 ** (attempt - 1))
            print(f"    Waiting {wait_seconds} seconds before retrying...")
            time.sleep(wait_seconds)