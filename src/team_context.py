"""Verified 2026-27 roster overlay; historical source files remain untouched."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def apply_team_context(frame: pd.DataFrame, moves_path=None) -> pd.DataFrame:
    path = Path(moves_path) if moves_path else ROOT / 'data/research/verified_2026_team_moves.csv'
    moves = pd.read_csv(path)
    if moves.PLAYER_NAME.duplicated().any():
        raise ValueError('Duplicate player in verified moves')
    out = frame.copy()
    out['HISTORICAL_TEAM_2025'] = out['TEAM_ABBREVIATION']
    lookup = moves.set_index('PLAYER_NAME')['TEAM_2026_27']
    out['TEAM_ABBREVIATION'] = out['PLAYER_NAME'].map(lookup).fillna(out['TEAM_ABBREVIATION'])
    out['TEAM_2026_VERIFIED'] = out['PLAYER_NAME'].isin(lookup.index)
    return out


def apply_availability_history(frame: pd.DataFrame, history_path=None) -> pd.DataFrame:
    """Multi-season GP evidence, no diagnoses or named-player overrides."""
    path = Path(history_path) if history_path else ROOT / 'data/processed/fantasy_player_seasons.csv'
    history = pd.read_csv(path, usecols=['PLAYER_NAME', 'SEASON_START_YEAR', 'GP'])
    history = history[history.SEASON_START_YEAR.isin([2023, 2024, 2025])]
    history = history.drop_duplicates(['PLAYER_NAME', 'SEASON_START_YEAR'])
    pivot = history.pivot(index='PLAYER_NAME', columns='SEASON_START_YEAR', values='GP')
    out = frame.copy()
    for year in (2023, 2024, 2025):
        out[f'GP_{year}'] = out.PLAYER_NAME.map(pivot[year]) if year in pivot else float('nan')
    return out
