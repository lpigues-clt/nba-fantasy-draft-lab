"""Pure, notebook-independent draft analysis functions. No ESPN/network calls."""
from __future__ import annotations
import numpy as np
import pandas as pd

AXIS_WEIGHTS = (0.35, 0.35, 0.20, 0.10)
TIER_SIZE = 50
TIER_LIMIT = 300

def scheduled_picks(teams: int, slot: int | None, rounds: int = 13) -> list[int]:
    if not isinstance(teams, int) or not 2 <= teams <= 14:
        raise ValueError('teams must be an integer between 2 and 14')
    if slot is None:
        return []
    if not isinstance(slot, int) or not 1 <= slot <= teams:
        raise ValueError('draft slot must be between 1 and team count')
    return [(r-1)*teams + (slot if r % 2 else teams-slot+1) for r in range(1, rounds+1)]

def chart_score(frame: pd.DataFrame, axes: tuple[str, str, str, str]) -> pd.Series:
    """Score full universe, never a filtered/drafted subset; missing values stay unscored."""
    if len(axes) != 4:
        raise ValueError('Exactly four axis columns required')
    combined = {}
    for column, weight in zip(axes, AXIS_WEIGHTS):
        if column not in frame:
            raise KeyError(f'Missing metric: {column}')
        combined[column] = combined.get(column, 0.0) + weight
    weighted = pd.Series(0.0, index=frame.index)
    valid = pd.Series(True, index=frame.index)
    for column, weight in combined.items():
        values = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        percentile = values.rank(pct=True, method='average') * 100
        valid &= percentile.notna()
        weighted += percentile.fillna(0) * weight
    return weighted.where(valid).round(2)

def make_tiers(frame: pd.DataFrame, axes: tuple[str, str, str, str], limit: int = TIER_LIMIT, tier_size: int = TIER_SIZE) -> pd.DataFrame:
    """Stable analysis tiers; drafted status and chart filters do not affect membership."""
    if limit <= 0 or tier_size <= 0:
        raise ValueError('limit and tier_size must be positive')
    ordered = frame.copy()
    ordered['CHART_SCORE'] = chart_score(ordered, axes)
    ordered = ordered.sort_values(['CHART_SCORE', 'PLAYER_NAME'], ascending=[False, True], na_position='last', kind='stable')
    ordered['TIER'] = pd.Series(pd.NA, index=ordered.index, dtype='Int64')
    valid_top = ordered.loc[ordered['CHART_SCORE'].notna()].head(limit)
    ordered.loc[valid_top.index, 'TIER'] = pd.array([i // tier_size + 1 for i in range(len(valid_top))], dtype='Int64')
    return ordered

def validate_picks(picks: list, players: set[str]) -> None:
    if not isinstance(picks, list):
        raise ValueError('Saved picks must be a list')
    seen = set()
    for number, pick in enumerate(picks, 1):
        if not isinstance(pick, dict):
            raise ValueError(f'Pick #{number} must be an object')
        player, owner = pick.get('player'), pick.get('owner')
        if player not in players or owner not in ('mine', 'opponent') or player in seen:
            raise ValueError(f'Invalid or duplicate pick #{number}: {pick}')
        seen.add(player)

def validate_pick_order(picks: list, teams: int, slot: int | None, rounds: int = 13) -> list[str]:
    """Warnings, not hard errors: user can record an incomplete or test draft."""
    if slot is None:
        return []
    scheduled = set(scheduled_picks(teams, slot, rounds))
    warnings = []
    for number, pick in enumerate(picks, 1):
        expected = 'mine' if number in scheduled else 'opponent'
        if pick['owner'] != expected:
            warnings.append(f'Pick #{number}: marked {pick["owner"]}, expected {expected}')
    return warnings
