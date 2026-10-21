"""Transparent, strategy-specific draft-board percentile scores.

Best Overall is independently overridden by best_overall_calibration in app.py.
These are heuristic strategies, not learned coefficients or market ADP estimates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from draft_engine import make_tiers

# Each strategy has distinct primary drivers. Percentiles use the full player pool.
PROFILES = {
    "High Upside vs Reliability": {
        "FANTASY_PTS_PER36": .42, "FANTASY_PTS_PER_GAME": .28,
        "OPPORTUNITY_ROLE_SCORE": .20, "AVAILABILITY_SCORE": .10,
    },
    "Consistency vs Production": {
        "HISTORICAL_SCORE": .38, "FANTASY_PTS_PER_GAME": .32,
        "AVAILABILITY_SCORE": .20, "GP": .10,
    },
    "Efficiency vs Opportunity": {
        "EFFICIENCY_SCORE": .42, "OPPORTUNITY_ROLE_SCORE": .38,
        "FANTASY_PTS_PER_GAME": .12, "MINUTES_PER_GAME": .08,
    },
    "Availability and Durability": {
        "AVAILABILITY_SCORE": .50, "GP": .25,
        "HISTORICAL_SCORE": .15, "FANTASY_PTS_PER_GAME": .10,
    },
    "Playoff-Focused Drafting": {
        "PLAYOFF_SCHEDULE_SCORE": .55, "FANTASY_PTS_PER_GAME": .20,
        "DRAFT_VALUE_SCORE": .15, "COMPETITIVE_EDGE_SCORE": .10,
    },
    "Find Undervalued Players": {
        "EDGE_ADJUSTED_DRAFT_SCORE": .42, "COMPETITIVE_EDGE_SCORE": .28,
        "DRAFT_VALUE_SCORE": .20, "FANTASY_PTS_PER_GAME": .10,
    },
    "Role Expansion and Breakouts": {
        "OPPORTUNITY_ROLE_SCORE": .38, "FANTASY_PTS_PER36": .32,
        "MINUTES_PER_GAME": .18, "FANTASY_PTS_PER_GAME": .12,
    },
}


def calibrated_tiers(frame: pd.DataFrame, axes: tuple[str, str, str, str], preset: str) -> pd.DataFrame:
    """Score on the original full universe; do not rerank when players are drafted."""
    if preset not in PROFILES:
        return make_tiers(frame, axes)
    weights = PROFILES[preset]
    missing = set(weights).difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for {preset}: {sorted(missing)}")
    result = frame.copy()
    score = pd.Series(0.0, index=result.index)
    valid = pd.Series(True, index=result.index)
    for column, weight in weights.items():
        values = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        ranks = values.rank(pct=True, method="average") * 100
        score = score.add(ranks.fillna(0) * weight)
        valid &= ranks.notna()
    result["CHART_SCORE"] = score.where(valid).round(2)
    result = result.sort_values(["CHART_SCORE", "PLAYER_NAME"], ascending=[False, True], na_position="last", kind="stable")
    result["TIER"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    top = result.loc[result["CHART_SCORE"].notna()].head(300)
    result.loc[top.index, "TIER"] = pd.array([i // 50 + 1 for i in range(len(top))], dtype="Int64")
    return result
