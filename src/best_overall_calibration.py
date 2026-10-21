"""Production-first *forecast* with explicitly bounded risk adjustments.

Scores are expressed on a 0-100 percentile scale for UI compatibility, but
ranking is based on the underlying risk-adjusted projected FP/G. This is an
interpretable heuristic, not an independently validated expected-season-total
forecast. No player-specific rules or external expert ranks.
"""
import numpy as np
import pandas as pd


def best_overall_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute a single forward-looking value without composite-score stacking."""
    forecast = (pd.to_numeric(frame['MODEL_DRAFT_FPPG'], errors='coerce')
                if 'MODEL_DRAFT_FPPG' in frame
                else pd.Series(np.nan, index=frame.index))
    # If no eligible role-aware forecast exists, use observed prior-year FP/G,
    # explicitly flagged as fallback; never silently impute a projection.
    prior = pd.to_numeric(frame['FANTASY_PTS_PER_GAME'], errors='coerce')
    fallback = forecast.isna()
    basis = forecast.fillna(prior)
    gp = pd.to_numeric(frame['GP'], errors='coerce').clip(0, 82)
    years = ((2025, .55), (2024, .30), (2023, .15))
    weighted = pd.Series(0.0, index=frame.index)
    observed_weight = pd.Series(0.0, index=frame.index)
    for year, weight in years:
        col = f"GP_{year}"
        if col in frame:
            val = pd.to_numeric(frame[col], errors="coerce").clip(0, 82)
            weighted += val.fillna(0) * weight
            observed_weight += val.notna().astype(float) * weight
    historical_gp = weighted.div(observed_weight.where(observed_weight.gt(0)))
    effective_gp = historical_gp.fillna(gp).fillna(65)
    gp_factor = 0.84 + 0.16 * effective_gp / 82
    raw = basis * gp_factor
    # Do not stack Draft Value, Edge, schedule, minutes or historical scores:
    # they overlap with production and/or are unverified for future seasons.
    pct = raw.rank(pct=True, method='average') * 100
    return pd.DataFrame({'BEST_OVERALL_FP_EQUIV': raw,
                         'BEST_OVERALL_SCORE': pct.round(2),
                         'BEST_OVERALL_FALLBACK': fallback,
                         'AVAILABILITY_GP_EVIDENCE': effective_gp.round(1),
                         'AVAILABILITY_DISCOUNT_PCT': ((1-gp_factor)*100).round(2)}, index=frame.index)
