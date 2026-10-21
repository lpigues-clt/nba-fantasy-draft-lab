import numpy as np
from backtest_projections import load_seasons
from analyze_role_changes import generate, summarize

def test_arithmetic_decomposition():
    p=generate(load_seasons())
    np.testing.assert_allclose(p.FPPG_CHANGE,p.MIN_CONTRIBUTION+p.RATE_CONTRIBUTION,atol=1e-9)
    assert not p.duplicated(['TARGET_YEAR','PLAYER_ID']).any()
    assert len(p)>800

def test_no_future_predictor_leakage():
    d=load_seasons()
    original=generate(d)
    modified=d.copy()
    # Mutating the target season must not change preseason-known inputs or projections.
    modified.loc[modified.SEASON_START_YEAR.ge(2024),'MIN'] *= 1.1
    altered=generate(modified)
    cols=['PRIOR_AGE','PRIOR_GP','PRIOR_MIN','PRIOR_FPPG','PRIOR_RATE','PRIOR_MIN_TREND','MIN_RATE_AGE']
    a=original[original.TARGET_YEAR.eq(2024)].sort_values('PLAYER_ID').reset_index(drop=True)
    b=altered[altered.TARGET_YEAR.eq(2024)].sort_values('PLAYER_ID').reset_index(drop=True)
    for col in cols:
        np.testing.assert_allclose(a[col],b[col],equal_nan=True)
    assert len(summarize(original))>0
