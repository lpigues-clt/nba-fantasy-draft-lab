import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_projections import load_seasons
from backtest_productivity_rate import generate, evaluate, METHODS


def test_no_future_leakage():
    d = load_seasons()
    a = generate(d)
    changed = d.copy()
    cols = ['TARGET_YEAR','PLAYER_ID',*METHODS,'PROJECTED_MIN','STEP3_AGE_FPPG']
    for year in (2024, 2025):
        changed = d.copy()
        changed.loc[changed.SEASON_START_YEAR.ge(year), 'FPPG'] += 200
        b = generate(changed)
        pd.testing.assert_frame_equal(a[a.TARGET_YEAR.eq(year)][cols].reset_index(drop=True),
                                      b[b.TARGET_YEAR.eq(year)][cols].reset_index(drop=True))


def test_matched_cohorts_and_rate_identity():
    p = generate(load_seasons())
    assert not p.duplicated(['TARGET_YEAR','PLAYER_ID']).any()
    assert set(p.TARGET_YEAR) == {2024,2025}
    assert np.allclose(p.CARRY_RATE_FPPG, p.PROJECTED_MIN*p.CARRY_RATE)
    m = evaluate(p)
    for year in (2024,2025):
        s = m[(m.TARGET_YEAR==year)&(m.COHORT=='all_returning')&(m.TARGET=='ACTUAL_FPPG')]
        assert s.N.nunique() == 1
