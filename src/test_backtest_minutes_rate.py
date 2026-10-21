import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_projections import load_seasons
from backtest_minutes_rate import generate, evaluate, METHODS


def test_no_future_leakage():
    d = load_seasons()
    original = generate(d)
    altered = d.copy()
    mask = altered.SEASON_START_YEAR.ge(2024)
    altered.loc[mask, 'MIN'] = 1
    altered.loc[mask, 'FPPG'] = 999
    altered.loc[mask, 'AGE'] = 99
    changed = generate(altered)
    cols = ['PLAYER_ID', *METHODS]
    a = original[original.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    b = changed[changed.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    assert a.equals(b)


def test_decomposition_and_matched_samples():
    p = generate(load_seasons())
    assert np.allclose(p.LAG1_FPPG, p.MIN_RATE_LAG1)
    m = evaluate(p)
    assert m[(m.TARGET_YEAR.eq('ALL')) & (m.COHORT.eq('all_returning'))].N.nunique() == 1
    assert p[['TARGET_YEAR','PLAYER_ID']].duplicated().sum() == 0
