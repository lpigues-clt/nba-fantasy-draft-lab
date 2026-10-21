import numpy as np
from backtest_projections import load_seasons
from build_experimental_projections import generate


def test_no_future_leakage():
    d = load_seasons()
    first = generate(d)
    extra = d[d.SEASON_START_YEAR.eq(2025)].copy()
    extra['SEASON_START_YEAR'] = 2026
    extra['MIN'] = 40
    extra['FPPG'] = 200
    import pandas as pd
    second = generate(pd.concat([d, extra], ignore_index=True))
    assert first.equals(second)


def test_projection_coverage_and_chronology():
    d = load_seasons()
    p = generate(d)
    assert p.PLAYER_ID.is_unique
    assert p.TRAIN_MAX_YEAR.max() < 2026
    assert np.isfinite(p.EXPERIMENTAL_ROLE50_FPPG).all()
    latest = d[d.SEASON_START_YEAR.eq(2025)]
    assert set(latest.PLAYER_ID).issubset(set(p.PLAYER_ID))
