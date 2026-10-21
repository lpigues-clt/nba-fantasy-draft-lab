import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from backtest_projections import load_seasons
from backtest_role_forecast import generate, transitions, evaluate


def test_chronological_training_and_matched_cohorts():
    d=load_seasons(); p=generate(d)
    assert len(p)==935
    assert not p.duplicated(['TARGET_YEAR','PLAYER_ID']).any()
    assert (p.TRAIN_MAX_YEAR<p.TARGET_YEAR).all()
    assert (p.TRAIN_N>0).all()
    assert np.isfinite(p[['STEP3_AGE','ROLE_25','ROLE_50']].to_numpy()).all()
    metrics=evaluate(p)
    all_rows=metrics[metrics.COHORT.eq('all_returning') & metrics.TARGET_YEAR.eq('ALL')]
    assert all_rows.N.nunique()==1


def test_future_data_cannot_change_prior_forecasts():
    d=load_seasons(); original=generate(d)
    modified=d.copy()
    modified.loc[modified.SEASON_START_YEAR.eq(2025),'MIN'] *= 0.5
    modified.loc[modified.SEASON_START_YEAR.eq(2025),'PTS'] += 100
    updated=generate(modified)
    cols=['PLAYER_ID','ROLE_DELTA_MIN','ROLE_25','ROLE_50']
    left=original[original.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    right=updated[updated.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right)
