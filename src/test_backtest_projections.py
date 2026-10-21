"""Run: python -m pytest src/test_backtest_projections.py -q"""
import pandas as pd
from backtest_projections import load_seasons, generate_predictions, TARGET_YEARS


def test_backtest_never_uses_target_or_future_season():
    d = load_seasons()
    baseline = generate_predictions(d)
    cols = ['TARGET_YEAR','PLAYER_ID','LAG1_FPPG','WEIGHTED2_FPPG','WEIGHTED3_FPPG','SHRUNK_FPPG','LAG1_GP','WEIGHTED2_GP','WEIGHTED3_GP']
    for target in TARGET_YEARS:
        mutated = d.copy()
        mutated.loc[mutated.SEASON_START_YEAR.ge(target), ['PTS','FPPG','GP']] = 999
        altered = generate_predictions(mutated)
        original_year = baseline[baseline.TARGET_YEAR.eq(target)][cols].reset_index(drop=True)
        altered_year = altered[altered.TARGET_YEAR.eq(target)][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_year, altered_year, check_exact=False)



def test_predictions_have_only_prior_history():
    d = load_seasons()
    p = generate_predictions(d)
    assert set(p.TARGET_YEAR.unique()) == set(TARGET_YEARS)
    assert (p.LAG_YEARS >= 1).all()
    assert not p.duplicated(['TARGET_YEAR','PLAYER_ID']).any()
