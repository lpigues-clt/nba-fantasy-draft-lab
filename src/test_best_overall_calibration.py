import pandas as pd
from best_overall_calibration import best_overall_values

def test_higher_forecast_dominates_when_gp_equal():
    frame = pd.DataFrame({'MODEL_DRAFT_FPPG':[55,45], 'FANTASY_PTS_PER_GAME':[54,44], 'GP':[64,64]})
    out = best_overall_values(frame)
    assert out.loc[0,'BEST_OVERALL_FP_EQUIV'] > out.loc[1,'BEST_OVERALL_FP_EQUIV']

def test_gp_penalty_is_bounded_and_monotonic():
    frame = pd.DataFrame({'MODEL_DRAFT_FPPG':[50]*3,'FANTASY_PTS_PER_GAME':[50]*3,'GP':[0,41,82]})
    out = best_overall_values(frame)
    assert out.BEST_OVERALL_FP_EQUIV.round(4).tolist() == [42,46,50]

def test_missing_projection_is_flagged_and_uses_prior():
    frame = pd.DataFrame({'MODEL_DRAFT_FPPG':[None,40], 'FANTASY_PTS_PER_GAME':[35,35], 'GP':[65,65]})
    out = best_overall_values(frame)
    assert out.BEST_OVERALL_FALLBACK.tolist() == [True,False]
    assert out.loc[0,'BEST_OVERALL_FP_EQUIV'] < out.loc[1,'BEST_OVERALL_FP_EQUIV']
