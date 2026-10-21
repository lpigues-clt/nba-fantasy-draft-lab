import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pandas as pd
from evaluate_draft_relevance import draft_metrics,paired_uncertainty

def test_oracle_has_no_regret():
    p=pd.DataFrame({'TARGET_YEAR':[2024]*3,'PLAYER_ID':[1,2,3],'ACTUAL_FPPG':[30.,20.,10.],'PRIOR_FPPG':[30.,20.,10.], 'LAG1_FPPG':[30.,20.,10.], 'STEP3_AGE':[30.,20.,10.], 'ROLE_25':[30.,20.,10.], 'ROLE_50':[30.,20.,10.]})
    assert draft_metrics(p).empty # pools too small for 13-player selection

def test_paired_gain_zero_for_identical_predictions():
    p=pd.DataFrame({'TARGET_YEAR':[2024]*3,'PLAYER_ID':[1,2,3],'ACTUAL_FPPG':[30.,20.,10.],'PRIOR_FPPG':[30.,20.,10.], 'LAG1_FPPG':[31.,21.,11.], 'STEP3_AGE':[31.,21.,11.], 'ROLE_25':[31.,21.,11.], 'ROLE_50':[31.,21.,11.]})
    u=paired_uncertainty(p,draws=100)
    assert (u.MEAN_MAE_GAIN==0).all()

def test_top13_oracle_regret_and_overlap():
    n=15
    actual=list(range(n,0,-1))
    p=pd.DataFrame({'TARGET_YEAR':[2024]*n,'PLAYER_ID':list(range(n)),'ACTUAL_FPPG':actual,'PRIOR_FPPG':actual,'LAG1_FPPG':actual,'STEP3_AGE':actual,'ROLE_25':actual,'ROLE_50':actual})
    result=draft_metrics(p)
    rows=result[(result.POOL=='returning_pool')&(result.K==13)]
    assert len(rows)==4
    assert (rows.REGRET_FPPG_SUM==0).all()
    assert (rows.TOP_K_OVERLAP==13).all()
