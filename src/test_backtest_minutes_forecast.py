import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from backtest_projections import load_seasons
from backtest_minutes_forecast import generate,evaluate,METHODS

def test_no_target_season_leakage():
    d=load_seasons(); p=generate(d)
    modified=d.copy(); mask=modified.SEASON_START_YEAR.ge(2024)
    modified.loc[mask,'MIN']=1; modified.loc[mask,'FPPG']=999; modified.loc[mask,'AGE']=99
    q=generate(modified)
    cols=['PLAYER_ID',*METHODS,'STEP3_AGE_FPPG']
    a=p[p.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    b=q[q.TARGET_YEAR.eq(2024)][cols].sort_values('PLAYER_ID').reset_index(drop=True)
    assert a.equals(b)

def test_matched_cohorts_and_unique_players():
    p=generate(load_seasons()); assert not p.duplicated(['TARGET_YEAR','PLAYER_ID']).any()
    m=evaluate(p); s=m[(m.TARGET_YEAR.eq('ALL'))&(m.COHORT.eq('all_returning'))&(m.TARGET.eq('ACTUAL_MIN'))]
    assert s.N.nunique()==1
    assert np.allclose(p.CARRY_MIN,p.PRIOR_MIN)
