"""Step 7: descriptive role-change audit. Prior-year predictors only; target-year outcomes for evaluation.
Run from project root: python src/analyze_role_changes.py
Writes exclusively to data/research/step7_*.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons, TARGET_YEARS
from backtest_minutes_rate import generate as step3_generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'

def generate(d):
    preds = step3_generate(d)[['TARGET_YEAR','PLAYER_ID','MIN_RATE_AGE','LAG1_FPPG']]
    rows = []
    for year in TARGET_YEARS:
        history = {pid: g.sort_values('SEASON_START_YEAR', ascending=False)
                   for pid,g in d[d.SEASON_START_YEAR.lt(year)].groupby('PLAYER_ID')}
        for actual in d[d.SEASON_START_YEAR.eq(year)].itertuples(index=False):
            h = history.get(actual.PLAYER_ID)
            if h is None or h.empty: continue
            recent = h.iloc[0]
            older = h.iloc[1] if len(h)>1 else None
            if not (np.isfinite(recent.MIN) and recent.MIN>0 and np.isfinite(actual.MIN) and actual.MIN>0): continue
            prior_rate = recent.FPPG/recent.MIN
            current_rate = actual.FPPG/actual.MIN
            old_min = older.MIN if older is not None and np.isfinite(older.MIN) else np.nan
            rows.append(dict(TARGET_YEAR=year, PLAYER_ID=actual.PLAYER_ID, PLAYER_NAME=actual.PLAYER_NAME,
                             PRIOR_AGE=recent.AGE, PRIOR_GP=recent.GP, PRIOR_MIN=recent.MIN,
                             PRIOR_FPPG=recent.FPPG, PRIOR_RATE=prior_rate, PRIOR_MIN_TREND=recent.MIN-old_min,
                             PRIOR_SEASONS=len(h), LAG_YEARS=year-int(recent.SEASON_START_YEAR),
                             ACTUAL_MIN=actual.MIN, ACTUAL_RATE=current_rate, ACTUAL_FPPG=actual.FPPG,
                             MIN_CHANGE=actual.MIN-recent.MIN, RATE_CHANGE=current_rate-prior_rate,
                             FPPG_CHANGE=actual.FPPG-recent.FPPG,
                             # Exact arithmetic decomposition, descriptive only:
                             MIN_CONTRIBUTION=(actual.MIN-recent.MIN)*prior_rate,
                             RATE_CONTRIBUTION=actual.MIN*(current_rate-prior_rate)))
    result = pd.DataFrame(rows).merge(preds,on=['TARGET_YEAR','PLAYER_ID'],how='inner',validate='one_to_one')
    result['AGE_MODEL_ERROR'] = result.MIN_RATE_AGE-result.ACTUAL_FPPG
    result['ABS_AGE_MODEL_ERROR'] = result.AGE_MODEL_ERROR.abs()
    result['AGE_GROUP'] = pd.cut(result.PRIOR_AGE,[-np.inf,24,29,np.inf],labels=['under25','25to29','30plus'])
    result['PRIOR_GP_GROUP'] = pd.cut(result.PRIOR_GP,[-1,29,59,np.inf],labels=['under30','30to59','60plus'])
    result['PRIOR_MIN_GROUP'] = pd.cut(result.PRIOR_MIN,[-np.inf,15,25,np.inf],labels=['under15','15to25','25plus'])
    result['PRIOR_TREND_GROUP'] = pd.cut(result.PRIOR_MIN_TREND,[-np.inf,-3,3,np.inf],labels=['falling','stable','rising'])
    result['ROLE_OUTCOME'] = np.select([result.MIN_CHANGE>=5,result.MIN_CHANGE<=-5],['gain5plus','loss5plus'],default='within5')
    return result

def summarize(p):
    results=[]
    for year in (*TARGET_YEARS,'ALL'):
        part = p if year=='ALL' else p[p.TARGET_YEAR.eq(year)]
        for group_col in ('AGE_GROUP','PRIOR_GP_GROUP','PRIOR_MIN_GROUP','PRIOR_TREND_GROUP','ROLE_OUTCOME'):
            for group_name,g in part.groupby(group_col,observed=True,dropna=False):
                results.append(dict(TARGET_YEAR=year, SEGMENT=group_col, GROUP=str(group_name), N=len(g),
                                    MEAN_MIN_CHANGE=g.MIN_CHANGE.mean(), MEAN_RATE_CHANGE=g.RATE_CHANGE.mean(),
                                    MEAN_FPPG_CHANGE=g.FPPG_CHANGE.mean(),
                                    PCT_GAIN5=(g.MIN_CHANGE>=5).mean()*100,
                                    PCT_LOSS5=(g.MIN_CHANGE<=-5).mean()*100,
                                    AGE_MODEL_MAE=g.ABS_AGE_MODEL_ERROR.mean(),
                                    AGE_MODEL_BIAS=g.AGE_MODEL_ERROR.mean()))
    return pd.DataFrame(results)

def main():
    p=generate(load_seasons()); s=summarize(p)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'step7_role_changes.csv',index=False)
    s.to_csv(OUT/'step7_role_segments.csv',index=False)
    print('Step 7: role-change diagnosis (descriptive, NOT an out-of-sample new forecast)')
    print('Matched returning observations:',len(p))
    print('\nPreseason-known cohorts, combined seasons:')
    print(s[(s.TARGET_YEAR=='ALL') & s.SEGMENT.isin(['PRIOR_MIN_GROUP','PRIOR_TREND_GROUP','PRIOR_GP_GROUP'])][['SEGMENT','GROUP','N','MEAN_MIN_CHANGE','PCT_GAIN5','PCT_LOSS5','AGE_MODEL_MAE']].round(2).to_string(index=False))
    print('\nLargest FPPG prediction misses:')
    print(p.nlargest(10,'ABS_AGE_MODEL_ERROR')[['TARGET_YEAR','PLAYER_NAME','PRIOR_MIN','ACTUAL_MIN','MIN_CONTRIBUTION','RATE_CONTRIBUTION','AGE_MODEL_ERROR']].round(2).to_string(index=False))
    print('\nReports saved to',OUT)

if __name__=='__main__': main()
