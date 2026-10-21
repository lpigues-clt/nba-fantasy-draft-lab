"""Step 8: chronological role-aware forecasting experiment; research outputs only.
Run from project root: python src/backtest_role_forecast.py
Historical transitions used for training strictly precede each evaluation year.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons, TARGET_YEARS
from backtest_minutes_rate import generate as step3_generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'
METHODS = ('LAG1_FPPG', 'STEP3_AGE', 'ROLE_25', 'ROLE_50')


def prior_features(history, year):
    """One row per player with prior-season-only inputs at prediction cutoff."""
    rows = []
    for pid, g in history[history.SEASON_START_YEAR.lt(year)].groupby('PLAYER_ID'):
        h = g.sort_values('SEASON_START_YEAR', ascending=False)
        r = h.iloc[0]
        if not (np.isfinite(r.MIN) and r.MIN > 0 and np.isfinite(r.FPPG)):
            continue
        older = h.iloc[1] if len(h) > 1 else None
        rows.append(dict(PLAYER_ID=pid, PRIOR_MIN=float(r.MIN), PRIOR_GP=float(r.GP),
                         PRIOR_AGE=float(r.AGE) if pd.notna(r.AGE) else np.nan,
                         PRIOR_RATE=float(r.FPPG / r.MIN),
                         PRIOR_FPPG=float(r.FPPG),
                         PRIOR_TREND=float(r.MIN - older.MIN) if older is not None and pd.notna(older.MIN) else np.nan,
                         LAG_YEARS=year-int(r.SEASON_START_YEAR)))
    return pd.DataFrame(rows)


def features_to_groups(f):
    result = f.copy()
    result['MIN_BIN'] = pd.cut(result.PRIOR_MIN, [-np.inf, 15, 25, np.inf], labels=['low','medium','high']).astype(str)
    result['GP_BIN'] = pd.cut(result.PRIOR_GP, [-np.inf, 29, 59, np.inf], labels=['low','medium','high']).astype(str)
    return result


def transitions(d, year):
    """Training examples with known outcomes from years strictly before `year`."""
    parts = []
    for outcome_year in sorted(int(x) for x in d.SEASON_START_YEAR.unique() if x < year):
        if outcome_year <= int(d.SEASON_START_YEAR.min()):
            continue
        f = prior_features(d, outcome_year)
        if f.empty:
            continue
        actual = d[d.SEASON_START_YEAR.eq(outcome_year)][['PLAYER_ID','MIN']].rename(columns={'MIN':'OUTCOME_MIN'})
        t = f.merge(actual, on='PLAYER_ID', how='inner', validate='one_to_one')
        t['MIN_CHANGE'] = t.OUTCOME_MIN-t.PRIOR_MIN
        t['OUTCOME_YEAR'] = outcome_year
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def role_adjustment(train, future):
    """Conservative empirical-Bayes mean minutes change for prior-MPG cohorts.
    The prior is zero change; 40 pseudo-observations temper small cohorts.
    """
    tr = features_to_groups(train)
    fu = features_to_groups(future)
    # Simple groups deliberately avoid learning a complex model from two seasons.
    stats = tr.groupby('MIN_BIN')['MIN_CHANGE'].agg(['sum','count'])
    lookup = (stats['sum'] / (stats['count']+40)).to_dict()
    return fu.MIN_BIN.map(lookup).fillna(0).astype(float).to_numpy()


def generate(d):
    step3 = step3_generate(d)[['TARGET_YEAR','PLAYER_ID','MIN_RATE_AGE','LAG1_FPPG']]
    rows = []
    for year in TARGET_YEARS:
        train = transitions(d, year)
        if train.empty:
            raise ValueError('No historical role transitions for '+str(year))
        f = prior_features(d, year)
        actual = d[d.SEASON_START_YEAR.eq(year)][['PLAYER_ID','PLAYER_NAME','FPPG']]
        sample = f.merge(actual, on='PLAYER_ID', how='inner', validate='one_to_one')
        sample['TARGET_YEAR'] = year
        sample['TRAIN_MAX_YEAR'] = int(train.OUTCOME_YEAR.max())
        sample['TRAIN_N'] = len(train)
        sample['ROLE_DELTA_MIN'] = role_adjustment(train, sample)
        sample = sample.merge(step3, on=['TARGET_YEAR','PLAYER_ID'], how='inner', validate='one_to_one')
        sample = sample.rename(columns={'FPPG':'ACTUAL_FPPG','MIN_RATE_AGE':'STEP3_AGE'})
        sample['LAG1_FPPG'] = sample['LAG1_FPPG']
        # Add role correction to the already age-adjusted Step 3 baseline.
        for strength in (25, 50):
            sample[f'ROLE_{strength}'] = sample.STEP3_AGE + (strength/100)*sample.ROLE_DELTA_MIN*sample.PRIOR_RATE
        rows.append(sample)
    return pd.concat(rows, ignore_index=True)


def evaluate(p):
    rows=[]
    for year in (*TARGET_YEARS, 'ALL'):
        part = p if year=='ALL' else p[p.TARGET_YEAR.eq(year)]
        groups = {'all_returning':part,
                  'prior_under30_gp':part[part.PRIOR_GP.lt(30)],
                  'prior_top100_fppg':pd.concat([g.nlargest(100,'PRIOR_FPPG') for _,g in part.groupby('TARGET_YEAR')])}
        for cohort, group in groups.items():
            g=group.dropna(subset=['ACTUAL_FPPG',*METHODS])
            for method in METHODS:
                error=g[method]-g.ACTUAL_FPPG
                rows.append(dict(TARGET_YEAR=year,COHORT=cohort,METHOD=method,N=len(g),
                                 MAE=error.abs().mean(),RMSE=np.sqrt((error**2).mean()),BIAS=error.mean()))
    return pd.DataFrame(rows)


def main():
    p=generate(load_seasons()); m=evaluate(p)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'step8_role_predictions.csv',index=False)
    m.to_csv(OUT/'step8_role_metrics.csv',index=False)
    print('Step 8: role-aware forecast, trained only on earlier-season transitions')
    print('Training chronology:')
    print(p.groupby('TARGET_YEAR')[['TRAIN_MAX_YEAR','TRAIN_N']].first().to_string())
    print('\nAll returning players (lower MAE better):')
    print(m[m.COHORT.eq('all_returning')][['TARGET_YEAR','METHOD','N','MAE','BIAS']].round(3).to_string(index=False))
    print('\nTop 100 prior-season FPPG cohort:')
    print(m[(m.COHORT.eq('prior_top100_fppg')) & m.TARGET_YEAR.eq('ALL')][['METHOD','N','MAE']].round(3).to_string(index=False))
    print('\nSaved research reports to',OUT)

if __name__=='__main__': main()
