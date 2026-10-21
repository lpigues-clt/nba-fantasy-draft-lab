"""Step 5 research-only minutes forecasts. Run: python src/backtest_minutes_forecast.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons, TARGET_YEARS
from backtest_minutes_rate import generate as step3_generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'
METHODS = ('CARRY_MIN', 'BLEND_MIN', 'AGE_MIN', 'SAMPLE_MIN', 'AGE_SAMPLE_MIN')


def generate(d):
    baseline = step3_generate(d).set_index(['TARGET_YEAR','PLAYER_ID'])
    rows = []
    for year in TARGET_YEARS:
        past = d[d.SEASON_START_YEAR.lt(year)]
        groups = {pid:g.sort_values('SEASON_START_YEAR',ascending=False) for pid,g in past.groupby('PLAYER_ID')}
        for a in d[d.SEASON_START_YEAR.eq(year)].itertuples(index=False):
            h=groups.get(a.PLAYER_ID)
            if h is None or h.empty or (year,a.PLAYER_ID) not in baseline.index: continue
            r=h.iloc[0]
            if not np.isfinite(r.MIN) or r.MIN<=0 or not np.isfinite(r.FPPG): continue
            old=h.iloc[1] if len(h)>1 else None
            old_min=float(old.MIN) if old is not None and np.isfinite(old.MIN) and old.MIN>0 else float(r.MIN)
            carry=float(r.MIN)
            blend=.8*carry+.2*old_min
            age=float(r.AGE) if pd.notna(r.AGE) else np.nan
            age_factor=1.02 if age<24 else .98 if age>=33 else 1.
            # Prior-GP-sensitive blending; explicitly hypothetical, not optimized.
            gp=float(r.GP) if np.isfinite(r.GP) else 0.
            w=gp/(gp+20.)
            sample=w*carry+(1-w)*old_min
            record={'TARGET_YEAR':year,'PLAYER_ID':a.PLAYER_ID,'PLAYER_NAME':a.PLAYER_NAME,
                    'ACTUAL_MIN':a.MIN,'ACTUAL_FPPG':a.FPPG,'PRIOR_MIN':carry,'PRIOR_GP':gp,
                    'PRIOR_AGE':age,'PRIOR_FPPG':float(r.FPPG),'PRIOR_RATE':float(r.FPPG/r.MIN),
                    'CARRY_MIN':carry,'BLEND_MIN':blend,'AGE_MIN':blend*age_factor,
                    'SAMPLE_MIN':sample,'AGE_SAMPLE_MIN':sample*age_factor,
                    'STEP3_AGE_FPPG':float(baseline.loc[(year,a.PLAYER_ID),'MIN_RATE_AGE'])}
            for method in METHODS:
                record[method+'_FPPG']=record[method]*record['PRIOR_RATE']
            rows.append(record)
    return pd.DataFrame(rows)


def evaluate(p):
    rows=[]
    for year in (*TARGET_YEARS,'ALL'):
        s=p if year=='ALL' else p[p.TARGET_YEAR.eq(year)]
        cohorts={'all_returning':s,'prior_under30_gp':s[s.PRIOR_GP.lt(30)],
                 'prior_60plus_gp':s[s.PRIOR_GP.ge(60)],
                 'prior_top100_fppg':pd.concat([g.nlargest(100,'PRIOR_FPPG') for _,g in s.groupby('TARGET_YEAR')])}
        for name,g in cohorts.items():
            for method in METHODS:
                for target,suffix in [('ACTUAL_MIN',''),('ACTUAL_FPPG','_FPPG')]:
                    t=g.dropna(subset=[target,method+suffix]); e=t[method+suffix]-t[target]
                    rows.append({'TARGET_YEAR':year,'COHORT':name,'TARGET':target,'METHOD':method,
                                 'N':len(t),'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),'BIAS':e.mean()})
            if name in cohorts:
                t=g.dropna(subset=['ACTUAL_FPPG','STEP3_AGE_FPPG'])
                e=t.STEP3_AGE_FPPG-t.ACTUAL_FPPG
                rows.append({'TARGET_YEAR':year,'COHORT':name,'TARGET':'ACTUAL_FPPG','METHOD':'STEP3_AGE',
                             'N':len(t),'MAE':e.abs().mean(),'RMSE':np.sqrt((e**2).mean()),'BIAS':e.mean()})
    return pd.DataFrame(rows)


def main():
    p=generate(load_seasons()); m=evaluate(p)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'step5_minutes_predictions.csv',index=False)
    m.to_csv(OUT/'step5_minutes_metrics.csv',index=False)
    for target in ['ACTUAL_MIN','ACTUAL_FPPG']:
        print('\n'+target+' (lower MAE better)')
        print(m[m.TARGET.eq(target)&m.COHORT.eq('all_returning')][['TARGET_YEAR','METHOD','N','MAE','BIAS']].round(3).to_string(index=False))
    print('\nSaved research reports to',OUT)

if __name__=='__main__': main()
