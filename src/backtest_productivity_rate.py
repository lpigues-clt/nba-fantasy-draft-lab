"""Step 6: leakage-safe per-minute productivity hypotheses. Research outputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons, TARGET_YEARS
from backtest_minutes_rate import generate as step3_generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'
METHODS = ('CARRY_RATE', 'BLEND_RATE', 'AGE_RATE', 'SAMPLE_RATE', 'AGE_SAMPLE_RATE')


def generate(d):
    step3 = step3_generate(d).set_index(['TARGET_YEAR', 'PLAYER_ID'])
    rows = []
    for year in TARGET_YEARS:
        prior = d[d.SEASON_START_YEAR.lt(year)]
        groups = {pid: g.sort_values('SEASON_START_YEAR', ascending=False) for pid, g in prior.groupby('PLAYER_ID')}
        for a in d[d.SEASON_START_YEAR.eq(year)].itertuples(index=False):
            h = groups.get(a.PLAYER_ID)
            if h is None or h.empty or (year, a.PLAYER_ID) not in step3.index:
                continue
            r = h.iloc[0]
            if not np.isfinite(r.MIN) or r.MIN <= 0 or not np.isfinite(r.FPPG):
                continue
            rate = float(r.FPPG / r.MIN)
            older = h.iloc[1] if len(h) > 1 else None
            old_rate = float(older.FPPG / older.MIN) if older is not None and np.isfinite(older.MIN) and older.MIN > 0 and np.isfinite(older.FPPG) else rate
            blend = .8 * rate + .2 * old_rate
            age = float(r.AGE) if pd.notna(r.AGE) else np.nan
            factor = 1.02 if age < 24 else .98 if age >= 33 else 1.
            gp = float(r.GP) if np.isfinite(r.GP) else 0.
            sample = gp / (gp + 20.) * rate + 20. / (gp + 20.) * old_rate
            rec = dict(TARGET_YEAR=year, PLAYER_ID=a.PLAYER_ID, PLAYER_NAME=a.PLAYER_NAME,
                       ACTUAL_MIN=a.MIN, ACTUAL_FPPG=a.FPPG,
                       ACTUAL_RATE=float(a.FPPG/a.MIN) if np.isfinite(a.MIN) and a.MIN > 0 else np.nan,
                       PRIOR_MIN=float(r.MIN), PRIOR_GP=gp, PRIOR_AGE=age,
                       PRIOR_FPPG=float(r.FPPG), LAG_YEARS=year-int(r.SEASON_START_YEAR),
                       CARRY_RATE=rate, BLEND_RATE=blend, AGE_RATE=blend*factor,
                       SAMPLE_RATE=sample, AGE_SAMPLE_RATE=sample*factor,
                       STEP3_AGE_FPPG=float(step3.loc[(year,a.PLAYER_ID),'MIN_RATE_AGE']))
            # Keep minutes fixed to the Step 3 hypothesis to isolate rate changes.
            projected_min = (.8 * float(r.MIN) + .2 * float(older.MIN)) if older is not None and np.isfinite(older.MIN) and older.MIN > 0 else float(r.MIN)
            rec['PROJECTED_MIN'] = projected_min
            for method in METHODS:
                rec[method+'_FPPG'] = projected_min * rec[method]
            rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(p):
    out = []
    for year in (*TARGET_YEARS, 'ALL'):
        s = p if year == 'ALL' else p[p.TARGET_YEAR.eq(year)]
        cohorts = {'all_returning':s, 'prior_under30_gp':s[s.PRIOR_GP.lt(30)],
                   'prior_60plus_gp':s[s.PRIOR_GP.ge(60)],
                   'prior_top100_fppg':pd.concat([g.nlargest(100,'PRIOR_FPPG') for _,g in s.groupby('TARGET_YEAR')])}
        for cohort, g in cohorts.items():
            for target, suffix in [('ACTUAL_RATE',''), ('ACTUAL_FPPG','_FPPG')]:
                methods = METHODS if target == 'ACTUAL_RATE' else (*METHODS, 'STEP3_AGE')
                for method in methods:
                    column = 'STEP3_AGE_FPPG' if method == 'STEP3_AGE' else method+suffix
                    t = g.dropna(subset=[target,column]); err = t[column]-t[target]
                    out.append(dict(TARGET_YEAR=year, COHORT=cohort, TARGET=target, METHOD=method,
                                    N=len(t), MAE=err.abs().mean(), RMSE=np.sqrt((err**2).mean()), BIAS=err.mean()))
    return pd.DataFrame(out)


def main():
    p = generate(load_seasons()); m = evaluate(p)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'step6_rate_predictions.csv',index=False)
    m.to_csv(OUT/'step6_rate_metrics.csv',index=False)
    for target in ('ACTUAL_RATE','ACTUAL_FPPG'):
        print('\n'+target+' (lower MAE better)')
        print(m[m.TARGET.eq(target)&m.COHORT.eq('all_returning')][['TARGET_YEAR','METHOD','N','MAE','BIAS']].round(3).to_string(index=False))
    print('\nTop-100 FPPG cohort:')
    print(m[m.TARGET.eq('ACTUAL_FPPG')&m.COHORT.eq('prior_top100_fppg')&m.TARGET_YEAR.eq('ALL')][['METHOD','N','MAE']].round(3).to_string(index=False))
    print('\nSaved research reports to',OUT)

if __name__ == '__main__':
    main()
