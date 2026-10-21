"""Research-only, leakage-safe minutes x fantasy points/minute backtest.
Run: python src/backtest_minutes_rate.py
Outputs only data/research/step3_*.csv; no production ranking changes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons, TARGET_YEARS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'
METHODS = ('LAG1_FPPG', 'MIN_RATE_LAG1', 'MIN_RATE_BLEND', 'MIN_RATE_TREND', 'MIN_RATE_AGE')


def generate(d):
    rows = []
    for year in TARGET_YEARS:
        historical = d[d.SEASON_START_YEAR.lt(year)].sort_values('SEASON_START_YEAR')
        by_player = {pid: g.sort_values('SEASON_START_YEAR', ascending=False)
                     for pid, g in historical.groupby('PLAYER_ID')}
        for a in d[d.SEASON_START_YEAR.eq(year)].itertuples(index=False):
            h = by_player.get(a.PLAYER_ID)
            if h is None or h.empty:
                continue
            r = h.iloc[0]
            if not np.isfinite(r.MIN) or r.MIN <= 0 or not np.isfinite(r.FPPG):
                continue
            rate = r.FPPG / r.MIN
            lag = float(r.FPPG)
            # The lag1 decomposition is mathematically identical to lag1 FPPG.
            record = dict(TARGET_YEAR=year, PLAYER_ID=a.PLAYER_ID,
                          PLAYER_NAME=a.PLAYER_NAME, ACTUAL_FPPG=a.FPPG,
                          PRIOR_GP=r.GP, PRIOR_MIN=r.MIN, PRIOR_AGE=r.AGE,
                          LAG_YEARS=year-int(r.SEASON_START_YEAR),
                          LAG1_FPPG=lag, MIN_RATE_LAG1=r.MIN*rate)
            if len(h) >= 2:
                old = h.iloc[1]
                if np.isfinite(old.MIN) and old.MIN > 0 and np.isfinite(old.FPPG):
                    # Mild mean reversion in minutes, with per-minute production
                    # weighted toward the latest year. Both use only prior seasons.
                    projected_min = .8*r.MIN + .2*old.MIN
                    projected_rate = .8*rate + .2*(old.FPPG/old.MIN)
                    record['MIN_RATE_BLEND'] = projected_min*projected_rate
                    # Capped minutes trend; recent per-minute productivity unchanged.
                    projected_trend_min = np.clip(r.MIN + .25*(r.MIN-old.MIN), 0, 38)
                    record['MIN_RATE_TREND'] = projected_trend_min*rate
                else:
                    record['MIN_RATE_BLEND'] = lag
                    record['MIN_RATE_TREND'] = lag
            else:
                record['MIN_RATE_BLEND'] = lag
                record['MIN_RATE_TREND'] = lag
            # Explicit, small age-based hypothesis, not trained on target outcomes.
            # Prior AGE is measured at the prior season; never use target-year age.
            age = r.AGE
            factor = (1.02 if pd.notna(age) and age < 24 else
                      .98 if pd.notna(age) and age >= 33 else 1.0)
            record['MIN_RATE_AGE'] = record['MIN_RATE_BLEND'] * factor
            rows.append(record)
    return pd.DataFrame(rows)


def evaluate(p):
    out = []
    for year in (*TARGET_YEARS, 'ALL'):
        subset = p if year == 'ALL' else p[p.TARGET_YEAR.eq(year)]
        for cohort, group in [('all_returning', subset),
                              ('prior_20plus_gp', subset[subset.PRIOR_GP.ge(20)]),
                              ('prior_top100_fppg', pd.concat([g.nlargest(100, 'LAG1_FPPG')
                                  for _, g in subset.groupby('TARGET_YEAR')]))]:
            # Matched sample across all methods, for fair comparison.
            s = group.dropna(subset=['ACTUAL_FPPG', *METHODS])
            for method in METHODS:
                err = s[method] - s.ACTUAL_FPPG
                out.append(dict(TARGET_YEAR=year, COHORT=cohort, METHOD=method,
                                N=len(s), MAE=err.abs().mean(),
                                RMSE=np.sqrt((err**2).mean()), BIAS=err.mean()))
    return pd.DataFrame(out)


def main():
    d = load_seasons()
    p = generate(d)
    m = evaluate(p)
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT/'step3_predictions.csv', index=False)
    m.to_csv(OUT/'step3_metrics.csv', index=False)
    print('Minutes x rate hypotheses (lower MAE is better):')
    print(m[m.COHORT.eq('all_returning')][['TARGET_YEAR','METHOD','N','MAE','RMSE','BIAS']].round(3).to_string(index=False))
    print('\nTop-100 prior FPPG cohort:')
    print(m[(m.COHORT.eq('prior_top100_fppg')) & m.TARGET_YEAR.eq('ALL')][['METHOD','N','MAE']].round(3).to_string(index=False))
    print(f'\nResearch reports saved to {OUT}')


if __name__ == '__main__':
    main()
