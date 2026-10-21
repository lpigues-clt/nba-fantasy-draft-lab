"""Step 10: reproducible, research-only 2026-27 role-aware FPPG forecasts.
Run from project root: python src/build_experimental_projections.py
Never overwrites the existing draft-value or competitive-edge models.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons
from backtest_role_forecast import prior_features, transitions, role_adjustment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research' / 'experimental_projections_2026_27.csv'
TARGET_YEAR = 2026


def generate(d, target_year=TARGET_YEAR):
    if (d.SEASON_START_YEAR >= target_year).any():
        # Future data is never used, even if supplied by mistake.
        d = d[d.SEASON_START_YEAR < target_year].copy()
    f = prior_features(d, target_year)
    train = transitions(d, target_year)
    if f.empty or train.empty:
        raise ValueError('Insufficient historical seasons for projections')
    history = d.sort_values('SEASON_START_YEAR', ascending=False)
    latest = history.drop_duplicates('PLAYER_ID')[['PLAYER_ID','PLAYER_NAME']]
    older = history.groupby('PLAYER_ID').nth(1).reset_index()[['PLAYER_ID','MIN','FPPG']]
    older = older.rename(columns={'MIN':'OLDER_MIN','FPPG':'OLDER_FPPG'})
    p = f.merge(latest, on='PLAYER_ID', validate='one_to_one').merge(older, on='PLAYER_ID', how='left', validate='one_to_one')
    good = p.OLDER_MIN.gt(0) & np.isfinite(p.OLDER_FPPG)
    prior_rate = p.PRIOR_RATE
    old_rate = p.OLDER_FPPG / p.OLDER_MIN.replace(0, np.nan)
    blend_min = np.where(good, .8*p.PRIOR_MIN+.2*p.OLDER_MIN, p.PRIOR_MIN)
    blend_rate = np.where(good, .8*prior_rate+.2*old_rate, prior_rate)
    age_factor = np.where(p.PRIOR_AGE.lt(24), 1.02, np.where(p.PRIOR_AGE.ge(33), .98, 1.0))
    p['STEP3_AGE_FPPG'] = blend_min*blend_rate*age_factor
    p['EXPECTED_ROLE_DELTA_MPG'] = role_adjustment(train, p)
    p['EXPERIMENTAL_ROLE50_FPPG'] = p.STEP3_AGE_FPPG + .5*p.EXPECTED_ROLE_DELTA_MPG*p.PRIOR_RATE
    p['EXPERIMENTAL_ROLE50_FPPG'] = p.EXPERIMENTAL_ROLE50_FPPG.clip(lower=0)
    p['TRAIN_MAX_YEAR'] = int(train.OUTCOME_YEAR.max())
    p['TARGET_YEAR'] = target_year
    return p[['PLAYER_ID','PLAYER_NAME','TARGET_YEAR','TRAIN_MAX_YEAR','PRIOR_GP','PRIOR_MIN',
              'LAG_YEARS','STEP3_AGE_FPPG','EXPECTED_ROLE_DELTA_MPG','EXPERIMENTAL_ROLE50_FPPG']]


def main():
    p = generate(load_seasons())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT, index=False)
    print(f'Saved {len(p)} experimental forecasts to {OUT}')
    print('Historical training outcomes end in:', int(p.TRAIN_MAX_YEAR.max()))
    print('This does NOT change production Draft Value rankings.')

if __name__ == '__main__':
    main()
