"""Leakage-safe, pre-season fantasy projection baselines (2024-25 and 2025-26).

Run from project root: python src/backtest_projections.py
Writes ONLY data/research/ outputs; does not modify production models.
Requires pandas and numpy, already used by the project.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'fantasy_player_seasons.csv'
OUT = ROOT / 'data' / 'research'
TARGET_YEARS = (2024, 2025)


def load_seasons(path=SOURCE):
    d = pd.read_csv(path)
    required = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON_START_YEAR', 'GP', 'MIN',
                'FGM', 'FGA', 'FTM', 'FTA', 'FG3M', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if d.duplicated(['PLAYER_ID', 'SEASON_START_YEAR']).any():
        raise ValueError('Expected one consolidated row per PLAYER_ID and season')
    for c in required[2:]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    # Use per-game NBA box-score columns, not NBA_FANTASY_PTS (different scoring rules).
    d['FPPG'] = (2*d.FGM-d.FGA+d.FTM-d.FTA+d.FG3M+d.REB+2*d.AST
                 +4*d.STL+4*d.BLK-2*d.TOV+d.PTS)
    d['GP'] = d.GP.clip(lower=0, upper=82)
    return d


def generate_predictions(d):
    rows = []
    for year in TARGET_YEARS:
        actual = d[d.SEASON_START_YEAR.eq(year)].copy()
        history = d[d.SEASON_START_YEAR.lt(year)].copy()
        # Prior seasons only. Players with no prior NBA season are not forecastable
        # by these baselines; report their exclusion rather than using target stats.
        history = history.sort_values(['PLAYER_ID', 'SEASON_START_YEAR'])
        prior = {pid: group for pid, group in history.groupby('PLAYER_ID')}
        for a in actual.itertuples(index=False):
            h = prior.get(a.PLAYER_ID)
            if h is None or h.empty:
                continue
            h = h.sort_values('SEASON_START_YEAR', ascending=False)
            recent = h.iloc[0]
            # Each method uses only the player's history BEFORE target year.
            # Missing intervening seasons remain absent; a stale season is still
            # eligible, but flagged through LAG_YEARS.
            record = dict(TARGET_YEAR=year, PLAYER_ID=a.PLAYER_ID,
                          PLAYER_NAME=a.PLAYER_NAME, ACTUAL_FPPG=a.FPPG,
                          ACTUAL_GP=a.GP, PRIOR_SEASONS=len(h),
                          LAG_YEARS=year-int(recent.SEASON_START_YEAR),
                          PRIOR_GP=recent.GP,
                          LAG1_FPPG=recent.FPPG,
                          LAG1_GP=recent.GP)
            for n in (2, 3):
                subset = h.head(n)
                # Recent seasons weighted 3:2:1, normalized if fewer are available.
                weights = np.arange(len(subset), 0, -1, dtype=float)
                record[f'WEIGHTED{n}_FPPG'] = np.average(subset.FPPG, weights=weights)
                record[f'WEIGHTED{n}_GP'] = np.average(subset.GP, weights=weights)
            rows.append(record)
    p = pd.DataFrame(rows)
    if p.empty:
        raise ValueError('No evaluable player-seasons')
    # Conservative shrinkage target: prior-year league-wide FPPG among players
    # with >=20 games. This is a population baseline, not a player future value.
    # GP weighting controls noisy small samples. No holdout outcomes used.
    for year in TARGET_YEARS:
        previous = d[d.SEASON_START_YEAR.eq(year-1)]
        stable = previous[previous.GP.ge(20) & previous.FPPG.notna()]
        population = float(stable.FPPG.median())
        mask = p.TARGET_YEAR.eq(year)
        strength = p.loc[mask, 'PRIOR_GP'] / (p.loc[mask, 'PRIOR_GP'] + 25)
        p.loc[mask, 'SHRUNK_FPPG'] = (strength*p.loc[mask, 'WEIGHTED3_FPPG']
                                      + (1-strength)*population)
    return p


def metrics(p):
    methods = ['LAG1_FPPG', 'WEIGHTED2_FPPG', 'WEIGHTED3_FPPG', 'SHRUNK_FPPG']
    output = []
    for year in [*TARGET_YEARS, 'ALL']:
        sample = p if year == 'ALL' else p[p.TARGET_YEAR.eq(year)]
        # Compare every method on identical player-seasons.
        sample = sample.dropna(subset=['ACTUAL_FPPG', *methods])
        for cohort, s in [('all_returning', sample),
                          ('prior_20plus_gp', sample[sample.PRIOR_GP.ge(20)]),
                          ('prior_40plus_gp', sample[sample.PRIOR_GP.ge(40)]),
                          ('prior_top100_fppg', sample.nlargest(100, 'LAG1_FPPG') if year != 'ALL'
                           else pd.concat([g.nlargest(100, 'LAG1_FPPG') for _,g in sample.groupby('TARGET_YEAR')]))]:
            for method in methods:
                errors = s[method]-s.ACTUAL_FPPG
                output.append(dict(TARGET_YEAR=year, COHORT=cohort, METHOD=method,
                                   N=len(s), MAE=errors.abs().mean(),
                                   RMSE=np.sqrt((errors**2).mean()),
                                   BIAS=errors.mean()))
    return pd.DataFrame(output)


def main():
    d = load_seasons()
    p = generate_predictions(d)
    m = metrics(p)
    coverage = []
    for year in TARGET_YEARS:
        actual = d[d.SEASON_START_YEAR.eq(year)]
        eligible = p[p.TARGET_YEAR.eq(year)]
        coverage.append(dict(TARGET_YEAR=year, ACTUAL_PLAYERS=len(actual),
                             RETURNING_EVALUABLE=len(eligible),
                             EXCLUDED_NO_PRIOR_HISTORY=len(actual)-len(eligible)))
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT/'baseline_predictions.csv', index=False)
    m.to_csv(OUT/'baseline_metrics.csv', index=False)
    pd.DataFrame(coverage).to_csv(OUT/'baseline_coverage.csv', index=False)
    print('Coverage:')
    print(pd.DataFrame(coverage).to_string(index=False))
    print('\nHoldout comparison — all returning players:')
    print(m[m.COHORT.eq('all_returning')][['TARGET_YEAR','METHOD','N','MAE','RMSE','BIAS']].round(3).to_string(index=False))
    print(f'\nSaved outputs to {OUT}')


if __name__ == '__main__':
    main()
