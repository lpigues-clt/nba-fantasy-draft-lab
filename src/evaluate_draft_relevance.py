"""Step 9: draft-relevance diagnostics; research-only, no website changes."""
from pathlib import Path
import numpy as np
import pandas as pd
from backtest_projections import load_seasons
from backtest_role_forecast import generate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'research'
METHODS = ('LAG1_FPPG','STEP3_AGE','ROLE_25','ROLE_50')

def evaluate(p):
    rows=[]
    for year, g in p.groupby('TARGET_YEAR'):
        g=g.dropna(subset=['ACTUAL_FPPG', *METHODS]).copy()
        for cohort_name, s in [('all_returning',g),('prior_top100',g.nlargest(100,'PRIOR_FPPG')),('prior_30plus_gp',g[g.PRIOR_GP.ge(30)]),('prior_under30_gp',g[g.PRIOR_GP.lt(30)])]:
            for method in METHODS:
                e=s[method]-s.ACTUAL_FPPG
                rows.append(dict(YEAR=year,COHORT=cohort_name,METHOD=method,N=len(s),MAE=e.abs().mean(),BIAS=e.mean(),RMSE=np.sqrt(np.mean(e**2))))
    return pd.DataFrame(rows)

def draft_metrics(p):
    rows=[]
    for year,g in p.groupby('TARGET_YEAR'):
        g=g.dropna(subset=['ACTUAL_FPPG',*METHODS]).copy()
        for pool_name,pool in [('returning_pool',g),('prior_top100',g.nlargest(100,'PRIOR_FPPG'))]:
            for k in (13,30,50):
                if len(pool)<k: continue
                actual_top=set(pool.nlargest(k,'ACTUAL_FPPG').PLAYER_ID)
                oracle=pool.nlargest(k,'ACTUAL_FPPG').ACTUAL_FPPG.sum()
                for method in METHODS:
                    chosen=pool.nlargest(k,method)
                    realized=chosen.ACTUAL_FPPG.sum()
                    rows.append(dict(YEAR=year,POOL=pool_name,K=k,METHOD=method,TOP_K_OVERLAP=len(set(chosen.PLAYER_ID)&actual_top),REALIZED_FPPG_SUM=realized,ORACLE_FPPG_SUM=oracle,REGRET_FPPG_SUM=oracle-realized))
    return pd.DataFrame(rows)

def paired_uncertainty(p, seed=2026, draws=3000):
    """Exploratory paired bootstrap; model selection already saw these holdouts."""
    rng=np.random.default_rng(seed)
    rows=[]
    for name,g in [('all',p),('prior_top100',pd.concat([x.nlargest(100,'PRIOR_FPPG') for _,x in p.groupby('TARGET_YEAR')]))]:
        g=g.dropna(subset=['ACTUAL_FPPG',*METHODS])
        base=(g.STEP3_AGE-g.ACTUAL_FPPG).abs().to_numpy()
        role=(g.ROLE_50-g.ACTUAL_FPPG).abs().to_numpy()
        diff=base-role
        ix=rng.integers(0,len(diff),size=(draws,len(diff)))
        samples=diff[ix].mean(axis=1)
        rows.append(dict(COHORT=name,N=len(diff),MEAN_MAE_GAIN=diff.mean(),BOOTSTRAP_LO=np.quantile(samples,.025),BOOTSTRAP_HI=np.quantile(samples,.975),PCT_PLAYER_SEASONS_IMPROVED=100*(diff>0).mean()))
    return pd.DataFrame(rows)

def main():
    p=generate(load_seasons())
    m=evaluate(p); d=draft_metrics(p); u=paired_uncertainty(p)
    OUT.mkdir(parents=True,exist_ok=True)
    m.to_csv(OUT/'step9_cohort_metrics.csv',index=False)
    d.to_csv(OUT/'step9_draft_proxy.csv',index=False)
    u.to_csv(OUT/'step9_uncertainty.csv',index=False)
    print('Step 9 — cohort accuracy:')
    print(m[m.COHORT.eq('all_returning')][['YEAR','METHOD','N','MAE','BIAS']].round(3).to_string(index=False))
    print('\nDraft proxy — top 13 from returning-player pool (NOT a snake draft simulation):')
    print(d[(d.POOL.eq('returning_pool')) & d.K.eq(13)][['YEAR','METHOD','TOP_K_OVERLAP','REALIZED_FPPG_SUM','REGRET_FPPG_SUM']].round(2).to_string(index=False))
    print('\nPaired uncertainty (exploratory; previously inspected holdouts):')
    print(u.round(3).to_string(index=False))
    print('\nReports saved to',OUT)

if __name__=='__main__': main()
