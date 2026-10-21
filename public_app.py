"""Read-only public analytics; intentionally does not import private draft modules."""
from pathlib import Path
import json
import sqlite3
import unicodedata
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data/live/nba_2026_27.sqlite3'
PROJECTIONS = ROOT / 'data/processed/fantasy_value_model.csv'
st.set_page_config(page_title='NBA Fantasy Analytics 2026–27', page_icon='🏀', layout='wide')
st.title('NBA Fantasy Analytics · 2026–27')
st.caption('Public read-only analysis · Live box scores are provisional')

def norm(value):
    value = unicodedata.normalize('NFKD', str(value))
    return ''.join(c for c in value if not unicodedata.combining(c)).casefold().strip()

@st.cache_data(ttl=300)
def projections():
    d = pd.read_csv(PROJECTIONS, usecols=['PLAYER_NAME','TEAM_ABBREVIATION','FANTASY_PTS_PER_GAME'])
    d = d.drop_duplicates('PLAYER_NAME')
    d['key'] = d.PLAYER_NAME.map(norm)
    return d[~d.key.duplicated(keep=False)]

@st.cache_data(ttl=60)
def results():
    if not DB.is_file():
        return pd.DataFrame(), None
    try:
        with sqlite3.connect(f'file:{DB}?mode=ro', uri=True) as con:
            d = pd.read_sql_query('SELECT player_id,player_name,game_id,is_final,game_status,stats_json,fantasy_points,updated_utc FROM player_games WHERE game_status IN (2,3)', con)
    except sqlite3.Error as error:
        st.warning(f'Database unavailable: {error}')
        return pd.DataFrame(), None
    if d.empty:
        return d, None
    def played(s):
        m = str(json.loads(s).get('minutes') or '')
        return bool(m) and m not in ('00:00','0:00','PT00M00.00S','PT0M0S')
    d = d[d.stats_json.map(played)].copy()
    return d, d.updated_utc.max() if len(d) else None

proj = projections()
d, freshness = results()
if freshness:
    st.caption(f'Latest record updated: {freshness} (UTC)')
else:
    st.info('No imported live box scores yet. Showing preseason projections.')
if d.empty:
    st.dataframe(proj.drop(columns='key').sort_values('FANTASY_PTS_PER_GAME', ascending=False),hide_index=True,use_container_width=True)
else:
    final = d[d.is_final == 1].groupby('player_id').agg(
        final_gp=('game_id','nunique'), actual_fppg=('fantasy_points','mean'), season_fp=('fantasy_points','sum'))
    live = d[d.is_final == 0].groupby('player_id').agg(
        live_gp=('game_id','nunique'), provisional_fp=('fantasy_points','sum'))
    names = d.sort_values('updated_utc').drop_duplicates('player_id',keep='last').set_index('player_id')['player_name']
    board = final.join(live,how='outer').join(names).reset_index()
    board['key'] = board.player_name.map(norm)
    board = board.merge(proj,on='key',how='left',validate='many_to_one')
    board[['final_gp','live_gp']] = board[['final_gp','live_gp']].fillna(0).astype(int)
    mode = st.radio('Statistics', ['Finalized games', 'In-game provisional'], horizontal=True)
    metric = 'actual_fppg' if mode == 'Finalized games' else 'provisional_fp'
    filtered = board[board.final_gp.gt(0) if mode == 'Finalized games' else board.live_gp.gt(0)]
    search = st.text_input('Search players')
    if search:
        filtered = filtered[filtered.player_name.str.contains(search,case=False,regex=False,na=False)]
    filtered = filtered.sort_values(metric,ascending=False)
    cols = ['player_name','final_gp','actual_fppg','season_fp','live_gp','provisional_fp','FANTASY_PTS_PER_GAME']
    st.dataframe(filtered[cols],hide_index=True,use_container_width=True)
    if len(filtered):
        fig = px.bar(filtered.head(25),x='player_name',y=metric,title=f'Top 25 · {mode}')
        fig.update_layout(xaxis_tickangle=-50)
        st.plotly_chart(fig,use_container_width=True,config={'scrollZoom':True})
    st.caption(f'{board.PLAYER_NAME.isna().sum()} players have no verified preseason name match. No approximate joins are applied.')
st.caption('Private roster and draft controls are not available in this application.')
