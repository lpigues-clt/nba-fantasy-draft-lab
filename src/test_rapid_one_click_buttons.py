from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / 'app.py').read_text()


def test_fast_buttons_both_ownership_options():
    assert 'key=f"lab_fast_mine_{quick_name}"' in APP
    assert 'key=f"lab_fast_opp_{quick_name}"' in APP
    assert 'run_action(write_pick, quick_name, "mine")' in APP
    assert 'run_action(write_pick, quick_name, "opponent")' in APP


def test_fast_buttons_use_available_ranked_pool():
    assert 'quick_rows = ranked_available.iloc[page_start:page_start + quick_button_count]' in APP
    assert 'ranked_available = available.sort_values(' in APP
    assert 'st.dataframe(board,' in APP
    assert 'st.selectbox("Players to display"' in APP
