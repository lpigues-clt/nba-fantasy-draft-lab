from pathlib import Path

APP = (Path(__file__).resolve().parents[1] / 'app.py').read_text()


def test_one_click_actions_precede_collapsed_grid():
    assert APP.index('ONE-CLICK DRAFT · TOP AVAILABLE') < APP.index('Full sortable rankings · click to expand')
    assert 'if mine_col.button("Mine"' in APP
    assert 'if opp_col.button("Opponent"' in APP


def test_draft_command_and_search_undo():
    for item in ('draft-command', 'NEXT MINE #', 'lab_rapid_player', 'lab_rapid_undo', 'MY ROSTER ·'):
        assert item in APP


def test_preserve_analysis_trackpad_and_presets():
    assert 'config={"scrollZoom": True}' in APP
    assert 'calibrated_tiers' in APP
    assert 'best_overall_values' in APP
