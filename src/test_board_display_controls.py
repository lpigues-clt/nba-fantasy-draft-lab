from pathlib import Path
APP = (Path(__file__).resolve().parents[1] / 'app.py').read_text()

def test_row_options_and_actual_slice():
    assert 'row_options = [5, 10, 20, 50, 100]' in APP
    assert 'displayed = ranked_available.head(row_count).copy()' in APP
    assert 'st.dataframe(board,' in APP

def test_preset_sort_callback():
    assert 'on_change=apply_selected_preset' in APP
    assert 'st.session_state["lab_draft_sort"] = "Preset Analysis Score"' in APP

def test_header_sorting_native_grid():
    assert 'st.dataframe(board,' in APP
    assert 'column_config={' in APP
