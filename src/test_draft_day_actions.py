import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
from draft_day_actions import owner_for_next_pick, replace_last_pick, pick_log_csv


def test_auto_owner_snake():
    assert owner_for_next_pick([], 10, 1) == 'mine'
    assert owner_for_next_pick([{'player':str(i)} for i in range(10)], 10, 1) == 'opponent'
    assert owner_for_next_pick([{'player':str(i)} for i in range(19)], 10, 1) == 'mine'
    assert owner_for_next_pick([], 10, None) is None


def test_correct_last_preserves_owner_and_other_picks():
    picks = [{'player':'A', 'owner':'mine'}, {'player':'B','owner':'opponent','timestamp_utc':'x'}]
    corrected = replace_last_pick(picks, 'C', {'A','B','C'})
    assert corrected == [{'player':'A','owner':'mine'}, {'player':'C','owner':'opponent','timestamp_utc':'x'}]
    assert picks[-1]['player'] == 'B'
    with pytest.raises(ValueError):
        replace_last_pick(picks, 'A', {'A','B','C'})
    with pytest.raises(ValueError):
        replace_last_pick([], 'A', {'A'})


def test_export_csv():
    csv = pick_log_csv([{'player':'A, Jr.', 'owner':'mine', 'timestamp_utc':'now'}], 10, 1)
    assert '"A, Jr."' in csv
    assert 'Expected owner' in csv
    assert '1,1,' in csv
