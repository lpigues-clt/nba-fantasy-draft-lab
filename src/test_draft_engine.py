import unittest
import pandas as pd
from draft_engine import chart_score, make_tiers, scheduled_picks, validate_picks, validate_pick_order

class DraftEngineTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'PLAYER_NAME': [f'P{i:03}' for i in range(350)], 'X': range(350), 'Y': range(350), 'C': range(350), 'S': range(350)})
        self.axes = ('X','Y','C','S')
    def test_six_tiers_of_fifty(self):
        tiers = make_tiers(self.data, self.axes)
        self.assertEqual(tiers['TIER'].value_counts().sort_index().tolist(), [50]*6)
        self.assertEqual(tiers['TIER'].isna().sum(), 50)
    def test_stable_under_filter_and_draft(self):
        before = make_tiers(self.data, self.axes).set_index('PLAYER_NAME')
        after = make_tiers(self.data.assign(DRAFT_OWNER='opponent'), self.axes).set_index('PLAYER_NAME')
        pd.testing.assert_series_equal(before['CHART_SCORE'], after['CHART_SCORE'])
        pd.testing.assert_series_equal(before['TIER'], after['TIER'])
    def test_duplicate_axis_weights(self):
        scores = chart_score(self.data, ('X','X','X','X'))
        expected = self.data['X'].rank(pct=True)*100
        pd.testing.assert_series_equal(scores, expected.round(2), check_names=False)
    def test_missing_metric_not_promoted(self):
        data = self.data.copy()
        data.loc[349, 'C'] = float('nan')
        tiers = make_tiers(data, self.axes).set_index('PLAYER_NAME')
        self.assertTrue(pd.isna(tiers.loc['P349','TIER']))
        self.assertTrue(pd.isna(tiers.loc['P349','CHART_SCORE']))
    def test_preset_reorders(self):
        data = self.data.copy()
        data['Y'] = list(reversed(range(350)))
        left = make_tiers(data, ('X','X','X','X')).set_index('PLAYER_NAME')
        right = make_tiers(data, ('Y','Y','Y','Y')).set_index('PLAYER_NAME')
        self.assertTrue(pd.isna(left.loc['P000','TIER']))
        self.assertEqual(int(right.loc['P000','TIER']), 1)
    def test_snake(self):
        self.assertEqual(scheduled_picks(9,5)[:3], [5,14,23])
        self.assertEqual(scheduled_picks(10,5)[:3], [5,16,25])
        self.assertEqual(scheduled_picks(10,None), [])
    def test_exact_order_warnings(self):
        picks = [{'player':f'P{i:03}', 'owner':'mine' if i==4 else 'opponent'} for i in range(9)]
        validate_picks(picks, set(self.data.PLAYER_NAME))
        self.assertEqual(validate_pick_order(picks,9,5), [])
        picks[0]['owner']='mine'
        self.assertEqual(len(validate_pick_order(picks,9,5)), 1)
    def test_duplicate_pick_rejected(self):
        with self.assertRaises(ValueError):
            validate_picks([{'player':'P000','owner':'mine'}]*2, set(self.data.PLAYER_NAME))

if __name__ == '__main__':
    unittest.main(verbosity=2)
