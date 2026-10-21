import pandas as pd
from preset_calibration import PROFILES, calibrated_tiers


def test_seven_distinct_presets_and_custom_fallback():
    assert len(PROFILES) == 7
    assert len({tuple(sorted(p.items())) for p in PROFILES.values()}) == 7
    for weights in PROFILES.values():
        assert abs(sum(weights.values()) - 1) < 1e-10
    metrics = set().union(*(set(p) for p in PROFILES.values()))
    frame = pd.DataFrame({'PLAYER_NAME': ['A', 'B', 'C']})
    for i, col in enumerate(sorted(metrics)):
        frame[col] = [1 + i % 3, 2 + (i + 1) % 3, 3 + (i + 2) % 3]
    axes = tuple(sorted(metrics)[:4])
    for name in PROFILES:
        result = calibrated_tiers(frame, axes, name)
        assert result.CHART_SCORE.notna().all()
        assert result.TIER.notna().all()
    assert calibrated_tiers(frame, axes, 'Custom').CHART_SCORE.notna().all()


def test_playoff_preset_has_majority_schedule_weight():
    assert PROFILES['Playoff-Focused Drafting']['PLAYOFF_SCHEDULE_SCORE'] > .5
