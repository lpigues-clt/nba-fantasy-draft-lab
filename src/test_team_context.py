import pandas as pd
from team_context import apply_team_context, apply_availability_history
from best_overall_calibration import best_overall_values


def test_verified_team_overlay_keeps_historical_team():
    df = pd.DataFrame({'PLAYER_NAME':['Giannis Antetokounmpo','Unknown Player'], 'TEAM_ABBREVIATION':['MIL','AAA']})
    result = apply_team_context(df)
    assert result.TEAM_ABBREVIATION.tolist() == ['MIA','AAA']
    assert result.HISTORICAL_TEAM_2025.tolist() == ['MIL','AAA']


def test_multiyear_gp_penalizes_repeated_absences_without_named_exceptions():
    df = pd.DataFrame({'MODEL_DRAFT_FPPG':[50,50], 'FANTASY_PTS_PER_GAME':[50,50], 'GP':[70,70],
                       'GP_2025':[70,70], 'GP_2024':[20,70], 'GP_2023':[30,70]})
    out = best_overall_values(df)
    assert out.loc[0,'BEST_OVERALL_FP_EQUIV'] < out.loc[1,'BEST_OVERALL_FP_EQUIV']
    assert out.AVAILABILITY_DISCOUNT_PCT.max() <= 16


def test_no_health_adjustment_to_role_projection_column():
    df = pd.DataFrame({'MODEL_DRAFT_FPPG':[55], 'FANTASY_PTS_PER_GAME':[55], 'GP':[40]})
    best_overall_values(df)
    assert df.loc[0,'MODEL_DRAFT_FPPG'] == 55
