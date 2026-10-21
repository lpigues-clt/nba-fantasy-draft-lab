"""Fantasy Draft Lab — public, session-isolated Streamlit website.

Place this file at the project root as app.py. Uses src/draft_engine.py
and the existing data/processed model outputs. No external API calls.
"""
from __future__ import annotations

import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from draft_engine import make_tiers, scheduled_picks, validate_pick_order, validate_picks
from preset_calibration import calibrated_tiers, PROFILES
from best_overall_calibration import best_overall_values
from team_context import apply_team_context, apply_availability_history
from draft_day_actions import owner_for_next_pick, replace_last_pick, pick_log_csv

DATA = ROOT / "data" / "processed"
STATE_FILE = DATA / "live_draft_state.json"
CONFIG_FILE = DATA / "draft_configuration.json"

PRESETS = {
    "Best Overall Draft Targets": ("FANTASY_PTS_PER_GAME", "DRAFT_VALUE_SCORE", "COMPETITIVE_EDGE_SCORE", "MINUTES_PER_GAME"),
    "High Upside vs Reliability": ("FANTASY_PTS_PER36", "FANTASY_PTS_PER_GAME", "AVAILABILITY_SCORE", "MINUTES_PER_GAME"),
    "Consistency vs Production": ("FANTASY_PTS_PER_GAME", "HISTORICAL_SCORE", "AVAILABILITY_SCORE", "GP"),
    "Efficiency vs Opportunity": ("EFFICIENCY_SCORE", "OPPORTUNITY_ROLE_SCORE", "FANTASY_PTS_PER_GAME", "MINUTES_PER_GAME"),
    "Availability and Durability": ("AVAILABILITY_SCORE", "DRAFT_VALUE_SCORE", "HISTORICAL_SCORE", "GP"),
    "Playoff-Focused Drafting": ("PLAYOFF_SCHEDULE_SCORE", "DRAFT_VALUE_SCORE", "COMPETITIVE_EDGE_SCORE", "FANTASY_PTS_PER_GAME"),
    "Find Undervalued Players": ("FANTASY_PTS_PER_GAME", "EDGE_ADJUSTED_DRAFT_SCORE", "COMPETITIVE_EDGE_SCORE", "GP"),
    "Role Expansion and Breakouts": ("MINUTES_PER_GAME", "FANTASY_PTS_PER36", "OPPORTUNITY_ROLE_SCORE", "FANTASY_PTS_PER_GAME"),
}
NOTES = {
    "Best Overall Draft Targets": "Forward-looking projected FP/G with a bounded, multi-season games-played risk adjustment; other composite metrics remain visible for independent review.",
    "High Upside vs Reliability": "Per-36 scoring and nightly production; per-minute output may not scale to a larger role.",
    "Consistency vs Production": "Nightly production, historical consistency, availability and games played.",
    "Efficiency vs Opportunity": "Efficiency and opportunity/role scores; neither predicts future coaching decisions.",
    "Availability and Durability": "Historical availability, Draft Value and consistency; not an injury forecast.",
    "Playoff-Focused Drafting": "Playoff schedule volume and Draft Value; team assignments are provisional.",
    "Find Undervalued Players": "Adjusted model value and production; a market bargain cannot be established without ADP.",
    "Role Expansion and Breakouts": "Minutes, per-36 production and opportunity; increased minutes are not guaranteed.",
    "Custom": "Custom four-metric weighted analysis.",
}
METRICS = {
    "Draft Value": "DRAFT_VALUE_SCORE",
    "Edge-Adjusted Draft Value": "EDGE_ADJUSTED_DRAFT_SCORE",
    "Competitive Edge": "COMPETITIVE_EDGE_SCORE",
    "Fantasy Points/Game": "FANTASY_PTS_PER_GAME",
    "Experimental 2026–27 Role FPPG": "EXPERIMENTAL_ROLE50_FPPG",
    "Fantasy Points/36": "FANTASY_PTS_PER36",
    "Production": "PRODUCTION_SCORE",
    "Historical Consistency": "HISTORICAL_SCORE",
    "Efficiency": "EFFICIENCY_SCORE",
    "Opportunity / Role": "OPPORTUNITY_ROLE_SCORE",
    "Availability": "AVAILABILITY_SCORE",
    "Playoff Schedule": "PLAYOFF_SCHEDULE_SCORE",
    "Minutes/Game": "MINUTES_PER_GAME",
    "Games Played": "GP",
}
TIERS = [
    ("#a54aff", "Elite Targets"),
    ("#218dff", "Strong Targets"),
    ("#00c9a7", "Solid Targets"),
    ("#f0b91e", "Mid-Round Value"),
    ("#ff7b35", "Later Options"),
    ("#ff496c", "Deep Value"),
]
BG = "#080b18"
PANEL = "#0e1427"
PURPLE = "#a64dff"

st.set_page_config(page_title="Fantasy Draft Lab · 2026", page_icon="🏀", layout="wide", initial_sidebar_state="collapsed")

CSS = """
<style>
:root { color-scheme: dark; }
html, body, [data-testid="stApp"], [data-testid="stMain"] {
  background: radial-gradient(ellipse at 54% -22%, #21103d 0%, #0a0d1d 45%, #060a15 100%) !important;
  color: #eae7ff;
}
[data-testid="stHeader"] {background: rgba(6,8,19,.85);}
[data-testid="stMainBlockContainer"] {max-width: 1920px; padding: .55rem 1.15rem 2rem;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
[data-testid="stVerticalBlockBorderWrapper"] {background: transparent;}
[data-testid="stVerticalBlock"] {gap:.7rem;}
[data-testid="stHorizontalBlock"] {align-items:stretch;}
[data-testid="stColumn"] {min-width:0;}
.stButton>button, .stDownloadButton>button {border-radius:9px; border:1px solid #5c318e;
 background:linear-gradient(115deg,#6412bf,#441093);color:#f9f3ff;font-weight:650;min-height:39px;}
.stButton>button:hover {border-color:#bf80ff;color:white;box-shadow:0 0 17px #7928d950;}
.stButton>button[kind="secondary"] {background:#171d36;border-color:#354064;color:#d8d5f3;}
.stButton>button[kind="secondary"]:hover {border-color:#a64dff;color:#fff;}
.stButton>button:disabled {opacity:.35;}
[data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {background:#11182e !important; color:#eae7ff !important;border-color:#35416b !important;}
[data-testid="stSelectbox"] [data-baseweb="select"] * {color:#eae7ff !important;}
[data-testid="stSlider"] [role="slider"] {background:#ac54ff;}
[data-testid="stMetric"] {background:#10172a;border:1px solid #2c3355;border-radius:10px;padding:12px;}
[data-testid="stMetricValue"] {color:#e9d9ff;}
[data-testid="stDataFrame"] {border:1px solid #35315e;border-radius:9px;overflow:hidden;}
[data-testid="stExpander"] {background:#0f1528;border:1px solid #302650;border-radius:10px;}
label, [data-testid="stWidgetLabel"] {color:#c6b8ea !important;}
h1,h2,h3 {color:#eee6ff !important;letter-spacing:.01em;}
p,li {color:#c6c3df;}
hr {border-color:#292448 !important;}
.dash-header {display:flex;align-items:center;justify-content:space-between;gap:16px;
 padding:6px 4px 17px;border-bottom:1px solid #382455;margin-bottom:13px;}
.brand {display:flex;align-items:center;gap:15px;}
.brand-icon {width:60px;height:60px;border-radius:50%;display:grid;place-items:center;
 border:2px solid #9c39ff;box-shadow:0 0 20px #7b2efc70;color:#c77bff;font-size:33px;}
.brand-title {font-weight:850;letter-spacing:.12em;font-size:24px;line-height:1.1;color:#f3eaff;}
.brand-title span {color:#b75bff;}
.brand-year {color:#ad8ce6;letter-spacing:.35em;font-size:13px;margin-top:5px;}
.header-side {font-size:12px;color:#b39dce;}
.glass {border:1px solid #3b2a65;background:linear-gradient(135deg,#11152b,#0b1223);
 border-radius:12px;padding:13px 14px;margin:0 0 10px;box-shadow:0 4px 20px #0003;}
.glass-purple {border-color:#60348e;background:linear-gradient(135deg,#21103c,#10132a 72%);}
.card-title {font-size:13px;font-weight:850;color:#c38aff;letter-spacing:.035em;margin-bottom:9px;}
.card-title-gold {color:#ffbd67;}
.mini {font-size:12px;color:#a7a9c7;line-height:1.5;}
.muted {color:#a8a4c5;}
.tier-grid {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:9px;}
.tier-card {border-radius:9px;padding:11px 9px;min-height:100px;border:1px solid var(--tc);
 background:linear-gradient(135deg,color-mix(in srgb,var(--tc) 20%,#0c1328),#0b1121 78%);}
.tier-number {display:inline-grid;place-items:center;border-radius:8px;width:28px;height:28px;
 background:var(--tc);color:white;font-size:16px;font-weight:900;margin-right:5px;}
.tier-title {font-size:11px;font-weight:750;color:#f2ebff;white-space:normal;}
.tier-range {font-size:12px;color:#c5c2dc;margin-top:10px;}
.tier-remain {font-size:11px;color:#b1accb;margin-top:3px;}
.kpi-row {display:flex;gap:10px;flex-wrap:wrap;}
.kpi {flex:1;min-width:85px;padding:11px;border-radius:9px;background:#10172c;border:1px solid #292e52;}
.kpi b {font-size:19px;color:#e4d4ff;display:block;}
.kpi span {font-size:11px;color:#b3afcb;}
.pick-row {display:flex;justify-content:space-between;gap:8px;padding:7px 2px;
 border-bottom:1px solid #242843;font-size:12px;color:#e0dafa;}
.pick-row:last-child {border-bottom:none;}
.pill {display:inline-block;border-radius:6px;padding:2px 6px;font-size:10px;
 background:#25123f;color:#d4a5ff;border:1px solid #7137aa;}
.player-table {width:100%;border-collapse:collapse;font-size:12px;color:#e8e7f8;}
.player-table th {text-align:left;background:#17172f;color:#cdb9ef;font-size:10px;
 text-transform:uppercase;letter-spacing:.03em;padding:10px 8px;position:sticky;top:0;}
.player-table td {padding:9px 8px;border-bottom:1px solid #242841;}
.player-table tr:nth-child(even) {background:#11182a;}
.player-table tr:hover {background:#20203b;}
.player-table .highlight {color:#d2a3ff;font-weight:800;}
.player-table .score {background:#351950;color:#ead2ff;text-align:center;font-weight:750;}
.scroll-table {max-height:620px;overflow:auto;border:1px solid #352750;border-radius:9px;}
.info-line {font-size:12px;color:#b9aed0;margin:4px 0 10px;}
.status-dot {display:inline-block;width:9px;height:9px;background:#27d6a1;border-radius:50%;margin-right:6px;}
.progress-track {background:#252043;height:11px;border-radius:10px;overflow:hidden;}
.progress-fill {height:100%;background:linear-gradient(90deg,#7d36ff,#c788ff);}
[data-testid="stRadio"] div[role="radiogroup"] {gap:8px;}
[data-testid="stRadio"] label {background:#12182d;padding:7px 12px;border:1px solid #302852;border-radius:9px;}
@media(max-width:1100px) {.tier-grid{grid-template-columns:repeat(3,minmax(0,1fr));}.brand-title{font-size:18px;}}
@media(max-width:700px) {.tier-grid{grid-template-columns:repeat(2,minmax(0,1fr));}.brand-icon{width:44px;height:44px;font-size:24px;}.header-side{display:none;}}
.draft-command {display:flex;gap:15px;align-items:center;flex-wrap:wrap;padding:10px 14px;margin:4px 0 10px;border:1px solid #7241b0;border-radius:10px;background:#17112c;color:#f0e4ff;font-size:13px;}
.draft-command b {white-space:nowrap;color:#e4c2ff;}
.draft-command span {margin-left:auto;color:#b9e8d6;font-size:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(value):
    return html.escape(str(value), quote=True)


def box(title, body, gold=False, purple=False):
    klass = "glass glass-purple" if purple else "glass"
    title_class = "card-title card-title-gold" if gold else "card-title"
    st.markdown(f'<div class="{klass}"><div class="{title_class}">{esc(title)}</div>{body}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_models():
    draft = pd.read_csv(DATA / "fantasy_value_model.csv")
    edge = pd.read_csv(DATA / "competitive_edge.csv")
    required_d = {"PLAYER_NAME", "DRAFT_VALUE_SCORE", "FANTASY_PTS_PER_GAME", "GP"}
    required_e = {"PLAYER_NAME", "COMPETITIVE_EDGE_SCORE", "EDGE_ADJUSTED_DRAFT_SCORE"}
    for label, frame, required in (("Draft Value", draft, required_d), ("Competitive Edge", edge, required_e)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        if frame["PLAYER_NAME"].isna().any() or frame["PLAYER_NAME"].duplicated().any():
            raise ValueError(f"{label} has missing or duplicate player names")
    merged = draft.merge(edge[list(required_e)], on="PLAYER_NAME", how="inner", validate="one_to_one")
    if len(merged) != len(draft) or len(merged) != len(edge):
        raise ValueError("Draft and Competitive Edge player lists do not match")
    for column in set(METRICS.values()) & set(merged.columns):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged[["DRAFT_VALUE_SCORE", "EDGE_ADJUSTED_DRAFT_SCORE"]].isna().any().any():
        raise ValueError("Draft Value contains missing scores")
    # Optional research overlay: never changes original Draft Value or Edge scores.
    research_file = ROOT / "data" / "research" / "experimental_projections_2026_27.csv"
    if research_file.exists():
        research = pd.read_csv(research_file)
        required = {"PLAYER_NAME", "EXPERIMENTAL_ROLE50_FPPG", "TARGET_YEAR"}
        if not required.issubset(research.columns):
            raise ValueError("Experimental projection file has missing columns")
        research = research[research["TARGET_YEAR"].eq(2026)].copy()
        if research["PLAYER_NAME"].duplicated().any():
            raise ValueError("Experimental projections contain duplicate player names")
        merged = merged.merge(research[["PLAYER_NAME", "EXPERIMENTAL_ROLE50_FPPG", "PRIOR_GP", "LAG_YEARS"]],
                              on="PLAYER_NAME", how="left", validate="one_to_one")
        merged["EXPERIMENTAL_ROLE50_FPPG"] = pd.to_numeric(
            merged["EXPERIMENTAL_ROLE50_FPPG"], errors="coerce")
    # Projection-led ranking: no arbitrary blend with last-season FPPG or legacy score.
    # A forecast is only eligible if the player has an immediately prior season.
    if "EXPERIMENTAL_ROLE50_FPPG" in merged:
        eligible = merged["LAG_YEARS"].eq(1) & merged["EXPERIMENTAL_ROLE50_FPPG"].notna()
        merged["MODEL_DRAFT_FPPG"] = merged["EXPERIMENTAL_ROLE50_FPPG"].where(eligible)
        merged["MODEL_DRAFT_RANK"] = merged["MODEL_DRAFT_FPPG"].rank(ascending=False, method="min")
    merged = apply_team_context(merged)
    merged = apply_availability_history(merged)
    merged["ORIGINAL_RANK"] = merged["DRAFT_VALUE_SCORE"].rank(ascending=False, method="min").astype(int)
    merged["ADJUSTED_RANK"] = merged["EDGE_ADJUSTED_DRAFT_SCORE"].rank(ascending=False, method="min").astype(int)
    return merged


def read_json(path, default):
    """Per-visitor draft data; no shared disk writes or cross-user records."""
    key = "public_draft_state" if path == STATE_FILE else "public_draft_config"
    if key not in st.session_state:
        st.session_state[key] = json.loads(json.dumps(default))
    return st.session_state[key]


def atomic_write(path, payload):
    key = "public_draft_state" if path == STATE_FILE else "public_draft_config"
    st.session_state[key] = json.loads(json.dumps(payload))


def write_pick(name, owner):
    current = read_json(STATE_FILE, {"version": 1, "picks": []})
    validate_picks(current["picks"], player_names)
    if name in {p["player"] for p in current["picks"]}:
        raise ValueError(f"Already drafted: {name}")
    current["picks"].append({"player": name, "owner": owner, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    validate_picks(current["picks"], player_names)
    atomic_write(STATE_FILE, current)


def undo_pick():
    current = read_json(STATE_FILE, {"version": 1, "picks": []})
    validate_picks(current["picks"], player_names)
    if not current["picks"]:
        return False
    current["picks"].pop()
    atomic_write(STATE_FILE, current)
    return True


def correct_last_pick(player):
    current = read_json(STATE_FILE, {"version": 1, "picks": []})
    validate_picks(current["picks"], player_names)
    current["picks"] = replace_last_pick(current["picks"], player, player_names)
    atomic_write(STATE_FILE, current)


def record_auto_pick(player):
    current = read_json(STATE_FILE, {"version": 1, "picks": []})
    validate_picks(current["picks"], player_names)
    draft_teams = int(st.session_state.get("lab_teams", config["teams"]))
    draft_slot = st.session_state.get("lab_slot", "Not announced")
    slot = int(draft_slot) if str(draft_slot).isdigit() else None
    owner = owner_for_next_pick(current["picks"], draft_teams, slot)
    if owner is None:
        raise ValueError("Set your draft position to use Auto Pick")
    write_pick(player, owner)


def reset_board():
    atomic_write(STATE_FILE, {"version": 1, "picks": []})
    atomic_write(CONFIG_FILE, {"teams": 10, "position": None})
    for key in list(st.session_state):
        if key.startswith("lab_"):
            del st.session_state[key]


def run_action(fn, *args):
    try:
        fn(*args)
        st.rerun()
    except Exception as exc:
        st.error(str(exc))


try:
    df = load_models()
    player_names = set(df["PLAYER_NAME"])
    saved = read_json(STATE_FILE, {"version": 1, "picks": []})
    validate_picks(saved["picks"], player_names)
    config = read_json(CONFIG_FILE, {"teams": 10, "position": None})
except Exception as exc:
    st.error(f"Cannot load model or saved draft: {exc}")
    st.info("Check data/processed CSVs and src/draft_engine.py. Existing files have not been modified.")
    st.stop()

if type(config.get("teams")) is not int or not 2 <= config["teams"] <= 14:
    config = {"teams": 10, "position": None}
if type(config.get("position")) is not int or not 1 <= config["position"] <= config["teams"]:
    config["position"] = None

picks = saved["picks"]
owners = {p["player"]: p["owner"] for p in picks}
df["STATUS"] = df["PLAYER_NAME"].map(owners).fillna("available")
metrics = {label: col for label, col in METRICS.items() if col in df.columns}
reverse_metrics = {column: label for label, column in metrics.items()}
valid_presets = {name: axes for name, axes in PRESETS.items() if all(col in metrics.values() for col in axes)}
if not valid_presets:
    st.error("None of the eight presets match the model columns.")
    st.stop()

def apply_selected_preset():
    """Apply the selected preset before scores are calculated on rerun."""
    selected = st.session_state.get("lab_preset")
    if selected in valid_presets:
        for key, metric in zip(("lab_x", "lab_y", "lab_color", "lab_size"), valid_presets[selected]):
            st.session_state[key] = reverse_metrics[metric]
        st.session_state["lab_last_applied_preset"] = selected
        st.session_state["lab_draft_sort"] = "Preset Analysis Score"


# All controls share session state, so Draft and Analysis modes always use the same preset.
default_preset = "Best Overall Draft Targets" if "Best Overall Draft Targets" in valid_presets else next(iter(valid_presets))
if st.session_state.get("lab_preset") not in [*valid_presets, "Custom"]:
    st.session_state["lab_preset"] = default_preset
if st.session_state.get("lab_last_applied_preset") != st.session_state["lab_preset"] and st.session_state["lab_preset"] != "Custom":
    for key, col in zip(("lab_x", "lab_y", "lab_color", "lab_size"), valid_presets[st.session_state["lab_preset"]]):
        st.session_state[key] = reverse_metrics[col]
    st.session_state["lab_last_applied_preset"] = st.session_state["lab_preset"]
for key, col in zip(("lab_x", "lab_y", "lab_color", "lab_size"), valid_presets[default_preset]):
    if st.session_state.get(key) not in metrics:
        st.session_state[key] = reverse_metrics[col]
axes = tuple(metrics[st.session_state[key]] for key in ("lab_x", "lab_y", "lab_color", "lab_size"))
if st.session_state["lab_preset"] == "Best Overall Draft Targets":
    extra = best_overall_values(df)
    df = df.join(extra)
    tiered = calibrated_tiers(df, axes, st.session_state["lab_preset"])
    # Use the production-and-GP based score rather than composite-percentile blend.
    tiered["CHART_SCORE"] = tiered["BEST_OVERALL_SCORE"]
    tiered = tiered.sort_values(["CHART_SCORE", "PLAYER_NAME"], ascending=[False, True], kind="stable")
    tiered["TIER"] = pd.Series(pd.NA, index=tiered.index, dtype="Int64")
    top = tiered.loc[tiered["CHART_SCORE"].notna()].head(300)
    tiered.loc[top.index, "TIER"] = pd.array([i // 50 + 1 for i in range(len(top))], dtype="Int64")
else:
    tiered = calibrated_tiers(df, axes, st.session_state["lab_preset"])
tier_map = tiered.set_index("PLAYER_NAME")[["CHART_SCORE", "TIER"]]
df = df.join(tier_map, on="PLAYER_NAME")

st.markdown('''<div class="dash-header"><div class="brand"><div class="brand-icon">🏀</div>
<div><div class="brand-title"><span>FANTASY</span> DRAFT LAB</div><div class="brand-year">2026</div></div></div>
<div class="header-side">◉ PUBLIC INTERACTIVE DRAFT ROOM &nbsp; ✦ PURPLE EDITION &nbsp; ☾</div></div>''', unsafe_allow_html=True)

if st.session_state.get("lab_mode") not in ("Draft Mode", "Analysis Mode"):
    st.session_state["lab_mode"] = "Draft Mode"
nav_l, nav_c, nav_r = st.columns([1.25, 1.9, 1.25])
with nav_c:
    mode = st.radio("Workspace", ["Draft Mode", "Analysis Mode"], horizontal=True, key="lab_mode", label_visibility="collapsed")

# Rapid Draft: the native searchable dropdown filters as the user types.
# Selecting a result records it immediately via on_change (no submit button).
# The draft owner is inferred from the snake schedule when a slot is configured.
def rapid_record_selection():
    selected_name = st.session_state.get("lab_rapid_player")
    if not selected_name:
        return
    try:
        current = read_json(STATE_FILE, {"version": 1, "picks": []})
        validate_picks(current["picks"], player_names)
        next_number = len(current["picks"]) + 1
        draft_slot = st.session_state.get("lab_slot", "Not announced")
        draft_teams = st.session_state.get("lab_teams", config["teams"])
        if draft_slot != "Not announced":
            my_turns = scheduled_picks(int(draft_teams), int(draft_slot))
            owner = "mine" if next_number in my_turns else "opponent"
        else:
            owner = st.session_state.get("lab_rapid_owner", "opponent")
        write_pick(selected_name, owner)
        st.session_state["lab_rapid_notice"] = f"Recorded #{next_number}: {selected_name} ({owner})"
    except Exception as exc:
        st.session_state["lab_rapid_notice"] = f"Could not record pick: {exc}"
    finally:
        # A callback may reset the widget's state before the next rerun.
        st.session_state["lab_rapid_player"] = None


if mode == "Draft Mode":
    # Draft-day command strip: compact, immediately visible above all controls.
    draft_total = int(st.session_state.get("lab_teams", config["teams"])) * 13
    draft_round = min(13, len(picks) // int(st.session_state.get("lab_teams", config["teams"])) + 1)
    my_names = [p["player"] for p in picks if p["owner"] == "mine"]
    last_label = (f'Last: #{len(picks)} {picks[-1]["player"]} · {"Mine" if picks[-1]["owner"] == "mine" else "Opponent"}'
                  if picks else "No selections recorded")
    command_slot = st.session_state.get("lab_slot", str(config["position"]) if config["position"] else "Not announced")
    command_teams = int(st.session_state.get("lab_teams", config["teams"]))
    command_turns = scheduled_picks(command_teams, int(command_slot)) if str(command_slot).isdigit() and 1 <= int(command_slot) <= command_teams else []
    next_turn = next((n for n in command_turns if n > len(picks)), None)
    next_turn_label = f"NEXT MINE #{next_turn} · {next_turn-len(picks)-1} away" if next_turn else "SET DRAFT SLOT" if not command_turns else "ALL PICKS DONE"
    st.markdown(f'<div class="draft-command"><b>{esc(next_turn_label)}</b><b>ROUND {draft_round}</b><b>PICK {len(picks)+1 if len(picks)<draft_total else draft_total} / {draft_total}</b>'
                f'<b>MY ROSTER {len(my_names)} / 13</b><span>{esc(last_label)}</span></div>', unsafe_allow_html=True)
    st.markdown("#### ⚡ RAPID PICK · Search any player")
    rapid_left, rapid_mid, rapid_right = st.columns([4, 1.35, 1.4], gap="small")
    rapid_available = df.loc[df["STATUS"].eq("available")].sort_values(
        ["CHART_SCORE", "PLAYER_NAME"], ascending=[False, True], na_position="last", kind="stable"
    )
    rapid_names = rapid_available["PLAYER_NAME"].tolist()
    if st.session_state.get("lab_rapid_player") not in rapid_names:
        st.session_state["lab_rapid_player"] = None
    with rapid_left:
        st.selectbox(
            "Search and select a drafted player (typing filters the list)",
            options=rapid_names,
            index=None,
            placeholder="Type a few letters, then select — saved instantly",
            key="lab_rapid_player",
            on_change=rapid_record_selection,
        )
    with rapid_mid:
        slot_for_rapid = st.session_state.get("lab_slot", "Not announced")
        if slot_for_rapid == "Not announced":
            st.selectbox("Pick belongs to", ["opponent", "mine"], key="lab_rapid_owner")
        else:
            upcoming = len(picks) + 1
            mine_turn = upcoming in scheduled_picks(int(st.session_state.get("lab_teams", config["teams"])), int(slot_for_rapid))
            st.metric("Next pick belongs to", "YOU" if mine_turn else "OPPONENT")
    with rapid_right:
        st.metric("Draft progress", f"{len(picks)} / {int(st.session_state.get('lab_teams', config['teams'])) * 13}")
        if st.button("↶ Undo last pick", key="lab_rapid_undo", disabled=not picks, use_container_width=True):
            run_action(undo_pick)
    if st.session_state.get("lab_rapid_notice"):
        st.caption(st.session_state["lab_rapid_notice"])
    st.caption("Search selects and records instantly. For top players, use Auto/Mine/Opponent below. Undo reverses the most recent pick.")
    # Pick corrections and a downloadable audit trail stay accessible without scrolling.
    with st.expander("🛠 Correct last pick / export draft log", expanded=False):
        if picks:
            correction_options = [name for name in rapid_names if name != picks[-1]["player"]]
            replacement = st.selectbox("Replace last drafted player with", correction_options, index=None,
                                       placeholder="Search correct player", key="lab_correct_last_player")
            if st.button("Replace last pick (keep owner)", disabled=replacement is None, key="lab_correct_last_button"):
                run_action(correct_last_pick, replacement)
        else:
            st.caption("Record a pick to enable correction.")
        st.download_button("⬇ Download draft pick log (CSV)",
                           data=pick_log_csv(picks, int(st.session_state.get("lab_teams", config["teams"])),
                                             int(command_slot) if str(command_slot).isdigit() else None),
                           file_name="fantasy_draft_pick_log.csv", mime="text/csv", key="lab_export_picks")

# Three-column composition closely follows the visual reference.
left, center, right = st.columns([1.28, 4.45, 1.36], gap="small")
with left:
    st.markdown('<div class="glass"><div class="card-title">♟ DRAFT CONFIGURATION</div></div>', unsafe_allow_html=True)
    if "lab_teams" not in st.session_state:
        st.session_state["lab_teams"] = config["teams"]
    teams = st.selectbox("Number of Teams", list(range(2, 15)), key="lab_teams")
    slots = ["Not announced"] + [str(n) for n in range(1, teams + 1)]
    if "lab_slot" not in st.session_state:
        st.session_state["lab_slot"] = str(config["position"]) if config["position"] else "Not announced"
    if st.session_state["lab_slot"] not in slots:
        st.session_state["lab_slot"] = "Not announced"
    slot_choice = st.selectbox("Your Draft Position", slots, key="lab_slot")
    slot = None if slot_choice == "Not announced" else int(slot_choice)
    if config != {"teams": teams, "position": slot}:
        atomic_write(CONFIG_FILE, {"teams": teams, "position": slot})
    schedule = scheduled_picks(teams, slot)
    next_picks = [n for n in schedule if n > len(picks)]
    if schedule:
        rows = "".join(f'<div class="pick-row"><b>#{p}</b><span>Round {r}</span></div>' for r, p in enumerate(schedule[:5], 1))
        box("YOUR UPCOMING PICKS", rows + '<div class="mini" style="margin-top:7px">Full schedule in Draft Strategy →</div>')
    else:
        box("YOUR UPCOMING PICKS", '<div class="mini">Select a draft position to see your scheduled turns.</div>')

    available = df[df["STATUS"] == "available"].copy()
    with st.expander("Alternative manual pick entry · optional"):
        st.markdown('<div class="glass"><div class="card-title">⚡ DRAFT CONTROLS</div></div>', unsafe_allow_html=True)
        if "lab_pick_search" not in st.session_state:
            st.session_state["lab_pick_search"] = ""
        search_pick = st.text_input("Search for a player", key="lab_pick_search", placeholder="Type a player name…")
        candidates = available
        if search_pick.strip():
            candidates = candidates[candidates["PLAYER_NAME"].str.contains(search_pick.strip(), case=False, regex=False, na=False)]
        candidates = candidates.sort_values(["ORIGINAL_RANK", "PLAYER_NAME"])
        options = candidates["PLAYER_NAME"].tolist()
        selected = st.selectbox("Select matching player", options, index=None, placeholder="Choose available player", key="lab_selected_pick") if options else None
        if not options:
            st.caption("No matching available players.")
        if st.button("♟  Mark as My Pick", use_container_width=True, disabled=selected is None, key="lab_pick_mine"):
            run_action(write_pick, selected, "mine")
        if st.button("♟  Mark as Opponent Pick", use_container_width=True, disabled=selected is None, key="lab_pick_opponent"):
            run_action(write_pick, selected, "opponent")
        if st.button("↶  Undo Last Pick", use_container_width=True, disabled=not picks, key="lab_undo"):
            run_action(undo_pick)
        if picks:
            st.caption(f'Last pick #{len(picks)}: {picks[-1]["player"]} ({picks[-1]["owner"]})')

    total = teams * 13
    percent = min(100, 100 * len(picks) / total)
    box("DRAFT PROGRESS", f'<div style="font-size:22px;color:#e8dcff"><b>{len(picks)} / {total}</b></div>'
        f'<div class="mini">Selections recorded · {max(0,total-len(picks))} remaining</div>'
        f'<div class="progress-track" style="margin-top:10px"><div class="progress-fill" style="width:{percent:.2f}%"></div></div>')
    box("SAVE STATUS", '<div class="mini"><span class="status-dot"></span>Saved locally after each pick or configuration change</div>')
    with st.expander("⚙ Settings · Reset board"):
        st.caption("Reset removes all picks, restores 10 teams and default analysis settings. A backup of picks is saved first.")
        confirm = st.checkbox("I understand this will erase the current board", key="lab_reset_confirm")
        if st.button("RESET BOARD TO DEFAULT", disabled=not confirm, use_container_width=True):
            run_action(reset_board)
    with st.expander("? Help & scoring"):
        st.caption("Pick a preset, search a player, and record every draft selection in order. The app never changes model CSVs.")
        st.caption("Weighted analysis: X 35%, Y 35%, color 20%, size 10%. Original Draft Value remains independent.")
        st.caption("ESPN position eligibility is not verified, so positional recommendations are intentionally omitted.")

with center:
    st.markdown('<div class="glass glass-purple"><div class="card-title">✦ ANALYSIS PRESET</div>'
                '<div class="mini">Switch your analysis perspective without changing Draft Value or recorded picks.</div></div>', unsafe_allow_html=True)
    # Same widget key in both modes: the selection persists when navigating.
    st.selectbox("Analysis Preset", [*valid_presets, "Custom"], key="lab_preset", on_change=apply_selected_preset)
    preset = st.session_state["lab_preset"]
    st.caption(NOTES[preset])
    if preset == "Best Overall Draft Targets":
        st.caption("Best Overall ranks by our 2026–27 projected FP/G with a bounded three-season games-played adjustment (0–16%). No stacked Draft Value, Competitive Edge, or playoff penalty. Players without a forecast use last-season FP/G as a flagged fallback. This heuristic has not been independently backtested.")
    if preset != "Custom" and axes != valid_presets[preset]:
        st.info("Applying the selected preset. Its axes and tiers update on the next refresh.")
    weights = [("X axis", 35), ("Y axis", 35), ("Color", 20), ("Size", 10)]
    weight_cards = "".join(f'<div class="kpi"><b>{pct}%</b><span>{esc(label)}</span></div>' for label,pct in weights)
    st.markdown('<div class="kpi-row">' + weight_cards + '</div>', unsafe_allow_html=True)
    with st.expander("View Preset Details"):
        for label, metric in zip(("X · 35%", "Y · 35%", "Color · 20%", "Size · 10%"), axes):
            st.write(f"**{label}:** {reverse_metrics.get(metric, metric)}")
        if preset == "Best Overall Draft Targets":
            st.caption("Best Overall: role-aware projected FP/G × (0.84 + 0.16 × weighted three-season GP / 82), then ranked against all players. Axes remain chart dimensions, not scoring weights.")
        elif preset in PROFILES:
            st.caption("Preset-specific weighted percentile scoring; the axes remain chart dimensions. See src/preset_calibration.py for exact weights.")
        else:
            st.caption("Custom: X 35%, Y 35%, Color 20%, Size 10% percentile ranks.")

    if mode == "Draft Mode":
        sort_choices = ["Preset Analysis Score", "2026–27 Model Projection", "Overall Draft Value", "Edge-Adjusted Draft Value", "Fantasy Points/Game", "Experimental Role FPPG", "Player Name"]
        if "MODEL_DRAFT_FPPG" not in df.columns or df["MODEL_DRAFT_FPPG"].notna().sum() == 0:
            sort_choices.remove("2026–27 Model Projection")
        if "EXPERIMENTAL_ROLE50_FPPG" not in df.columns:
            sort_choices.remove("Experimental Role FPPG")
        if st.session_state.get("lab_draft_sort") not in sort_choices:
            st.session_state["lab_draft_sort"] = "Preset Analysis Score"
        sort_left, sort_right = st.columns([2.7, 1.4])
        with sort_left:
            st.markdown('### ▥ BEST AVAILABLE PLAYERS')
            st.caption(f"Default: selected preset score · {len(available)} players available. Click any column header to sort the visible table.")
        with sort_right:
            st.selectbox("Sort by", sort_choices, key="lab_draft_sort")
        sort_by = st.session_state["lab_draft_sort"]
        sort_columns = {"2026–27 Model Projection":"MODEL_DRAFT_FPPG", "Preset Analysis Score":"CHART_SCORE", "Overall Draft Value":"DRAFT_VALUE_SCORE",
                        "Edge-Adjusted Draft Value":"EDGE_ADJUSTED_DRAFT_SCORE", "Fantasy Points/Game":"FANTASY_PTS_PER_GAME",
                        "Experimental Role FPPG":"EXPERIMENTAL_ROLE50_FPPG", "Player Name":"PLAYER_NAME"}
        sort_col = sort_columns[sort_by]
        ranked_available = available.sort_values(
            [sort_col, "PLAYER_NAME"], ascending=[sort_col == "PLAYER_NAME", True],
            na_position="last", kind="stable"
        )
        # Fast, explicit ownership buttons for the highest-ranked available players.
        # The sortable dataframe remains intact below the action controls.
        st.markdown("#### ⚡ ONE-CLICK DRAFT · TOP AVAILABLE")
        st.caption("Click **Mine** or **Opponent** beside a player. Each click immediately records one pick, removes that player from the available list, and saves your draft. Use Undo above for mistakes.")
        # Watchlist persists in the browser session; drafted players are filtered out.
        if "lab_watchlist" not in st.session_state:
            st.session_state["lab_watchlist"] = []
        watchlist = [name for name in st.session_state["lab_watchlist"] if name in set(ranked_available["PLAYER_NAME"])]
        st.session_state["lab_watchlist"] = watchlist
        with st.expander(f"⭐ Watchlist · {len(watchlist)} available", expanded=bool(watchlist)):
            add_watch = st.selectbox("Pin a player for quick access", ranked_available["PLAYER_NAME"].tolist(),
                                     index=None, key="lab_add_watch", placeholder="Search any available player")
            if st.button("Pin to watchlist", disabled=add_watch is None or add_watch in watchlist,
                         key="lab_pin_watch"):
                st.session_state["lab_watchlist"].append(add_watch)
                st.rerun()
            for watch_name in watchlist:
                wa, wb, wc, wd, we = st.columns([3, 1, 1, 1.3, 1], gap="small")
                wa.write(watch_name)
                if wb.button("Auto", key=f"lab_watch_auto_{watch_name}", disabled=not command_turns):
                    run_action(record_auto_pick, watch_name)
                if wc.button("Mine", key=f"lab_watch_mine_{watch_name}"):
                    run_action(write_pick, watch_name, "mine")
                if wd.button("Opponent", key=f"lab_watch_opp_{watch_name}"):
                    run_action(write_pick, watch_name, "opponent")
                if we.button("Unpin", key=f"lab_unpin_{watch_name}"):
                    st.session_state["lab_watchlist"].remove(watch_name)
                    st.rerun()
        quick_button_count = st.selectbox(
            "Quick-action players per page", [5, 10, 15, 20], index=1,
            key="lab_quick_button_count",
        )
        page_total = max(1, (len(ranked_available) + quick_button_count - 1) // quick_button_count)
        if st.session_state.get("lab_quick_page", 1) > page_total:
            st.session_state["lab_quick_page"] = page_total
        if "lab_quick_page" not in st.session_state:
            st.session_state["lab_quick_page"] = 1
        page_prev, page_info, page_next = st.columns([1, 2, 1])
        if page_prev.button("◀ Previous", disabled=st.session_state["lab_quick_page"] <= 1,
                            key="lab_page_prev", use_container_width=True):
            st.session_state["lab_quick_page"] -= 1
            st.rerun()
        page_info.caption(f'Page {st.session_state["lab_quick_page"]} of {page_total} · jump beyond top 20 without typing')
        if page_next.button("Next ▶", disabled=st.session_state["lab_quick_page"] >= page_total,
                            key="lab_page_next", use_container_width=True):
            st.session_state["lab_quick_page"] += 1
            st.rerun()
        page_start = (st.session_state["lab_quick_page"] - 1) * quick_button_count
        quick_rows = ranked_available.iloc[page_start:page_start + quick_button_count]
        hdr_name, hdr_score, hdr_auto, hdr_mine, hdr_opp = st.columns([3.8, 1, 1.1, 1, 1.3], gap="small")
        hdr_name.markdown("**Player**")
        hdr_score.markdown("**Score**")
        hdr_auto.markdown("**Auto**")
        hdr_mine.markdown("**Mine**")
        hdr_opp.markdown("**Opponent**")
        for quick_rank, (_, quick_player) in enumerate(quick_rows.iterrows(), page_start + 1):
            quick_name = str(quick_player["PLAYER_NAME"])
            quick_team = str(quick_player.get("TEAM_ABBREVIATION", ""))
            quick_score = quick_player.get("CHART_SCORE")
            name_col, score_col, auto_col, mine_col, opp_col = st.columns([3.8, 1, 1.1, 1, 1.3], gap="small")
            name_col.markdown(f"**{quick_rank}. {esc(quick_name)}**  ·  {esc(quick_team)}")
            score_col.write(f"{quick_score:.1f}" if pd.notna(quick_score) else "—")
            if auto_col.button("Auto", key=f"lab_fast_auto_{quick_name}", disabled=not command_turns,
                               use_container_width=True, help="Assign Mine/Opponent from snake-draft position"):
                run_action(record_auto_pick, quick_name)
            if mine_col.button("Mine", key=f"lab_fast_mine_{quick_name}", use_container_width=True):
                run_action(write_pick, quick_name, "mine")
            if opp_col.button("Opponent", key=f"lab_fast_opp_{quick_name}", use_container_width=True):
                run_action(write_pick, quick_name, "opponent")
        st.caption("These buttons explicitly mark ownership, regardless of your configured snake-draft slot. The Rapid Draft dropdown above still auto-assigns ownership when your slot is configured.")

        with st.expander("Full sortable rankings · click to expand", expanded=False):
            # Streamlit's native grid supports sortable column headers, including direction arrows.
            # Slice the DataFrame BEFORE rendering so the row-count selector actually limits rows.
            row_options = [5, 10, 20, 50, 100]
            if st.session_state.get("lab_draft_rows") not in row_options:
                st.session_state["lab_draft_rows"] = 20
            row_count = st.selectbox("Players to display", row_options, key="lab_draft_rows")
            displayed = ranked_available.head(row_count).copy()
            displayed.insert(0, "#", range(1, len(displayed) + 1))
            board_cols = {
                "#": "#", "PLAYER_NAME": "Player", "TEAM_ABBREVIATION": "Team",
                "TIER": "Tier", "CHART_SCORE": "Preset Score", "ORIGINAL_RANK": "Overall Rank",
                "DRAFT_VALUE_SCORE": "Draft Value", "FANTASY_PTS_PER_GAME": "FP/G",
                "EXPERIMENTAL_ROLE50_FPPG": "2026–27 Exp. FP/G", "GP": "GP",
            }
            visible_cols = [c for c in board_cols if c in displayed.columns]
            board = displayed[visible_cols].rename(columns=board_cols)
            if "Tier" in board.columns:
                board["Tier"] = board["Tier"].map(lambda n: f"T{int(n)}" if pd.notna(n) else "—")
            st.dataframe(board, hide_index=True, use_container_width=True,
                         height=min(760, 38 + len(board) * 35),
                         column_config={
                             name: st.column_config.NumberColumn(name, format="%.2f")
                             for name in ("Preset Score", "Draft Value", "FP/G", "2026–27 Exp. FP/G")
                             if name in board.columns
                         })
        st.caption("Click a column heading to sort ascending or descending; the grid shows its active sort arrow. Changing the preset resets the initial ordering to that preset's score. The row selector limits the displayed pool before interactive sorting.")
        if "EXPERIMENTAL_ROLE50_FPPG" in df.columns:
            with st.expander("About the experimental 2026–27 projection"):
                st.write("ROLE_50 combines an age-adjusted minutes × productivity forecast with a 50% adjustment for historical role changes. It is an experimental per-game forecast conditional on playing, not a games-played or injury forecast.")
        with st.expander("Quick draft from the top targets"):
            quick_options = displayed["PLAYER_NAME"].tolist()
            quick = st.selectbox("Select player from displayed rankings", quick_options, index=None, key="lab_quick_player")
            qa, qb = st.columns(2)
            if qa.button("Draft to My Team", disabled=quick is None, use_container_width=True, key="lab_quick_mine"):
                run_action(write_pick, quick, "mine")
            if qb.button("Draft to Opponent", disabled=quick is None, use_container_width=True, key="lab_quick_opp"):
                run_action(write_pick, quick, "opponent")
    else:
        st.markdown('### ◈ ANALYSIS MODE')
        st.caption("Same preset and tiers as Draft Mode. Customize axes or filter the chart without changing a player's score.")
        metric_labels = list(metrics)
        def mark_custom():
            st.session_state["lab_preset"] = "Custom"
            st.session_state["lab_last_applied_preset"] = "Custom"

        controls = st.columns(4)
        for col, key, label in zip(controls, ("lab_x", "lab_y", "lab_color", "lab_size"), ("X axis", "Y axis", "Color", "Size")):
            with col:
                st.selectbox(label, metric_labels, key=key, on_change=mark_custom)
        current_axes = tuple(metrics[st.session_state[k]] for k in ("lab_x", "lab_y", "lab_color", "lab_size"))
        filter1, filter2 = st.columns(2)
        with filter1:
            low, high = st.slider("Overall draft rank range", 1, len(df), (1, min(300,len(df))), key="lab_rank_range")
            population = st.radio("Chart population", ["All Players", "Available Only"], horizontal=True, key="lab_population")
        with filter2:
            min_gp = st.slider("Minimum games played", 0, 82, 0, key="lab_gp_filter")
            query = st.text_input("Search chart players", key="lab_chart_search")
        view = df[df["ORIGINAL_RANK"].between(low,high) & (df["GP"] >= min_gp)].copy()
        if population == "Available Only":
            view = view[view["STATUS"] == "available"]
        if query.strip():
            view = view[view["PLAYER_NAME"].str.contains(query.strip(), case=False, regex=False, na=False)]
        view = view.dropna(subset=list(set(axes))).copy()
        if view.empty:
            st.info("No players match the current chart filters.")
        else:
            fig = go.Figure()
            sizes = pd.to_numeric(view[axes[3]], errors="coerce")
            spread = sizes.max() - sizes.min()
            view["_MARKER_SIZE"] = 17.0 if not np.isfinite(spread) or spread <= 0 else 10 + 18 * (sizes - sizes.min()) / spread
            for owner, label in (("opponent","Drafted — Opponent"),("available","Available"),("mine","Drafted — My Team")):
                subset = view[view["STATUS"] == owner]
                if subset.empty:
                    continue
                custom = np.stack([subset["PLAYER_NAME"].astype(str),subset["ORIGINAL_RANK"].astype(str),
                                   subset["CHART_SCORE"].round(2).astype(str)],axis=-1)
                marker = {"size":subset["_MARKER_SIZE"],"opacity":.35 if owner=="opponent" else .92,
                          "line":{"width":1.4 if owner=="mine" else .4,"color":"#e0ceff"}}
                if owner == "available":
                    marker.update({"color":subset[axes[2]],"colorscale":"Plasma","showscale":True,
                                   "colorbar":{"title":reverse_metrics.get(axes[2],axes[2])}})
                else:
                    marker["color"] = "#35dba5" if owner == "mine" else "#7c7898"
                fig.add_trace(go.Scattergl(x=subset[axes[0]],y=subset[axes[1]],mode="markers",name=label,
                    marker=marker,customdata=custom,hovertemplate="<b>%{customdata[0]}</b><br>Overall rank: %{customdata[1]}<br>Preset score: %{customdata[2]}<extra></extra>"))
            fig.update_layout(height=570,paper_bgcolor=BG,plot_bgcolor=BG,font={"color":"#e9dfff"},
                xaxis={"title":reverse_metrics.get(axes[0],axes[0]),"gridcolor":"#30284b"},
                yaxis={"title":reverse_metrics.get(axes[1],axes[1]),"gridcolor":"#30284b"},
                legend={"orientation":"h","y":1.14},margin={"l":45,"r":60,"t":70,"b":50})
            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
            st.markdown("### CHART RESULTS")
            st.caption("Exactly the plotted players; scores and tiers calculated from the complete model population.")
            sort_options = {"Weighted Chart Score":"CHART_SCORE","Chart Y":axes[1],"Chart X":axes[0],
                            "Chart Color":axes[2],"Chart Size":axes[3],"Overall Draft Rank":"ORIGINAL_RANK","Player Name":"PLAYER_NAME"}
            s1,s2 = st.columns(2)
            sort_label = s1.selectbox("Sort chart results",list(sort_options),key="lab_chart_sort")
            limit = s2.selectbox("Table rows",[25,50,100,200,len(df)],index=1,key="lab_chart_rows")
            sort_key = sort_options[sort_label]
            view = view.sort_values([sort_key,"PLAYER_NAME"] if sort_key != "PLAYER_NAME" else ["PLAYER_NAME"],
                ascending=(sort_label in ("Overall Draft Rank","Player Name")),kind="stable",na_position="last")
            cols = list(dict.fromkeys(["PLAYER_NAME","TIER","STATUS","ORIGINAL_RANK","CHART_SCORE",*axes]))
            result = view[cols].head(limit).copy()
            result["TIER"] = result["TIER"].apply(lambda x:f"Tier {int(x)}" if pd.notna(x) else "Unclassified")
            rename = {"PLAYER_NAME":"Player","TIER":"Tier","STATUS":"Status","ORIGINAL_RANK":"Overall Rank",
                      "CHART_SCORE":"Preset Score",**reverse_metrics}
            st.dataframe(result.rename(columns=rename),hide_index=True,use_container_width=True)

    # Tier cards are shared between Draft and Analysis modes, and update with the preset.
    show_tiers = st.toggle("Show Tiers · 300 players", value=True, key="lab_show_tiers")
    if show_tiers:
        st.markdown("### ▥ TIER SUMMARY (300 PLAYERS)")
    tier_cards = []
    for i,(color,title) in enumerate(TIERS,1):
        remaining = int(((df["TIER"]==i)&(df["STATUS"]=="available")).sum())
        tier_cards.append(f'<div class="tier-card" style="--tc:{color}"><span class="tier-number">{i}</span>'
                          f'<span class="tier-title">{esc(title)}</span><div class="tier-range">#{(i-1)*50+1}–{i*50}</div>'
                          f'<div class="tier-remain">{remaining} available / 50</div></div>')
    if show_tiers:
        st.markdown('<div class="tier-grid">'+''.join(tier_cards)+'</div>',unsafe_allow_html=True)
        st.caption("Six fixed groups of 50 based on the selected preset, not overall Draft Value. Drafting does not shift tiers.")

with right:
    st.markdown('<div class="glass"><div class="card-title card-title-gold">♕ DRAFT STRATEGY</div></div>',unsafe_allow_html=True)
    if slot is None:
        box("NEXT PICK", '<div class="mini">Set your draft position to activate the snake-draft countdown.</div>',gold=True)
    elif next_picks:
        upcoming = next_picks[0]
        round_no = schedule.index(upcoming)+1
        between = upcoming-len(picks)-1
        next_info = f'<div style="font-size:16px;color:#eee1ff;font-weight:800">Next Pick: #{upcoming} (Round {round_no})</div>'
        next_info += f'<div class="mini">{between} selections before your turn.</div>'
        if len(next_picks)>1:
            next_info += f'<div class="mini">Following pick: #{next_picks[1]}</div>'
        box("YOUR TURN",next_info,gold=True)
    else:
        box("YOUR TURN",'<div class="mini">All 13 scheduled turns have passed.</div>',gold=True)
    warnings = validate_pick_order(picks,teams,slot)
    if warnings:
        with st.expander(f"⚠ Pick-order warnings ({len(warnings)})"):
            for warning in warnings[:25]:
                st.warning(warning)
    # Decision panel: uses the selected preset, not stale overall-draft-value rank.
    top_preset = available.sort_values(["CHART_SCORE", "PLAYER_NAME"], ascending=[False, True], na_position="last").head(5)
    suggestions = ''.join(
        f'<div class="pick-row"><span class="pill">{n}</span><span>{esc(row["PLAYER_NAME"])}<br>'
        f'<span class="mini">Projected {row["MODEL_DRAFT_FPPG"]:.1f} FP/G · Score {row["CHART_SCORE"]:.1f}</span></span></div>'
        if pd.notna(row.get("MODEL_DRAFT_FPPG")) and pd.notna(row.get("CHART_SCORE")) else
        f'<div class="pick-row"><span class="pill">{n}</span><span>{esc(row["PLAYER_NAME"])} · Score {row["CHART_SCORE"]:.1f}</span></div>'
        for n, (_, row) in enumerate(top_preset.iterrows(), 1))
    box("NEXT-PICK TARGETS · CURRENT PRESET", suggestions or '<div class="mini">No available targets.</div>', gold=True)
    my_roster = [p["player"] for p in picks if p["owner"] == "mine"]
    roster_body = ''.join(f'<div class="pick-row"><span>{i}. {esc(name)}</span></div>' for i, name in enumerate(my_roster, 1))
    box(f"MY ROSTER · {len(my_roster)}/13", roster_body or '<div class="mini">Your selections will appear here.</div>')
    recent = []
    for number,pick in reversed(list(enumerate(picks,1))[-6:]):
        label = "You" if pick["owner"] == "mine" else "Opponent"
        color = "#32e0ad" if pick["owner"] == "mine" else "#ff7488"
        recent.append(f'<div class="pick-row"><span>#{number}</span><span style="color:{color}">{label}</span><span>{esc(pick["player"])}</span></div>')
    box("◷ RECENT PICKS",''.join(recent) if recent else '<div class="mini">No picks recorded yet.</div>')
    mine = [p["player"] for p in picks if p["owner"] == "mine"]
    with st.expander(f"MY TEAM ({len(mine)})"):
        if mine:
            for index,name in enumerate(mine,1):
                st.write(f"{index}. {name}")
        else:
            st.caption("No players drafted to your team.")
    with st.expander("VIEW ALL SCHEDULED PICKS"):
        if schedule:
            st.write(", ".join(f"#{n}" for n in schedule))
        else:
            st.caption("Draft position not selected.")

st.markdown('<div class="info-line" style="margin-top:20px">PUBLIC SESSION VERSION · Model CSVs are read-only · No ESPN API calls · Playoff team assignments are provisional · Do not record picks in the notebook and website at the same time.</div>',unsafe_allow_html=True)
