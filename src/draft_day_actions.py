"""Pure helpers for draft-day actions; never modify projections."""
from __future__ import annotations
import csv
import io
from draft_engine import scheduled_picks, validate_picks


def owner_for_next_pick(picks: list, teams: int, slot: int | None) -> str | None:
    if slot is None:
        return None
    return 'mine' if len(picks) + 1 in scheduled_picks(teams, slot) else 'opponent'


def replace_last_pick(picks: list, player: str, players: set[str]) -> list:
    if not picks:
        raise ValueError('No picks to correct')
    if player not in players:
        raise ValueError('Unknown player')
    if player in {p['player'] for p in picks[:-1]}:
        raise ValueError('That player was drafted earlier')
    updated = [dict(p) for p in picks]
    updated[-1]['player'] = player
    validate_picks(updated, players)
    return updated


def pick_log_csv(picks: list, teams: int, slot: int | None) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Pick', 'Round', 'Player', 'Recorded owner', 'Expected owner', 'Timestamp UTC'])
    for number, pick in enumerate(picks, 1):
        expected = 'mine' if slot is not None and number in scheduled_picks(teams, slot) else ('opponent' if slot is not None else '')
        writer.writerow([number, (number - 1) // teams + 1, pick['player'], pick['owner'], expected, pick.get('timestamp_utc', '')])
    return output.getvalue()
