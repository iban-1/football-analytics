"""Phase 3 tests: unit tests on tiny hand-made data, plus sanity checks of
analysis totals against the raw events (skipped if the data cache is absent)."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.events import (add_match_time, is_key_pass, is_progressive,
                                 network_metrics, pass_network, penalty_conversion_rate,
                                 player_summary, shots_with_xg, touches)
from src.data.statsbomb import _cache_dir, load_all_events, load_all_matches
from src.features.shots import goal_events


# ---------- unit tests ----------

def _passes(rows):
    base = {"type": "Pass", "match_id": 1, "team": "A", "player": "p", "pass_recipient": None,
            "pass_outcome": None, "pass_type": None, "pass_shot_assist": None,
            "pass_goal_assist": None, "location": [60.0, 40.0], "pass_end_location": [60.0, 40.0],
            "index": 1, "minute": 1, "period": 1, "ball_receipt_outcome": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_progressive_pass_definition():
    p = _passes([
        {"location": [60.0, 40.0], "pass_end_location": [75.0, 40.0]},                # +15 -> yes
        {"location": [60.0, 40.0], "pass_end_location": [65.0, 40.0]},                # +5 -> no
        {"location": [60.0, 40.0], "pass_end_location": [75.0, 40.0], "pass_outcome": "Incomplete"},
        {"location": [5.0, 40.0], "pass_end_location": [60.0, 40.0], "pass_type": "Goal Kick"},
        {"location": [60.0, 40.0], "pass_end_location": [40.0, 40.0]},                # backwards
    ])
    assert is_progressive(p).tolist() == [True, False, False, False, False]


def test_key_pass_flag():
    p = _passes([{"pass_shot_assist": True}, {"pass_goal_assist": True}, {}])
    assert is_key_pass(p).tolist() == [True, True, False]


def test_pass_network_only_before_first_sub_and_undirected():
    rows = [
        {"index": 1, "player": "A", "pass_recipient": "B", "location": [10.0, 10.0]},
        {"index": 2, "player": "B", "pass_recipient": "A", "location": [30.0, 20.0]},
        {"index": 3, "player": "A", "pass_recipient": "C", "location": [20.0, 30.0]},
        {"index": 4, "player": "A", "pass_recipient": "B", "pass_outcome": "Incomplete"},
        {"index": 10, "type": "Substitution", "player": "C"},
        {"index": 11, "player": "A", "pass_recipient": "B", "location": [10.0, 10.0]},
    ]
    ev = _passes(rows)
    ev.loc[ev["type"] == "Substitution", "type"] = "Substitution"
    nodes, edges, minute = pass_network(ev, "A")
    w = {(r.a, r.b): r.weight for r in edges.itertuples()}
    assert w == {("A", "B"): 2, ("A", "C"): 1}  # A->B and B->A merged; late/failed passes excluded
    assert nodes.set_index("player").loc["A", "passes"] == 2
    assert nodes.set_index("player").loc["A", "x"] == pytest.approx(15.0)
    assert minute == 1
    m = network_metrics(nodes, edges).set_index("player")
    assert m.loc["A", "degree"] == 2 and m.loc["A", "strength"] == 3
    assert m["betweenness"].idxmax() == "A"  # A connects B and C


def test_touches_drop_failed_receipts():
    ev = _passes([{}, {"type": "Ball Receipt*", "ball_receipt_outcome": "Incomplete"},
                  {"type": "Ball Receipt*"}, {"type": "Pressure"}])
    assert len(touches(ev)) == 2


def test_match_time_is_monotonic_across_periods():
    events = pd.DataFrame({"match_id": [1] * 4, "period": [1, 2, 2, 3], "minute": [47, 45, 96, 90]})
    shots = pd.DataFrame({"match_id": [1, 1, 1], "period": [1, 2, 3], "minute": [47.0, 96.0, 91.0]})
    t = add_match_time(shots, events)["t"].tolist()
    assert t == [47.0, 98.0, 91.0 + 2 + 6] and t == sorted(t)


# ---------- validation against raw events ----------

needs_data = pytest.mark.skipif(not (_cache_dir() / "competitions.pkl").exists(),
                                reason="StatsBomb cache missing: run scripts.download_data")


@pytest.fixture(scope="module")
def data():
    matches = load_all_matches()
    events = load_all_events(matches)
    pen = penalty_conversion_rate(events)
    shots = shots_with_xg(events, pen)
    return matches, events, shots, player_summary(events, shots)


@needs_data
def test_goals_per_team_match_official_scoreline(data):
    matches, events, shots, _ = data
    goals = goal_events(events).groupby(["match_id", "team"]).size()
    for m in matches.itertuples():
        assert goals.get((m.match_id, m.home_team), 0) == m.home_score, m.match_id
        assert goals.get((m.match_id, m.away_team), 0) == m.away_score, m.match_id


@needs_data
def test_shots_table_matches_raw_events(data):
    _, events, shots, summ = data
    raw = events[(events["type"] == "Shot") & (events["period"] != 5)]
    assert len(shots) == len(raw)
    assert summ["shots"].sum() == len(raw)
    assert summ["goals"].sum() == (raw["shot_outcome"] == "Goal").sum() == shots["is_goal"].sum()
    assert ((shots["xg"] > 0) & (shots["xg"] < 1)).all()
    assert summ["xg"].sum() == pytest.approx(shots["xg"].sum(), abs=0.5)  # rounding per player


@needs_data
def test_single_match_xg_equals_tournament_computation(data):
    """The dashboard computes xG from one match's events alone; it must agree."""
    matches, events, shots, _ = data
    pen = penalty_conversion_rate(events)
    for mid in (int(matches["match_id"].iloc[0]), int(matches["match_id"].iloc[-1])):
        alone = shots_with_xg(events[events["match_id"] == mid].drop(
            columns=["competition_id", "season_id"]), pen)
        together = shots[shots["match_id"] == mid].reset_index(drop=True)
        assert np.allclose(alone["xg"].to_numpy(), together["xg"].to_numpy())
        assert (alone["is_goal"].to_numpy() == together["is_goal"].to_numpy()).all()


@needs_data
def test_pass_counts_match_raw_events(data):
    _, events, _, summ = data
    raw = events[events["type"] == "Pass"]
    assert summ["passes"].sum() == len(raw)
    assert summ["passes_completed"].sum() == raw["pass_outcome"].isna().sum()
    assert summ["progressive_passes"].sum() <= summ["passes_completed"].sum()


@needs_data
def test_key_passes_equal_shots_with_a_key_pass_id(data):
    _, events, _, summ = data
    assert summ["key_passes"].sum() == events["shot_key_pass_id"].notna().sum()


@needs_data
def test_pass_network_pass_counts_reconcile(data):
    matches, events, _, _ = data
    mid = int(matches["match_id"].iloc[0])
    ev = events[events["match_id"] == mid]
    for team in ev["team"].unique():
        nodes, edges, _ = pass_network(ev, team)
        # every completed pass is an edge increment unless passer == recipient
        assert edges["weight"].sum() <= nodes["passes"].sum()
        assert nodes["passes"].sum() > 0
