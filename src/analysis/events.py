"""Event-based team and player analysis (Phase 3).

Coordinates are StatsBomb 120 x 80, always oriented so the team performing
the action attacks toward x = 120.
"""
from __future__ import annotations

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from src.features.geometry import shot_distance
from src.features.shots import SHOOTOUT_PERIOD, build_shot_dataset
from src.models.xg import predict_xg
from src.utils.config import load_config, resolve

# A completed pass is "progressive" if it ends at least this many pitch units
# (~yards) closer to the centre of the opponent's goal than it started.
PROGRESSIVE_MIN_GAIN = 10.0
# Goal kicks and kick-offs are excluded: a goal kick is forward by construction
# and would make goalkeepers look like the most progressive passers.
NON_PROGRESSIVE_TYPES = ("Goal Kick", "Kick Off")

# On-ball actions that count as a "touch" for heatmaps.
TOUCH_TYPES = ("Pass", "Ball Receipt*", "Carry", "Dribble", "Shot", "Miscontrol",
               "Dispossessed", "Clearance", "Interception", "Ball Recovery")


# ---------- shots with xG ----------

def penalty_conversion_rate(events: pd.DataFrame) -> float:
    """Empirical in-match penalty conversion rate (shootouts excluded).

    Penalties are kept out of the xG model, so for team totals they are given
    this single data-derived value instead of an invented one.
    """
    pens = _penalties(events)
    return float(pens["is_goal"].mean())


def _penalties(events: pd.DataFrame) -> pd.DataFrame:
    s = events[(events["type"] == "Shot") & (events["shot_type"] == "Penalty")
               & (events["period"] != SHOOTOUT_PERIOD)].copy()
    s["is_goal"] = (s["shot_outcome"] == "Goal").astype(int)
    return s


_PERIOD_START = {1: 0, 2: 45, 3: 90, 4: 105}


def add_match_time(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add `t`: a strictly forward-running match clock in minutes.

    StatsBomb's `minute` restarts at each period's nominal start (45, 90, 105),
    so stoppage time overlaps the next period (e.g. second-half minute 96 comes
    before extra-time minute 91). For each match we shift a period by however
    far the previous period overran its nominal end, so plots never run backwards.
    """
    ends = events.groupby(["match_id", "period"])["minute"].max()
    shift: dict[tuple[int, int], float] = {}
    for match_id in shots["match_id"].unique():
        total = 0.0
        for p in (1, 2, 3, 4):
            if p > 1 and (match_id, p - 1) in ends.index:
                total += max(0.0, ends[(match_id, p - 1)] - _PERIOD_START[p])
            shift[(match_id, p)] = total
    out = shots.copy()
    out["t"] = [m + shift.get((mid, p), 0.0) for m, mid, p
                in zip(out["minute"], out["match_id"], out["period"])]
    return out


def shots_with_xg(events: pd.DataFrame, penalty_xg: float, model=None) -> pd.DataFrame:
    """All in-match shots (shootouts excluded) with an `xg` column.

    Open-play/set-piece shots use the saved xG model; penalties use
    `penalty_xg`. NOTE: the saved model was fitted on all shots, including these,
    so xG here is descriptive, not an out-of-sample evaluation.
    """
    if model is None:
        model = joblib.load(resolve(load_config()["paths"]["results"]) / "xg_model.joblib")
    base = build_shot_dataset(events)
    base["xg"] = predict_xg(base, model=model)
    base["is_penalty"] = False
    pens = _penalties(events)
    pens["x"] = pens["location"].map(lambda v: v[0])
    pens["y"] = pens["location"].map(lambda v: v[1])
    pens["xg"] = penalty_xg
    pens["sb_xg"] = pens["shot_statsbomb_xg"]   # StatsBomb's own value: benchmark only
    pens["is_penalty"] = True
    pens["minute"] = pens["minute"].astype(float)
    cols = ["match_id", "team", "player", "period", "minute", "index", "x", "y",
            "is_goal", "xg", "sb_xg", "is_penalty"]
    out = pd.concat([base[cols], pens[cols]], ignore_index=True)
    out = out.sort_values(["match_id", "index"]).reset_index(drop=True)
    return add_match_time(out, events)


# ---------- player summary ----------

def is_progressive(passes: pd.DataFrame) -> pd.Series:
    """Boolean mask of progressive passes (definition at top of module)."""
    sx, sy = passes["location"].str[0], passes["location"].str[1]
    ex, ey = passes["pass_end_location"].str[0], passes["pass_end_location"].str[1]
    gain = shot_distance(sx, sy) - shot_distance(ex, ey)
    complete = passes["pass_outcome"].isna()
    allowed = ~passes["pass_type"].isin(NON_PROGRESSIVE_TYPES)
    return complete & allowed & (gain >= PROGRESSIVE_MIN_GAIN)


def is_key_pass(passes: pd.DataFrame) -> pd.Series:
    """A pass that directly led to a shot (`pass_shot_assist`) or a goal (`pass_goal_assist`)."""
    return passes["pass_shot_assist"].fillna(False).astype(bool) | \
        passes["pass_goal_assist"].fillna(False).astype(bool)


def player_summary(events: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per player: shots, goals, xG, goals - xG, passing counts.

    `shots` must come from `shots_with_xg` for the same events.
    """
    passes = events[events["type"] == "Pass"].copy()
    passes["completed"] = passes["pass_outcome"].isna()
    passes["progressive"] = is_progressive(passes)
    passes["key_pass"] = is_key_pass(passes)
    pg = passes.groupby(["player", "team"]).agg(
        passes=("completed", "size"), passes_completed=("completed", "sum"),
        progressive_passes=("progressive", "sum"), key_passes=("key_pass", "sum"))
    sg = shots.groupby(["player", "team"]).agg(
        shots=("is_goal", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
    out = pg.join(sg, how="outer").fillna(0)
    out["goals_minus_xg"] = out["goals"] - out["xg"]
    ints = ["passes", "passes_completed", "progressive_passes", "key_passes", "shots", "goals"]
    out[ints] = out[ints].astype(int)
    out["xg"] = out["xg"].round(3)
    out["goals_minus_xg"] = out["goals_minus_xg"].round(3)
    return out.reset_index().sort_values("xg", ascending=False).reset_index(drop=True)


# ---------- pass network ----------

def first_sub_index(match_events: pd.DataFrame, team: str) -> float:
    """Event index of the team's first substitution (inf if none)."""
    subs = match_events[(match_events["type"] == "Substitution") & (match_events["team"] == team)]
    return float(subs["index"].min()) if len(subs) else float("inf")


def pass_network(match_events: pd.DataFrame, team: str,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    """Nodes, edges and first-sub minute for one team in one match.

    Only completed passes before the team's first substitution are used, so the
    same eleven players appear. Node position = mean pass-origin location of the
    player; node size = passes made. Edge weight = completed passes between the
    pair in EITHER direction (an undirected graph, easier to read and to feed
    into centrality).
    """
    ev = match_events.sort_values("index")
    cutoff = first_sub_index(ev, team)
    p = ev[(ev["type"] == "Pass") & (ev["team"] == team) & (ev["index"] < cutoff)
           & ev["pass_outcome"].isna() & ev["pass_recipient"].notna()].copy()
    p["x"] = p["location"].str[0]
    p["y"] = p["location"].str[1]
    nodes = p.groupby("player").agg(x=("x", "mean"), y=("y", "mean"),
                                    passes=("x", "size")).reset_index()
    # sort each (passer, recipient) pair so A->B and B->A fall in the same edge
    p["a"] = np.where(p["player"] < p["pass_recipient"], p["player"], p["pass_recipient"])
    p["b"] = np.where(p["player"] < p["pass_recipient"], p["pass_recipient"], p["player"])
    edges = p[p["a"] != p["b"]].groupby(["a", "b"]).size().rename("weight").reset_index()
    if np.isfinite(cutoff):
        minute = int(ev.loc[ev["index"] == cutoff, "minute"].iloc[0])
    else:
        minute = None
    return nodes, edges, minute


def network_metrics(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Centrality table for a pass network.

    degree      number of distinct passing partners
    strength    total passes involving the player (weighted degree)
    betweenness how often the player lies on shortest passing routes between
                teammates; edge length = 1 / weight, so frequent partners are "close"
    """
    g = nx.Graph()
    g.add_nodes_from(nodes["player"])
    for r in edges.itertuples():
        g.add_edge(r.a, r.b, weight=r.weight, distance=1.0 / r.weight)
    table = pd.DataFrame({
        "player": list(g.nodes),
        "degree": [g.degree(n) for n in g.nodes],
        "strength": [g.degree(n, weight="weight") for n in g.nodes],
    })
    bc = nx.betweenness_centrality(g, weight="distance", normalized=True)
    table["betweenness"] = table["player"].map(bc).round(4)
    return table.sort_values(["betweenness", "strength"], ascending=False).reset_index(drop=True)


# ---------- touches ----------

def touches(events: pd.DataFrame, team: str | None = None,
            player: str | None = None) -> pd.DataFrame:
    """On-ball events with a location (x, y); failed receipts are dropped."""
    e = events[events["type"].isin(TOUCH_TYPES) & events["location"].notna()]
    e = e[e["ball_receipt_outcome"].isna()]  # 'Incomplete' receipt = ball never arrived
    if team is not None:
        e = e[e["team"] == team]
    if player is not None:
        e = e[e["player"] == player]
    return pd.DataFrame({"x": e["location"].str[0], "y": e["location"].str[1]}).reset_index(drop=True)
