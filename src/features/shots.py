"""Build the shot dataset and xG features from raw StatsBomb events.

Every feature here must be known at the moment the shot is taken.

LEAKAGE WARNING: columns such as `shot_statsbomb_xg` (StatsBomb's own model
output), `shot_outcome`, `shot_end_location` (where the ball ended up),
`shot_deflected`, `shot_saved_*`, `shot_redirect` describe or depend on what
happened AFTER the shot. Using them would let the model "see the answer", so
they are kept out of the feature matrix. StatsBomb's xG is carried along only
as an external benchmark column (`sb_xg`) and the outcome only as the label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.geometry import in_shot_cone, shot_angle, shot_distance
from src.utils.coords import SB_GOAL_X, SB_GOAL_Y

SHOOTOUT_PERIOD = 5

BASE_FEATURES_NUM = [
    "distance", "angle", "first_time", "follows_through_ball", "follows_cross",
    "under_pressure", "minute", "score_diff",
]
BASE_FEATURES_CAT = ["body_part", "shot_type", "play_pattern"]
FREEZE_FEATURES = [
    "ff_defenders_in_cone", "ff_gk_in_cone", "ff_gk_distance", "ff_nearest_defender",
]


def feature_columns(use_freeze_frame: bool = False) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature names for the chosen feature set."""
    num = BASE_FEATURES_NUM + (FREEZE_FEATURES if use_freeze_frame else [])
    return num, list(BASE_FEATURES_CAT)


# ---------- game state ----------

def goal_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per goal that counts in the match score (shootout kicks excluded).

    A goal is either a Shot with outcome Goal, or an 'Own Goal For' event
    (the beneficiary team's side of an own goal; own goals are not shots).
    """
    shot_goals = events[(events["type"] == "Shot") & (events["shot_outcome"] == "Goal")
                        & (events["period"] != SHOOTOUT_PERIOD)]
    own = events[events["type"] == "Own Goal For"]
    cols = ["match_id", "index", "team"]
    return pd.concat([shot_goals[cols], own[cols]], ignore_index=True)


def add_score_diff(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add `score_diff` = (shooter's goals - opponent's goals) just before the shot.

    Counted from event order (`index`) so a shot never sees its own goal.
    """
    goals = goal_events(events)
    out = np.zeros(len(shots), dtype=int)
    pos = {i: k for k, i in enumerate(shots.index)}
    for match_id, sh in shots.groupby("match_id"):
        g = goals[goals["match_id"] == match_id]
        for idx, row in sh.iterrows():
            before = g[g["index"] < row["index"]]
            own = (before["team"] == row["team"]).sum()
            out[pos[idx]] = int(own - (len(before) - own))
    shots = shots.copy()
    shots["score_diff"] = out
    return shots


# ---------- pre-shot pass context ----------

def key_pass_flags(shots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Flag shots assisted by a through ball or a cross (via `shot_key_pass_id`).

    The pass happens before the shot, so this is legitimate information.
    """
    passes = events.loc[events["type"] == "Pass", ["id", "pass_through_ball", "pass_cross"]]
    passes = passes.set_index("id")
    keys = shots["shot_key_pass_id"]
    through = keys.map(passes["pass_through_ball"]).fillna(False).astype(bool)
    cross = keys.map(passes["pass_cross"]).fillna(False).astype(bool)
    shots = shots.copy()
    shots["follows_through_ball"] = through.astype(int)
    shots["follows_cross"] = cross.astype(int)
    return shots


# ---------- freeze-frame features (optional) ----------

def freeze_frame_features(location: list[float], frame: object) -> dict[str, float]:
    """Defender/goalkeeper context from the freeze frame stored with the shot.

    `frame` is a list of {location, teammate, position} dicts (NaN if missing,
    ~4% of shots). Features:
      ff_defenders_in_cone  outfield opponents inside the shooter-posts triangle
      ff_gk_in_cone         goalkeeper inside that triangle (1/0)
      ff_gk_distance        goalkeeper distance to goal centre
      ff_nearest_defender   distance to the closest outfield opponent
    """
    nan = {k: np.nan for k in FREEZE_FEATURES}
    if not isinstance(frame, list) or not frame:
        return nan
    opp = [p for p in frame if not p["teammate"]]
    gk = [p for p in opp if p["position"]["name"] == "Goalkeeper"]
    field = [p for p in opp if p["position"]["name"] != "Goalkeeper"]
    sx, sy = location
    res = dict(nan)
    if field:
        fx = np.array([p["location"][0] for p in field])
        fy = np.array([p["location"][1] for p in field])
        res["ff_defenders_in_cone"] = float(in_shot_cone(fx, fy, sx, sy).sum())
        res["ff_nearest_defender"] = float(np.min(np.hypot(fx - sx, fy - sy)))
    else:
        res["ff_defenders_in_cone"] = 0.0
    if gk:
        gx, gy = gk[0]["location"]
        res["ff_gk_in_cone"] = float(in_shot_cone(np.array([gx]), np.array([gy]), sx, sy)[0])
        res["ff_gk_distance"] = float(np.hypot(SB_GOAL_X - gx, SB_GOAL_Y - gy))
    return res


# ---------- dataset ----------

def _body_part(v: str) -> str:
    return "Head" if v == "Head" else "Foot" if "Foot" in v else "Other"


def _shot_type(v: str) -> str:
    """Corner (6 shots) is merged into 'Other' - too few to learn from."""
    return v if v in ("Open Play", "Free Kick") else "Other"


def build_shot_dataset(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per modelled shot with features, label and benchmark.

    Exclusions (reported by `exclusion_counts`): penalties (a separate,
    near-constant process) and penalty-shootout kicks (period 5).
    Own goals are not shots in the data, so they cannot appear.
    Free kicks and headers are KEPT, with `shot_type` / `body_part` features
    letting the model give them their own probabilities.
    """
    events = events.sort_values(["match_id", "index"]).reset_index(drop=True)
    shots = events[events["type"] == "Shot"].copy()
    shots = shots[(shots["shot_type"] != "Penalty") & (shots["period"] != SHOOTOUT_PERIOD)]
    shots["x"] = shots["location"].map(lambda v: v[0])
    shots["y"] = shots["location"].map(lambda v: v[1])
    shots["distance"] = shot_distance(shots["x"], shots["y"])
    shots["angle"] = shot_angle(shots["x"], shots["y"])
    shots["body_part"] = shots["shot_body_part"].map(_body_part)
    shots["shot_type"] = shots["shot_type"].map(_shot_type)
    shots["first_time"] = shots["shot_first_time"].fillna(False).astype(int)
    shots["under_pressure"] = shots["under_pressure"].fillna(False).astype(int)
    shots["minute"] = shots["minute"].astype(float)
    shots = key_pass_flags(shots, events)
    shots = add_score_diff(shots, events)
    ff = pd.DataFrame(
        [freeze_frame_features(loc, fr)
         for loc, fr in zip(shots["location"], shots["shot_freeze_frame"])],
        index=shots.index)
    shots = pd.concat([shots, ff], axis=1)
    for optional in ("competition_id", "season_id"):  # absent when one match is passed alone
        if optional not in shots:
            shots[optional] = np.nan
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    shots["sb_xg"] = shots["shot_statsbomb_xg"]  # benchmark only, never a feature
    keep = (["match_id", "competition_id", "season_id", "team", "player", "period",
             "index", "x", "y", "play_pattern"]
            + [c for c in BASE_FEATURES_NUM + BASE_FEATURES_CAT + FREEZE_FEATURES
               if c not in ("play_pattern",)]
            + ["is_goal", "sb_xg"])
    return shots[list(dict.fromkeys(keep))].reset_index(drop=True)


def exclusion_counts(events: pd.DataFrame) -> dict[str, int]:
    """How many shots were left out of the model dataset, and why."""
    shots = events[events["type"] == "Shot"]
    shootout = shots["period"] == SHOOTOUT_PERIOD
    pens = (shots["shot_type"] == "Penalty") & ~shootout
    return {"all_shots": len(shots), "excluded_penalties": int(pens.sum()),
            "excluded_shootout_kicks": int(shootout.sum()),
            "modelled_shots": len(shots) - int(pens.sum()) - int(shootout.sum())}
