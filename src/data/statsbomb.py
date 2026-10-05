"""StatsBomb open-data access with a local on-disk cache.

Only the free open-data mode of `statsbombpy` is used (no credentials).
Everything downloaded is cached under data/raw/statsbomb so a second run
needs no network access.

Security note: the cache uses pickle (keeps list/dict columns such as
freeze frames intact). It is safe here because only this module writes
these files, from data it just downloaded. Never unpickle files you did not
create yourself.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsbombpy import sb

from src.utils.config import load_config, resolve

warnings.filterwarnings("ignore", message=".*credentials.*")


def _cache_dir() -> Path:
    d = resolve(load_config()["paths"]["raw"]) / "statsbomb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_competitions() -> pd.DataFrame:
    """All competition/season pairs available in the open data."""
    path = _cache_dir() / "competitions.pkl"
    if path.exists():
        return pd.read_pickle(path)
    comps = sb.competitions()
    comps.to_pickle(path)
    return comps


def load_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Match list for one competition-season (cached)."""
    path = _cache_dir() / f"matches_{competition_id}_{season_id}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches.to_pickle(path)
    return matches


# statsbombpy only creates a column if at least one event in the loaded match uses it
# (e.g. a match with no through balls has no `pass_through_ball`). Every column our
# code reads is therefore forced to exist, so single matches behave like the full set.
REQUIRED_EVENT_COLUMNS = (
    "type", "team", "player", "period", "minute", "index", "id", "location", "play_pattern",
    "under_pressure", "ball_receipt_outcome", "pass_recipient", "pass_outcome", "pass_type",
    "pass_end_location", "pass_through_ball", "pass_cross", "pass_shot_assist",
    "pass_goal_assist", "shot_outcome", "shot_type", "shot_body_part", "shot_first_time",
    "shot_key_pass_id", "shot_freeze_frame", "shot_statsbomb_xg", "substitution_replacement",
)


def ensure_event_columns(events: pd.DataFrame) -> pd.DataFrame:
    """Add any missing required column, filled with NaN."""
    missing = [c for c in REQUIRED_EVENT_COLUMNS if c not in events.columns]
    for c in missing:
        events[c] = np.nan
    return events


def load_events(match_id: int) -> pd.DataFrame:
    """All events for one match (cached as a pickle: keeps list columns intact)."""
    path = _cache_dir() / f"events_{match_id}.pkl"
    if path.exists():
        return ensure_event_columns(pd.read_pickle(path))
    events = sb.events(match_id=match_id)
    events.to_pickle(path)
    return ensure_event_columns(events)


def load_nicknames(match_id: int) -> dict[str, str]:
    """Map full player name -> display nickname (only players that have one).

    Event data only has long legal names ('Lionel Andrés Messi Cuccittini');
    the lineup file carries the name fans use ('Lionel Messi').
    """
    path = _cache_dir() / f"lineups_{match_id}.pkl"
    if path.exists():
        lineups = pd.read_pickle(path)
    else:
        lineups = sb.lineups(match_id=match_id)
        pd.to_pickle(lineups, path)
    out: dict[str, str] = {}
    for df in lineups.values():
        for r in df.itertuples():
            if isinstance(r.player_nickname, str) and r.player_nickname:
                out[r.player_name] = r.player_nickname
    return out


def configured_competitions() -> list[tuple[int, int]]:
    """(competition_id, season_id) pairs chosen in config.yaml."""
    return [(c["competition_id"], c["season_id"])
            for c in load_config()["statsbomb"]["competitions"]]


def load_all_matches() -> pd.DataFrame:
    """Match lists for every configured competition-season, stacked."""
    frames = [load_matches(c, s).assign(competition_id=c, season_id=s)
              for c, s in configured_competitions()]
    return pd.concat(frames, ignore_index=True)


def load_all_events(matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Events for every configured match, with match/competition ids attached."""
    matches = load_all_matches() if matches is None else matches
    frames = []
    for row in matches.itertuples():
        ev = load_events(row.match_id).copy()
        ev["match_id"] = row.match_id
        ev["competition_id"] = row.competition_id
        ev["season_id"] = row.season_id
        frames.append(ev)
    return pd.concat(frames, ignore_index=True)
