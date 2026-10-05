"""Parse Metrica Sports sample games 1 and 2 (CSV) into clean DataFrames.

Raw format (verified from the files):
  * Tracking CSV per team: 3 header rows (team, shirt number, column label), then
    one row per frame: Period, Frame, Time [s], then an x,y pair per squad member
    and finally the ball. 25 frames per second (0.04 s). Coordinates are
    normalised 0-1 with (0,0) = top-left corner of the pitch. NaN = player not
    on the pitch. Files list the whole squad, so substitutes are NaN until they enter.
  * Events CSV: one row per event with Start/End Frame and normalised X/Y.

Our clean convention (metres):
  * x runs 0 -> 105 along the pitch, y runs 0 -> 68 across it, ORIGIN BOTTOM-LEFT
    (y is flipped from Metrica's top-left origin so plots look the usual way).
  * Pitch size is ASSUMED to be 105 x 68 m (the Metrica README states these).
  * Directions are normalised so the HOME team attacks toward +x (right) in every
    period and the AWAY team toward -x. Raw data swaps ends at half-time, and
    which team starts on which side differs between games.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metrica_download import download_game
from src.utils.config import load_config

FPS_EXPECTED = 25.0


@dataclass
class TrackingGame:
    """One cleaned game. `home`/`away`/`ball` are indexed by frame number."""
    game: int
    fps: float
    pitch_length: float
    pitch_width: float
    meta: pd.DataFrame            # index frame; columns period, time_s
    home: pd.DataFrame            # columns "PlayerN_x", "PlayerN_y" (metres)
    away: pd.DataFrame
    ball: pd.DataFrame            # columns x, y (metres)
    events: pd.DataFrame
    flipped_periods: list[int] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def team(self, name: str) -> pd.DataFrame:
        return self.home if name == "home" else self.away


# ---------- raw readers ----------

def read_tracking_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one team's raw tracking file -> (meta, wide positions incl. 'Ball_x/y').

    Positions stay normalised here (conversion happens in `convert_to_metres`).
    """
    header = pd.read_csv(path, header=None, nrows=3, dtype=str)
    labels = header.iloc[2].tolist()
    names, current = [], None
    for lab in labels[3:]:
        if isinstance(lab, str) and lab:           # x column carries the name
            current = lab
            names.append(f"{current}_x")
        else:                                      # unnamed column right after = y
            names.append(f"{current}_y")
    data = pd.read_csv(path, header=None, skiprows=3)
    data.columns = ["period", "frame", "time_s"] + names
    data = data.set_index("frame")
    meta = data[["period", "time_s"]].astype({"period": int})
    pos = data[names].astype(float)
    return meta, pos


def read_events_csv(path: Path) -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev = ev.rename(columns={"Team": "team", "Type": "type", "Subtype": "subtype",
                            "Period": "period", "Start Frame": "start_frame",
                            "Start Time [s]": "start_s", "End Frame": "end_frame",
                            "End Time [s]": "end_s", "From": "player_from", "To": "player_to",
                            "Start X": "start_x", "Start Y": "start_y",
                            "End X": "end_x", "End Y": "end_y"})
    ev["team"] = ev["team"].str.lower()
    return ev


# ---------- conversion and direction ----------

def convert_to_metres(pos: pd.DataFrame, length: float, width: float) -> pd.DataFrame:
    """Normalised (origin top-left) -> metres (origin bottom-left)."""
    out = pos.copy()
    for c in out.columns:
        out[c] = out[c] * length if c.endswith("_x") else (1.0 - out[c]) * width
    return out


def attack_direction_at_kickoff(team_pos: pd.DataFrame, meta: pd.DataFrame,
                                period: int, length: float) -> int:
    """+1 if the team starts the period in the left half (so attacks right), else -1.

    At a kick-off every team stands in its own half, so the mean x of the players
    on the pitch at the first frame says which end they defend.
    """
    first = meta.index[meta["period"] == period][0]
    xs = team_pos.loc[first, [c for c in team_pos.columns
                              if c.endswith("_x") and not c.startswith("Ball")]]
    return 1 if np.nanmean(xs.to_numpy(float)) < length / 2 else -1


def flip_frames(df: pd.DataFrame, mask: np.ndarray, length: float, width: float) -> None:
    """Rotate positions 180 degrees in place for the rows selected by `mask`."""
    for c in df.columns:
        if c.endswith("_x"):
            df.loc[mask, c] = length - df.loc[mask, c]
        elif c.endswith("_y"):
            df.loc[mask, c] = width - df.loc[mask, c]


# ---------- quality report ----------

def quality_report(raw_home: pd.DataFrame, raw_away: pd.DataFrame, meta: pd.DataFrame,
                   fps: float, length: float, width: float) -> dict:
    """Facts about the RAW (pre-cleaning, in-metres) data, all computed, none assumed."""
    rep: dict = {"frames": int(len(meta)), "duration_min": round(len(meta) / fps / 60, 2),
                 "fps": fps,
                 "frame_gaps": int((meta.index.to_series().diff().dropna() != 1).sum())}
    for name, df in (("home", raw_home), ("away", raw_away)):
        players = sorted({c[:-2] for c in df.columns if c.startswith("Player")})
        xy = df[[c for c in df.columns if c.startswith("Player")]]
        on_pitch = pd.DataFrame({p: df[f"{p}_x"].notna() for p in players})
        px = df[[f"{p}_x" for p in players]].to_numpy()
        py = df[[f"{p}_y" for p in players]].to_numpy()
        outside = ((px < 0) | (px > length) | (py < 0) | (py > width))
        rep[name] = {
            "squad_columns": len(players),
            "max_players_on_pitch": int(on_pitch.sum(axis=1).max()),
            "frames_with_fewer_than_11": int((on_pitch.sum(axis=1) < 11).sum()),
            "nan_share_squad_cells": round(float(xy.isna().to_numpy().mean()), 4),
            "share_positions_outside_pitch": round(float(np.nansum(outside) / np.isfinite(px).sum()), 5),
            "ball_nan_frames": int(df["Ball_x"].isna().sum()),
            "frozen_player_rows": int((xy.diff().abs().sum(axis=1) == 0).sum()),
        }
        # raw (unsmoothed) frame-to-frame speed: how many physically impossible values?
        step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * fps
        rep[name]["raw_speed_over_12ms"] = int(np.nansum(step > 12.0))
        rep[name]["raw_speed_max_ms"] = round(float(np.nanmax(step)), 2)
    return rep


# ---------- spike removal ----------

DESPIKE_WINDOW = 17      # frames (0.68 s) for the rolling-median reference
DESPIKE_MAX_DEV_M = 2.0  # metres from the local median before a point is "a glitch"
DESPIKE_MAX_GAP = 25     # frames (1 s): longest removed stretch we repair by interpolation


MAX_PLAUSIBLE_SPEED = 12.0  # m/s; faster than any footballer (world-record sprint ~12.4)


def _repair(x: np.ndarray, y: np.ndarray, bad: np.ndarray, fps: float,
            max_gap: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Blank the flagged points, then bridge each gap by straight-line interpolation IF
    (a) the gap is at most `max_gap` frames, (b) both neighbours are real data, and
    (c) the implied speed between the neighbours is physically plausible.

    Rule (c) matters: if the track permanently jumped 30 m, bridging the gap would
    draw a fake 150 m/s run. Such gaps are left as NaN and act as breaks in the track.
    Returns (x, y, number of points left missing).
    """
    x, y = x.copy(), y.copy()
    x[bad], y[bad] = np.nan, np.nan
    left = 0
    edges = np.diff(np.r_[0, bad.astype(int), 0])
    for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):  # [s, e)
        a, b = s - 1, e
        ok = (a >= 0 and b < len(x) and (e - s) <= max_gap
              and np.isfinite(x[a]) and np.isfinite(x[b]))
        if ok and np.hypot(x[b] - x[a], y[b] - y[a]) <= MAX_PLAUSIBLE_SPEED * (b - a) / fps:
            t = np.arange(s, e)
            x[s:e] = np.interp(t, [a, b], [x[a], x[b]])
            y[s:e] = np.interp(t, [a, b], [y[a], y[b]])
        else:
            left += e - s
    return x, y, left


def despike(team: pd.DataFrame, meta: pd.DataFrame, fps: float = FPS_EXPECTED,
            ) -> tuple[pd.DataFrame, dict]:
    """Remove tracking glitches (one-off jumps) and repair short gaps.

    A point is a glitch if it lies more than 2 m from the rolling median of that
    player's own track. A real player cannot be 2 m away from where he was
    0.3 s before and after in a straight line, so this keeps genuine sprints
    but removes sensor jumps. Periods are processed separately because the
    half-time break is a genuine discontinuity (kick-off positions).
    """
    out = team.copy()
    players = sorted({c[:-2] for c in team.columns})
    n_flagged = n_fast = n_left = 0
    for period in meta["period"].unique():
        idx = meta.index[meta["period"] == period]
        sub = team.loc[idx]
        med = sub.rolling(DESPIKE_WINDOW, center=True, min_periods=5).median()
        for p in players:
            x, y = sub[f"{p}_x"].to_numpy(float), sub[f"{p}_y"].to_numpy(float)
            # Pass A: points far from the player's own local median (isolated glitches).
            dev = np.hypot(x - med[f"{p}_x"].to_numpy(float), y - med[f"{p}_y"].to_numpy(float))
            bad = np.nan_to_num(dev, nan=0.0) > DESPIKE_MAX_DEV_M
            n_flagged += int(bad.sum())
            x, y, left = _repair(x, y, bad, fps, DESPIKE_MAX_GAP)
            n_left += left
            # Pass B: every frame touching a physically impossible step (> 12 m/s). The
            # provider appears to fill lost-tracking stretches with straight-line ramps
            # (constant huge steps), so those frames are blanked, and bridged only if
            # the endpoints are plausible.
            for _ in range(2):
                step = np.hypot(np.diff(x), np.diff(y)) * fps
                fast = np.nan_to_num(step, nan=0.0) > MAX_PLAUSIBLE_SPEED
                bad = np.r_[fast, False] | np.r_[False, fast]
                if not bad.any():
                    break
                n_fast += int(bad.sum())
                x, y, left = _repair(x, y, bad, fps, DESPIKE_MAX_GAP)
                n_left += left
            out.loc[idx, f"{p}_x"], out.loc[idx, f"{p}_y"] = x, y
    return out, {"glitch_points_flagged": n_flagged, "impossible_step_points_blanked": n_fast,
                 "points_left_missing_after_repair_attempts": int(n_left)}


# ---------- main loader ----------

def load_game(game: int, clean: bool = True) -> TrackingGame:
    """Download (if needed), parse, convert to metres, normalise direction, despike."""
    cfg = load_config()["pitch"]
    length, width = float(cfg["metrica_length_m"]), float(cfg["metrica_width_m"])
    paths = download_game(game)
    meta_h, pos_h = read_tracking_csv(paths["home"])
    meta_a, pos_a = read_tracking_csv(paths["away"])
    assert meta_h.equals(meta_a), "home/away files must share frames and periods"
    fps = round(1.0 / float(meta_h["time_s"].diff().median()), 3)

    home, away = convert_to_metres(pos_h, length, width), convert_to_metres(pos_a, length, width)
    qual = quality_report(home, away, meta_h, fps, length, width)

    events = read_events_csv(paths["events"])
    for c, kind in (("start_x", "x"), ("end_x", "x"), ("start_y", "y"), ("end_y", "y")):
        events[c] = events[c] * length if kind == "x" else (1 - events[c]) * width

    flipped = []
    for period in sorted(meta_h["period"].unique()):
        if attack_direction_at_kickoff(home, meta_h, period, length) == -1:
            mask = (meta_h["period"] == period).to_numpy()
            flip_frames(home, mask, length, width)
            flip_frames(away, mask, length, width)
            emask = (events["period"] == period).to_numpy()
            events.loc[emask, ["start_x", "end_x"]] = length - events.loc[emask, ["start_x", "end_x"]]
            events.loc[emask, ["start_y", "end_y"]] = width - events.loc[emask, ["start_y", "end_y"]]
            flipped.append(int(period))
    qual["periods_flipped_to_make_home_attack_right"] = flipped

    ball = home[["Ball_x", "Ball_y"]].rename(columns={"Ball_x": "x", "Ball_y": "y"})
    ball_away = away[["Ball_x", "Ball_y"]].rename(columns={"Ball_x": "x", "Ball_y": "y"})
    qual["ball_columns_identical_in_both_files"] = bool(
        np.allclose(ball.fillna(-1).to_numpy(), ball_away.fillna(-1).to_numpy()))
    home = home.drop(columns=["Ball_x", "Ball_y"])
    away = away.drop(columns=["Ball_x", "Ball_y"])
    if clean:
        home, qh = despike(home, meta_h, fps)
        away, qa = despike(away, meta_h, fps)
        qual["despike"] = {"home": qh, "away": qa}
        for name, df in (("home", home), ("away", away)):
            px = df.filter(like="_x").to_numpy()
            py = df.filter(like="_y").to_numpy()
            same_period = (meta_h["period"].to_numpy()[1:] == meta_h["period"].to_numpy()[:-1])
            step = np.hypot(np.diff(px, axis=0), np.diff(py, axis=0)) * fps
            step[~same_period] = np.nan  # half-time jump is not a glitch
            qual[name]["raw_speed_over_12ms_after_despike"] = int(np.nansum(step > 12.0))
    return TrackingGame(game, fps, length, width, meta_h, home, away, ball, events,
                        flipped, qual)


def long_format(game: TrackingGame, step: int = 1) -> pd.DataFrame:
    """Tidy table: frame, period, time_s, team, player, x, y (metres). `step` thins frames."""
    parts = []
    for team in ("home", "away"):
        df = game.team(team).iloc[::step]
        for p in sorted({c[:-2] for c in df.columns}):
            part = pd.DataFrame({"x": df[f"{p}_x"], "y": df[f"{p}_y"]}).dropna()
            part["player"], part["team"] = p, team
            parts.append(part.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out.merge(game.meta.reset_index(), on="frame")
