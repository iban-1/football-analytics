"""Tracking-data analysis for Metrica games (Phase 4).

All inputs are `TrackingGame` objects from src.data.metrica: metres, origin
bottom-left, home attacking +x and away attacking -x in every period.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.metrica import TrackingGame

# ---------- settings (named so each choice can be justified) ----------
SMOOTH_WINDOW = 13        # frames = 0.52 s at 25 Hz; odd, as Savitzky-Golay requires
SMOOTH_POLY = 2           # local quadratic: gives both velocity and acceleration
SPRINT_SPEED = 7.0        # m/s (25.2 km/h)
SPRINT_MIN_S = 1.0        # a sprint must be held this long
# speed zones in km/h (common sports-science bands); the top band equals SPRINT_SPEED
ZONE_EDGES_KMH = [0.0, 7.2, 14.4, 19.8, 25.2, np.inf]
ZONE_NAMES = ["walking", "jogging", "running", "high_speed", "sprinting"]


def players_of(team_df: pd.DataFrame) -> list[str]:
    return sorted({c[:-2] for c in team_df.columns},
                  key=lambda s: int(s.replace("Player", "")))


# ---------- velocity and acceleration ----------

def _contiguous_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    edges = np.diff(np.r_[0, valid.astype(int), 0])
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def smooth_kinematics(team_df: pd.DataFrame, meta: pd.DataFrame, fps: float,
                      window: int = SMOOTH_WINDOW, poly: int = SMOOTH_POLY,
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smoothed speed (m/s) and acceleration magnitude (m/s^2) per player and frame.

    WHY SMOOTH: tracking positions carry centimetre-level noise. Speed is a
    difference of positions divided by 0.04 s, which multiplies that noise by 25,
    and acceleration differences it again (by 25 more). Raw differencing therefore
    gives jittery speeds and absurd accelerations. A Savitzky-Golay filter fits a
    small quadratic to each sliding 0.52 s window and reads velocity/acceleration
    off the fitted curve, averaging the noise away while keeping real sprints.

    Runs are processed separately per period and per stretch of valid data, so
    nothing is smoothed across half-time or across a gap. Stretches shorter than
    the window are left NaN, and half a window is trimmed from each stretch end.
    """
    dt = 1.0 / fps
    periods = meta["period"].to_numpy()
    speed = pd.DataFrame(np.nan, index=team_df.index, columns=players_of(team_df))
    accel = speed.copy()
    for p in speed.columns:
        x, y = team_df[f"{p}_x"].to_numpy(float), team_df[f"{p}_y"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        sp, ac = np.full(len(x), np.nan), np.full(len(x), np.nan)
        for per in np.unique(periods):
            pm = periods == per
            offset = np.where(pm)[0][0]
            for s, e in _contiguous_runs(valid[pm]):
                if e - s < window:
                    continue
                sl = slice(offset + s, offset + e)
                vx = savgol_filter(x[sl], window, poly, deriv=1, delta=dt)
                vy = savgol_filter(y[sl], window, poly, deriv=1, delta=dt)
                ax = savgol_filter(x[sl], window, poly, deriv=2, delta=dt)
                ay = savgol_filter(y[sl], window, poly, deriv=2, delta=dt)
                sp_run, ac_run = np.hypot(vx, vy), np.hypot(ax, ay)
                # The filter has one-sided data at the ends of a stretch and extrapolates,
                # which produced speeds up to 20 m/s. Drop half a window at each end.
                h = window // 2
                sp_run[:h], sp_run[-h:] = np.nan, np.nan
                ac_run[:h], ac_run[-h:] = np.nan, np.nan
                sp[sl], ac[sl] = sp_run, ac_run
        speed[p], accel[p] = sp, ac
    return speed, accel


def raw_kinematics(team_df: pd.DataFrame, meta: pd.DataFrame, fps: float,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plain finite-difference speed/acceleration (for the noise comparison only)."""
    same = (meta["period"].to_numpy()[1:] == meta["period"].to_numpy()[:-1])
    speed = pd.DataFrame(np.nan, index=team_df.index, columns=players_of(team_df))
    accel = speed.copy()
    for p in speed.columns:
        x, y = team_df[f"{p}_x"].to_numpy(float), team_df[f"{p}_y"].to_numpy(float)
        sp = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) * fps]
        sp[1:][~same] = np.nan
        ac = np.r_[np.nan, np.abs(np.diff(sp)) * fps]
        speed[p], accel[p] = sp, ac
    return speed, accel


# ---------- physical metrics ----------

def goalkeepers(game: TrackingGame) -> dict[str, str]:
    """The goalkeeper of each team = player whose mean x is closest to his own goal."""
    gk = {}
    for team, pick in (("home", "idxmin"), ("away", "idxmax")):
        mx = game.team(team).filter(like="_x").mean()
        gk[team] = getattr(mx, pick)()[:-2]
    return gk


def _sprint_count(speed: np.ndarray, fps: float) -> int:
    above = np.nan_to_num(speed, nan=0.0) >= SPRINT_SPEED
    min_len = int(round(SPRINT_MIN_S * fps))
    return sum(1 for s, e in _contiguous_runs(above) if e - s >= min_len)


def physical_table(game: TrackingGame, team: str, speed: pd.DataFrame) -> pd.DataFrame:
    """Per-player running load: minutes, distance, top speed, sprints, speed zones.

    Distance = sum of (smoothed speed x frame time) over frames where speed is
    valid. Time with no valid speed (off pitch, gaps) is excluded, so substitutes
    show only the minutes they were on.
    """
    dt = 1.0 / game.fps
    gk = goalkeepers(game)[team]
    rows = []
    for p in speed.columns:
        s = speed[p].to_numpy(float)
        v = np.isfinite(s)
        if not v.any():
            continue
        kmh = s[v] * 3.6
        zone_idx = np.digitize(kmh, ZONE_EDGES_KMH[1:-1])
        zone_dist = [float(np.sum(s[v][zone_idx == z]) * dt) for z in range(len(ZONE_NAMES))]
        rows.append({
            "team": team, "player": p, "is_gk": p == gk,
            "minutes": round(v.sum() * dt / 60, 1),
            "distance_m": round(float(np.sum(s[v]) * dt), 0),
            "top_speed_kmh": round(float(kmh.max()), 1),
            "sprints": _sprint_count(s, game.fps),
            **{f"dist_{n}_m": round(d, 0) for n, d in zip(ZONE_NAMES, zone_dist)},
        })
    return pd.DataFrame(rows)


# ---------- possession ----------

POSSESSION_EVENTS = ("PASS", "SHOT", "RECOVERY", "SET PIECE")


def possession_series(game: TrackingGame) -> pd.Series:
    """Per-frame team in possession: 'home', 'away' or '' (unknown / dead ball).

    Approximation from the event feed: the team of the latest PASS, SHOT, RECOVERY
    or SET PIECE holds possession; BALL OUT and the start of each period reset it
    to unknown. Event timestamps are discrete, so possession during a long pass
    stays with the passer until the receiver's next event.
    """
    idx = game.meta.index
    mark = pd.Series(np.nan, index=idx, dtype=object)
    ev = game.events[game.events["type"].isin(POSSESSION_EVENTS + ("BALL OUT",))]
    for r in ev.sort_values("start_frame").itertuples():   # later events overwrite earlier
        if r.start_frame in idx:
            mark.loc[r.start_frame] = "" if r.type == "BALL OUT" else r.team
    for first in game.meta.groupby("period").head(1).index:  # reset at each period start
        if pd.isna(mark.loc[first]):
            mark.loc[first] = ""
    return mark.ffill().fillna("")


# ---------- team shape ----------

def _hull_area(xy: np.ndarray) -> float:
    try:
        return float(ConvexHull(xy).volume)  # 2-D "volume" = area
    except Exception:
        return float("nan")


def team_shape(game: TrackingGame, team: str, step: int = 5) -> pd.DataFrame:
    """Outfield shape every `step` frames (5 -> 5 Hz): centroid, length, width, hull area.

    Goalkeepers are excluded: a keeper 40 m behind the line would dominate
    "length" and "area". length = max x - min x, width = max y - min y.
    """
    df = game.team(team).iloc[::step]
    gk = goalkeepers(game)[team]
    ps = [p for p in players_of(df) if p != gk]
    x = df[[f"{p}_x" for p in ps]].to_numpy(float)
    y = df[[f"{p}_y" for p in ps]].to_numpy(float)
    n = np.isfinite(x).sum(axis=1)
    out = pd.DataFrame({"n_outfield": n,
                        "centroid_x": np.nanmean(np.where(n[:, None] > 0, x, np.nan), axis=1),
                        "centroid_y": np.nanmean(np.where(n[:, None] > 0, y, np.nan), axis=1),
                        "length": np.nanmax(x, axis=1) - np.nanmin(x, axis=1),
                        "width": np.nanmax(y, axis=1) - np.nanmin(y, axis=1)},
                       index=df.index)
    out["hull_area"] = [
        _hull_area(np.c_[x[i][m], y[i][m]]) if (m := np.isfinite(x[i])).sum() >= 8 else np.nan
        for i in range(len(df))]
    out[out["n_outfield"] < 8] = np.nan
    meta = game.meta.loc[df.index]
    out["period"], out["time_s"] = meta["period"], meta["time_s"]
    return out


def shape_by_possession(shape: pd.DataFrame, poss: pd.Series, team: str) -> pd.DataFrame:
    """Mean shape metrics when the team has the ball vs when the opponent has it."""
    state = poss.reindex(shape.index)
    label = np.where(state == team, "in possession",
                     np.where(state == "", "unknown/dead ball", "out of possession"))
    cols = ["centroid_x", "length", "width", "hull_area"]
    t = shape.assign(state=label).groupby("state")[cols].agg(["mean", "std", "count"])
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t.round(2)


# ---------- formation / role analysis ----------

def attack_relative(df: pd.DataFrame, team: str, length: float, width: float) -> pd.DataFrame:
    """Express positions so the team attacks +x (flip the away team 180 degrees)."""
    if team == "home":
        return df
    out = df.copy()
    for c in out.columns:
        out[c] = length - out[c] if c.endswith("_x") else width - out[c]
    return out


def formation_from_positions(mean_xy: pd.DataFrame, k_range=(2, 3, 4), seed: int = 0,
                             ) -> dict:
    """Cluster 10 outfield average positions into lines by DEPTH (x) with K-means.

    K is the number of lines (defence/midfield/attack ...), chosen by silhouette
    score: the k whose clusters are most compact and best separated. Only x is
    clustered; clustering x AND y would split by flank instead of by line. The
    formation string lists cluster sizes from deepest to most advanced.
    """
    x = mean_xy["x"].to_numpy().reshape(-1, 1)
    scores, models = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        scores[k] = float(silhouette_score(x, km.labels_))
        models[k] = km  # with a single k (fixed-lines variant) it is simply selected
    k = max(scores, key=lambda kk: scores[kk])
    km = models[k]
    order = np.argsort(km.cluster_centers_.ravel())
    rank = {c: i for i, c in enumerate(order)}
    labels = np.array([rank[l] for l in km.labels_])
    sizes = [int((labels == i).sum()) for i in range(k)]
    return {"k": k, "formation": "-".join(map(str, sizes)), "silhouette": scores[k],
            "silhouette_by_k": scores, "labels": labels}


def formation_windows(game: TrackingGame, team: str, minutes: float = 5.0,
                      min_seconds: float = 25.0, seed: int = 0,
                      poss: pd.Series | None = None, k_range: tuple[int, ...] = (2, 3, 4),
                      ) -> tuple[pd.DataFrame, dict]:
    """Formation per 5-minute window, separately in and out of possession.

    A (window, state) is used only if it has at least `min_seconds` of frames and
    exactly 10 outfield players were on the pitch for >= 90% of them (so average
    positions are not distorted by substitutions).
    """
    poss = possession_series(game) if poss is None else poss
    rel = attack_relative(game.team(team), team, game.pitch_length, game.pitch_width)
    gk = goalkeepers(game)[team]
    win = int(minutes * 60 * game.fps)
    rows, positions = [], {}
    for period in sorted(game.meta["period"].unique()):
        pidx = game.meta.index[game.meta["period"] == period]
        for w0 in range(0, len(pidx), win):
            widx = pidx[w0:w0 + win]
            for state, mask in (("in possession", poss.reindex(widx) == team),
                                ("out of possession", (poss.reindex(widx) != team)
                                 & (poss.reindex(widx) != ""))):
                fidx = widx[mask.to_numpy()]
                if len(fidx) < min_seconds * game.fps:
                    continue
                sub = rel.loc[fidx]
                ps = [p for p in players_of(sub) if p != gk
                      and sub[f"{p}_x"].notna().mean() >= 0.9]
                if len(ps) != 10:
                    continue
                mean_xy = pd.DataFrame({"x": [sub[f"{p}_x"].mean() for p in ps],
                                        "y": [sub[f"{p}_y"].mean() for p in ps]}, index=ps)
                res = formation_from_positions(mean_xy, k_range=k_range, seed=seed)
                key = (int(period), round(w0 / game.fps / 60), state)
                positions[key] = mean_xy.assign(line=res["labels"])
                rows.append({"team": team, "period": int(period),
                             "window_start_min": round(w0 / game.fps / 60),
                             "state": state, "frames": len(fidx), "k": res["k"],
                             "formation": res["formation"],
                             "silhouette": round(res["silhouette"], 3),
                             **{f"sil_k{k}": round(v, 3)
                                for k, v in res["silhouette_by_k"].items()}})
    return pd.DataFrame(rows), positions


def formation_reliability(windows: pd.DataFrame) -> pd.DataFrame:
    """How stable is the detected formation? Share of windows per formation string."""
    g = windows.groupby(["team", "state"])["formation"]
    out = g.agg(windows="size", modal_formation=lambda s: s.value_counts().index[0],
                modal_share=lambda s: round(s.value_counts().iloc[0] / len(s), 2),
                distinct_formations="nunique")
    return out.reset_index()


# ---------- pitch control (simple Voronoi) ----------

def positions_at(game: TrackingGame, frame: int) -> pd.DataFrame:
    """Players on the pitch at one frame: team, player, x, y."""
    rows = []
    for team in ("home", "away"):
        r = game.team(team).loc[frame]
        for p in players_of(game.team(team)):
            if np.isfinite(r[f"{p}_x"]):
                rows.append({"team": team, "player": p, "x": r[f"{p}_x"], "y": r[f"{p}_y"]})
    return pd.DataFrame(rows)


def voronoi_control(pos: pd.DataFrame, length: float, width: float,
                    cell: float = 0.5) -> tuple[np.ndarray, float]:
    """Assign each grid cell to the nearest player (simple Voronoi pitch control).

    Returns (grid of 0=home / 1=away, home share of pitch area). This ignores
    velocity and reaction time, so it is a rough picture of who could reach a
    spot first, not a full pitch-control model.
    """
    xs = np.arange(cell / 2, length, cell)
    ys = np.arange(cell / 2, width, cell)
    gx, gy = np.meshgrid(xs, ys)
    _, nearest = cKDTree(pos[["x", "y"]].to_numpy()).query(np.c_[gx.ravel(), gy.ravel()])
    team = (pos["team"].to_numpy()[nearest] == "away").astype(int).reshape(gx.shape)
    return team, float(1.0 - team.mean())
