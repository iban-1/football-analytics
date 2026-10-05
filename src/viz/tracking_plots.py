"""Plots for tracking data (metres, 105 x 68 pitch, home blue / away orange)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from mplsoccer import Pitch

from src.analysis.tracking import players_of, positions_at, voronoi_control
from src.data.metrica import TrackingGame

HOME_C, AWAY_C = "#1f77b4", "#d95f02"


def tracking_pitch(length: float = 105.0, width: float = 68.0, ax=None,
                   figsize=(10, 6.5)) -> tuple[plt.Figure, plt.Axes, Pitch]:
    pitch = Pitch(pitch_type="custom", pitch_length=length, pitch_width=width,
                  line_color="#555555", pitch_color="white", line_zorder=2)
    if ax is None:
        fig, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
        fig = ax.figure
    return fig, ax, pitch


def frame_plot(game: TrackingGame, frame: int, ax=None, title: str | None = None,
               label_players: bool = True) -> plt.Figure:
    """Players (blue = home, orange = away, shirt numbers) and the ball at one frame."""
    fig, ax, pitch = tracking_pitch(game.pitch_length, game.pitch_width, ax)
    pos = positions_at(game, frame)
    for team, color in (("home", HOME_C), ("away", AWAY_C)):
        t = pos[pos["team"] == team]
        pitch.scatter(t["x"], t["y"], s=260, color=color, edgecolors="black", ax=ax, zorder=3)
        if label_players:
            for r in t.itertuples():
                ax.text(r.x, r.y, r.player.replace("Player", ""), ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
    b = game.ball.loc[frame]
    if np.isfinite(b["x"]):
        pitch.scatter(b["x"], b["y"], s=70, color="white", edgecolors="black", ax=ax, zorder=5)
    m = game.meta.loc[frame]
    ax.set_title(title or f"frame {frame}  |  period {m['period']}  |  {m['time_s'] / 60:.1f} min")
    return fig


def shape_timeline(shapes: dict[str, pd.DataFrame], column: str, ylabel: str,
                   smooth_s: float = 60.0, fps: float = 25.0, step: int = 5) -> plt.Figure:
    """Rolling mean of a shape metric over match time for both teams."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for team, color in (("home", HOME_C), ("away", AWAY_C)):
        s = shapes[team]
        window = max(1, int(smooth_s * fps / step))
        for period, part in s.groupby("period"):
            ax.plot(part["time_s"] / 60, part[column].rolling(window, min_periods=1).mean(),
                    color=color, label=team if period == 1 else None)
    ax.set_xlabel("match time (min)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def formation_plot(positions: pd.DataFrame, title: str = "", length: float = 105.0,
                   width: float = 68.0) -> plt.Figure:
    """Average outfield positions (attack-relative, attacking right) coloured by detected line."""
    fig, ax, pitch = tracking_pitch(length, width)
    palette = plt.get_cmap("Set1")
    for line, grp in positions.groupby("line"):
        pitch.scatter(grp["x"], grp["y"], s=300, color=palette(int(line)), edgecolors="black",
                      ax=ax, zorder=3, label=f"line {int(line) + 1} ({len(grp)} players)")
        for p, r in grp.iterrows():
            ax.text(r["x"], r["y"], str(p).replace("Player", ""), ha="center", va="center",
                    fontsize=8, color="white", zorder=4)
    ax.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    ax.set_title(title)
    return fig


def voronoi_plot(game: TrackingGame, frame: int, title: str | None = None) -> tuple[plt.Figure, float]:
    """Nearest-player (Voronoi) control map at one frame; returns (figure, home share)."""
    pos = positions_at(game, frame)
    grid, home_share = voronoi_control(pos, game.pitch_length, game.pitch_width)
    fig, ax, pitch = tracking_pitch(game.pitch_length, game.pitch_width)
    ax.imshow(grid, extent=(0, game.pitch_length, 0, game.pitch_width), origin="lower",
              cmap=ListedColormap([HOME_C, AWAY_C]), alpha=0.28, zorder=1, aspect="auto")
    for team, color in (("home", HOME_C), ("away", AWAY_C)):
        t = pos[pos["team"] == team]
        pitch.scatter(t["x"], t["y"], s=200, color=color, edgecolors="black", ax=ax, zorder=3)
    b = game.ball.loc[frame]
    if np.isfinite(b["x"]):
        pitch.scatter(b["x"], b["y"], s=60, color="white", edgecolors="black", ax=ax, zorder=5)
    m = game.meta.loc[frame]
    ax.set_title(title or f"Nearest-player control, {m['time_s'] / 60:.1f} min: "
                          f"home {home_share:.0%} / away {1 - home_share:.0%} of pitch area")
    return fig, home_share
