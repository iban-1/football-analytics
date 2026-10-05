"""Match-level plots: shot map, xG race chart, pass network, touch heatmap."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from src.utils.coords import flip_statsbomb
from src.viz.pitch import draw_pitch

COLORS = ("#1f77b4", "#d95f02")  # blue / orange: distinguishable for colour-blind readers


def short_name(name: str) -> str:
    """'Theo Bernard François Hernández' -> 'T. Hernández' (first initial + last word)."""
    parts = name.split()
    return name if len(parts) == 1 else f"{parts[0][0]}. {parts[-1]}"


def shot_map(shots: pd.DataFrame, teams: tuple[str, str], title: str = "") -> plt.Figure:
    """Both teams on one pitch; team 0 attacks right, team 1 is mirrored to attack left.

    Marker area is proportional to xG. Goals are filled, other shots hollow.
    """
    fig, ax, pitch = draw_pitch()
    for team, color in zip(teams, COLORS):
        s = shots[shots["team"] == team]
        x, y = s["x"].to_numpy(), s["y"].to_numpy()
        if team == teams[1]:
            x, y = flip_statsbomb(x, y)
        area = 40 + 1800 * s["xg"].to_numpy()
        goal = s["is_goal"].to_numpy() == 1
        pitch.scatter(x[~goal], y[~goal], s=area[~goal], ax=ax, facecolors="none",
                      edgecolors=color, linewidths=1.5)
        pitch.scatter(x[goal], y[goal], s=area[goal], ax=ax, color=color, edgecolors="black",
                      linewidths=1.5, alpha=0.9)
    handles = [Line2D([], [], marker="o", ls="", color=c, label=f"{t} (attacks "
                      f"{'right' if i == 0 else 'left'})")
               for i, (t, c) in enumerate(zip(teams, COLORS))]
    handles += [Line2D([], [], marker="o", ls="", mfc="none", mec="grey", label="no goal"),
                Line2D([], [], marker="o", ls="", color="grey", label="goal (filled)")]
    ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, frameon=True,
              bbox_to_anchor=(0.5, -0.12))
    ax.set_title((title + "\n" if title else "") + "Marker area ∝ xG")
    return fig


def xg_race(shots: pd.DataFrame, teams: tuple[str, str], title: str = "") -> plt.Figure:
    """Cumulative xG over match time for both teams, with goals marked."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    end = max(90.0, float(shots["t"].max()) + 1)
    for team, color in zip(teams, COLORS):
        s = shots[shots["team"] == team].sort_values("index")
        cum = s["xg"].cumsum()  # running total over ALL shots (goals are picked out below)
        t = np.concatenate([[0.0], s["t"].to_numpy(), [end]])
        c = np.concatenate([[0.0], cum.to_numpy()])
        c = np.append(c, c[-1])
        ax.step(t, c, where="post", color=color, lw=2,
                label=f"{team}  (xG {c[-1]:.2f})")
        goals = s["is_goal"] == 1
        ax.scatter(s.loc[goals, "t"], cum[goals], color=color, s=80,
                   edgecolors="black", zorder=3)
    ax.set_xlabel("match time (minutes, stoppage time added to its period)")
    ax.set_ylabel("cumulative xG")
    ax.set_title((title + "\n" if title else "") + "Dots mark goals")
    ax.legend(loc="upper left")
    ax.set_xlim(0, end)
    fig.tight_layout()
    return fig


def pass_network_plot(nodes: pd.DataFrame, edges: pd.DataFrame, color: str,
                      min_edge: int = 3, title: str = "",
                      nicknames: dict[str, str] | None = None) -> plt.Figure:
    """Nodes at average pass-origin positions (size = passes made); edge width = pass count.

    Pairs with fewer than `min_edge` passes are hidden to keep the picture readable.
    """
    fig, ax, pitch = draw_pitch()
    pos = nodes.set_index("player")[["x", "y"]]
    e = edges[edges["weight"] >= min_edge]
    for r in e.itertuples():
        if r.a in pos.index and r.b in pos.index:
            pitch.lines(pos.loc[r.a, "x"], pos.loc[r.a, "y"], pos.loc[r.b, "x"],
                        pos.loc[r.b, "y"], lw=0.5 + r.weight * 0.6, color=color, alpha=0.5,
                        ax=ax, zorder=2)
    pitch.scatter(nodes["x"], nodes["y"], s=60 + nodes["passes"] * 12, color=color,
                  edgecolors="black", linewidths=1.2, ax=ax, zorder=3)
    for r in nodes.itertuples():
        label = (nicknames or {}).get(r.player) or short_name(r.player)
        pitch.annotate(label, (r.x, r.y - 4.5), ax=ax, ha="center", va="center",
                       fontsize=8, zorder=4)
    ax.set_title((title + "\n" if title else "") +
                 f"Node size = passes made; edges = pairs with >= {min_edge} passes")
    return fig


def heatmap(xy: pd.DataFrame, title: str = "", sigma: float = 1.5,
            bins: tuple[int, int] = (24, 16)) -> plt.Figure:
    """Smoothed touch density: count touches per grid cell, then Gaussian-blur the grid.

    `sigma` is the blur width in grid cells (1.5 cells ~ 7.5 x 7.5 pitch units).
    Colour shows share of the player's/team's touches, so different sample
    sizes are comparable.
    """
    fig, ax, pitch = draw_pitch()
    stat = pitch.bin_statistic(xy["x"], xy["y"], statistic="count", bins=bins)
    stat["statistic"] = gaussian_filter(stat["statistic"].astype(float), sigma=sigma)
    total = stat["statistic"].sum()
    stat["statistic"] = stat["statistic"] / total if total else stat["statistic"]
    hm = pitch.heatmap(stat, ax=ax, cmap="YlOrRd", edgecolors=None)
    fig.colorbar(hm, ax=ax, shrink=0.6, label="share of touches per cell")
    ax.set_title((title + "\n" if title else "") + f"{len(xy)} touches (attacking left to right)")
    return fig
