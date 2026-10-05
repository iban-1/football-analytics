"""Pitch drawing helper (thin wrapper around mplsoccer).

We always use the 'statsbomb' pitch type so plotted coordinates match the
raw data (120 x 80, y downward) with no manual conversion.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch


def draw_pitch(
    ax: plt.Axes | None = None, vertical: bool = False, half: bool = False,
    figsize: tuple[float, float] = (10, 6.5),
) -> tuple[plt.Figure, plt.Axes, Pitch]:
    """Draw a StatsBomb-coordinate pitch; returns (fig, ax, pitch).

    Use the returned `pitch` object to plot (pitch.scatter, pitch.arrows, ...)
    so that orientation is handled consistently.
    """
    cls = VerticalPitch if vertical else Pitch
    # line_zorder=2 keeps pitch markings visible on top of heatmaps
    pitch = cls(pitch_type="statsbomb", line_color="#555555", pitch_color="white",
                half=half, line_zorder=2)
    if ax is None:
        fig, ax = pitch.draw(figsize=figsize)
    else:
        pitch.draw(ax=ax)
        fig = ax.figure
    return fig, ax, pitch
