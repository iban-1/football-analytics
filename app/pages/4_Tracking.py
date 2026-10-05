"""Tracking: speed/distance table, team shape over time, frame viewer."""
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common
from src.data.metrica import load_game
from src.viz.tracking_plots import frame_plot, shape_timeline

common.setup_page("Tracking")
R = common.RESULTS / "phase4"
import json

game_no = st.radio("Which match? (Metrica sample game)", [1, 2], horizontal=True,
                   format_func=lambda g: f"Sample Game {g}")
_q = R / f"quality_game{game_no}.json"
if _q.exists():
    q = json.loads(_q.read_text())
    st.markdown(f"**Selected: Sample Game {game_no}**, an anonymised real match (teams unknown), "
                f"{q['duration_min']:.0f} minutes of tracking at {q['fps']:.0f} frames per second, "
                f"{q['frames']:,} frames in total.")


@st.cache_resource(show_spinner="Downloading and cleaning tracking data (first time only)…",
                   max_entries=1)
def get_game(n: int):
    return load_game(n)


with st.expander("What am I looking at? (read this first)", expanded=True):
    st.markdown(f"""
**What are the "sample games"?** Metrica Sports (a sports-data company) published tracking data
from two real matches for people to practise on. A tracking system records where **every player
and the ball are, 25 times a second**. The data is **anonymised**: there are no team or player
names, only **Home** and **Away** and shirt-style IDs like `Player5`. We do not know which teams
or players these are. You are viewing **Sample Game {game_no}**; choose the other game with the
buttons above. (Do not confuse it with the StatsBomb matches on the other pages, which are a
different dataset.)

**The three sections below:**
1. **Distance and speed per player**: a table of how much each player ran. `minutes` is time on
   the pitch, `distance_m` the metres covered, `top_speed_kmh` the fastest speed, `sprints` the
   number of runs of 25.2 km/h or more held for 1 second. The `dist_...` columns split the
   distance by speed (walking, jogging, running, high_speed, sprinting). `is_gk` ticked = the
   goalkeeper. Substitutes show fewer minutes.
2. **Team shape over time**: how stretched out each team is. *Hull area* is the size of the
   shape around the outfield players, *length* front-to-back, *width* side-to-side. Teams
   usually spread out when they have the ball and compress when defending.
3. **Frame viewer**: a snapshot of the match. Blue dots = home, orange = away, numbers = the
   player ID, white dot = ball. Drag the slider to move through the game, or press Play.

The numbers come from cleaned data: raw glitches (players "teleporting") were removed first.
""")

st.subheader("Distance and speed per player")
st.caption("Speed and acceleration come from a Savitzky-Golay smoothing filter, not raw "
           "frame-to-frame differences (which are noisy). Sprint = at least 7 m/s (25.2 km/h) "
           "held for 1 s. Home attacks left to right in this data; players are anonymised.")
path = R / f"physical_game{game_no}.csv"
if path.exists():
    phys = pd.read_csv(path)
    team = st.radio("Team", ["home", "away"], horizontal=True)
    show = phys[phys["team"] == team].drop(columns="team")
    st.dataframe(show, hide_index=True)
else:
    st.info("Run `python -m scripts.phase4_tracking` first.")

st.subheader("Team shape over time")
shape_files = {t: R / f"shape_game{game_no}_{t}_1hz.csv" for t in ("home", "away")}
if all(p.exists() for p in shape_files.values()):
    shapes = {t: pd.read_csv(p, index_col=0) for t, p in shape_files.items()}
    metric = st.selectbox("Metric", ["hull_area", "length", "width", "centroid_x"],
                          format_func=lambda s: {"hull_area": "Convex hull area (m²)",
                                                 "length": "Length (m)", "width": "Width (m)",
                                                 "centroid_x": "Centroid depth x (m)"}[s])
    fig = shape_timeline(shapes, metric, metric, smooth_s=60, fps=1.0, step=1)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Outfield players only, 60-second rolling mean. Wider/longer shape and a larger "
               "hull usually mean the team has the ball.")

st.subheader("Frame viewer")
game = get_game(game_no)
n = len(game.meta)
frames = game.meta.index
step = 25  # slider moves one second at a time
idx = st.slider("Time (seconds into the recording)", 0, (n - 1) // step, 90)
frame = int(frames[idx * step])
holder = st.empty()


def draw(f: int) -> None:
    fig = frame_plot(game, f)
    holder.pyplot(fig)
    plt.close(fig)


draw(frame)
if st.button("▶ Play next 10 seconds"):
    for f in frames[idx * step: idx * step + 10 * 25 + 1: 5]:
        draw(int(f))
        time.sleep(0.05)
ball_missing = game.quality["home"]["ball_nan_frames"] / n
st.caption(f"Blue = home, orange = away, white dot = ball (the ball is missing in "
           f"{ball_missing:.0%} of this game's frames). Players off the pitch are not drawn.")
common.metrica_attribution()
