"""Player Analysis: per-player summary and touch heatmap."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common
from src.analysis.events import player_summary, touches
from src.viz.match_plots import heatmap

common.setup_page("Player Analysis")
m = common.match_selector()
ev = common.match_events(int(m["match_id"]))
shots = common.match_shots(int(m["match_id"]))
summary = player_summary(ev, shots)

st.subheader("This match")
st.caption("Progressive pass = completed pass ending at least 10 units closer to the opponent's "
           "goal (goal kicks and kick-offs excluded). Key pass = pass that directly led to a shot.")
st.dataframe(summary, hide_index=True)

player = st.selectbox("Player heatmap", summary.sort_values("passes", ascending=False)["player"])
xy = touches(ev, player=player)
fig = heatmap(xy, player)
st.pyplot(fig)
plt.close(fig)

st.subheader("All five tournaments")
path = common.RESULTS / "phase3" / "player_summary_tournaments.csv"
if path.exists():
    allp = pd.read_csv(path)
    min_shots = st.slider("Minimum shots", 0, 40, 10)
    st.caption("Goals minus xG over a handful of shots is mostly luck, so treat it as "
               "descriptive, not as a measure of finishing skill.")
    st.dataframe(allp[allp["shots"] >= min_shots].sort_values("xg", ascending=False),
                 hide_index=True)
else:
    st.info("Run `python -m scripts.phase3_demo` to create the tournament-wide table.")
common.statsbomb_attribution()
