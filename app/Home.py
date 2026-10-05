"""Landing page: Match Overview (score, xG timeline, shot map)."""
import matplotlib.pyplot as plt
import streamlit as st

import common
from src.viz.match_plots import shot_map, xg_race

common.setup_page("Match Overview")
m = common.match_selector()
teams = (m["home_team"], m["away_team"])
shots = common.match_shots(int(m["match_id"]))

st.header(f"{teams[0]} {m['home_score']} – {m['away_score']} {teams[1]}")
c = st.columns(2)
for col, t in zip(c, teams):
    s = shots[shots["team"] == t]
    col.metric(f"{t} xG", f"{s['xg'].sum():.2f}", f"{int(s['is_goal'].sum())} goals, "
               f"{len(s)} shots", delta_color="off")
st.caption(f"{m['competition']} · {m['competition_stage']} · {m['match_date']}. "
           "Penalty shoot-outs are excluded. Penalties count at the observed in-match "
           "conversion rate; all other shots use the xG model.")

left, right = st.columns(2)
with left:
    st.subheader("xG timeline")
    fig = xg_race(shots, teams)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Shot map")
    fig = shot_map(shots, teams)
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Shot table"):
    st.dataframe(shots[["team", "player", "minute", "x", "y", "xg", "is_goal", "is_penalty"]]
                 .rename(columns={"is_goal": "goal"}).round(3), hide_index=True)
st.caption("xG here comes from a model fitted on all tournament shots (including this match), "
           "so it is descriptive, not an out-of-sample test.")
common.statsbomb_attribution()
