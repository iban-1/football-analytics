"""Team Analysis: pass network and touch heatmap."""
import matplotlib.pyplot as plt
import streamlit as st

import common
from src.analysis.events import network_metrics, pass_network, touches
from src.viz.match_plots import COLORS, heatmap, pass_network_plot

common.setup_page("Team Analysis")
m = common.match_selector()
teams = [m["home_team"], m["away_team"]]
team = st.sidebar.radio("Team", teams)
color = COLORS[teams.index(team)]
ev = common.match_events(int(m["match_id"]))

nodes, edges, minute = pass_network(ev, team)
st.subheader("Pass network")
st.caption(f"Completed passes before {team}'s first substitution "
           f"({'minute ' + str(minute) if minute is not None else 'none made'}). "
           "Node = average pass-origin position, size = passes made; edge width = passes "
           "between the pair in either direction.")
min_edge = st.slider("Hide pairs with fewer passes than", 1, 8, 3)
fig = pass_network_plot(nodes, edges, color, min_edge=min_edge,
                        title=f"{team} - {m['home_team']} {m['home_score']}-{m['away_score']} "
                              f"{m['away_team']}", nicknames=common.nicknames(int(m["match_id"])))
st.pyplot(fig)
plt.close(fig)

st.subheader("Key players (graph metrics)")
st.caption("degree = distinct passing partners; strength = total passes involving the player; "
           "betweenness = how often the player sits on the shortest passing routes between "
           "teammates (edge length = 1 / passes).")
st.dataframe(network_metrics(nodes, edges), hide_index=True)

st.subheader("Touch heatmap")
fig = heatmap(touches(ev, team=team), f"{team} touches")
st.pyplot(fig)
plt.close(fig)
common.statsbomb_attribution()
