"""Video: the annotated broadcast clip (player boxes, IDs, team colours, trails)."""
import streamlit as st

import common

common.setup_page("Video analysis")
V = common.RESULTS / "phase7"
video = V / "annotated_web.mp4"

st.caption("YOLO player and ball detection with BoT-SORT tracking on a broadcast clip. "
           "Boxes: light blue = light kit, red = dark kit, yellow = other (referees, "
           "goalkeepers, unclear). The number is the track ID; IDs restart at every camera cut "
           "and often change for the same player (see the failure cases in the README). "
           "White lines are trails; in the pitch-mapped shot (~frames 1521-1721, about 61-69 s) "
           "yellow trails follow the grass as the camera pans. The caption at the bottom says "
           "which shots were analysed.")

if video.exists():
    st.video(str(video))
else:
    st.info("No video found. Put your clip in data/video/, then run "
            "`python -m scripts.phase7_video` and `python -m scripts.make_web_videos`.")

st.warning("This video contains broadcast footage, so it is shown only on your own computer. "
           "It is not part of the repository and must not be put on a public website.")

with st.expander("Per-shot tracking numbers (proxy metrics, not accuracy)"):
    import pandas as pd
    for name, label in (("shots.csv", "Shots found in the clip"),
                        ("tracker_comparison.csv", "ByteTrack vs BoT-SORT on the mapped shot"),
                        ("tracking_quality.csv", "Tracking proxies per tracked shot")):
        if (V / name).exists():
            st.markdown(f"**{label}**")
            st.dataframe(pd.read_csv(V / name), hide_index=True)
