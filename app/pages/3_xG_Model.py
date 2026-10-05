"""xG Model: results, calibration, and an interactive shot calculator."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import common
from src.features.geometry import shot_angle, shot_distance
from src.models.xg import predict_xg
from src.viz.pitch import draw_pitch

common.setup_page("xG Model")
R = common.RESULTS

with st.expander("What am I looking at? (read this first)", expanded=True):
    st.markdown("""
**xG (expected goals)** is the chance, from 0 to 1, that a shot becomes a goal. A shot with
xG 0.10 is one that scores about 1 time in 10. The model only looks at what is known at the
moment of the shot: where it is taken from, the angle to the goal, head or foot, the type of
play, whether it was under pressure, the minute and the score.

**This page is not about one match.** It is the model's *report card*. The model learned from
**6,347 shots in 262 matches** (World Cup 2018 and 2022, Euro 2020 and 2024, Copa America 2024)
and was tested on matches it had never seen. To see xG for a single match, use the
**Home** page ("Match Overview").

**The table compares five models** (each row), plus StatsBomb's own xG as a reference:
- `constant_rate`: guesses the same chance for every shot. The "know nothing" baseline.
- `logreg_distance`: uses only the distance to goal.
- `logreg_all`: a simple model using every input. This is the one saved and used in the app.
- `lightgbm` and `mlp`: fancier models. They score about the same as the simple one.
- `statsbomb_xg`: StatsBomb's professional model, shown for comparison.

**The columns** (a "±" number shows how much the score moved between the 5 test groups):
- `log_loss`: how badly the model is surprised by what happened. **Lower is better.** The main score.
- `brier`: average squared error of the probabilities. Lower is better.
- `roc_auc`: how well it ranks goals above non-goals. 0.5 = guessing, 1.0 = perfect. Higher is better.
- `ece`: whether predictions match reality (say 10% shots really score about 10%). Lower is better.

**Why no "accuracy"?** Only about 9 in 100 shots are goals, so always saying "no goal" would be
91% "accurate" and useless.
""")

st.subheader("Check the model on a real match")
st.caption("Pick any match from the five tournaments. The table compares the model's xG with "
           "StatsBomb's xG and with the goals that were really scored.")
c1, c2 = st.columns(2)
m = common.match_selector(container=c1)  # same choice as on the other pages
c2.markdown(f"**Selected:** {m['home_team']} {m['home_score']}-{m['away_score']} "
            f"{m['away_team']}  \n{m['competition']} · {m['competition_stage']} · "
            f"{m['match_date']}")
ms = common.match_shots(int(m["match_id"]))
by_team = ms.groupby("team").agg(shots=("xg", "size"), goals=("is_goal", "sum"),
                                 our_model_xg=("xg", "sum"), statsbomb_xg=("sb_xg", "sum"))
st.dataframe(by_team.round(2).reset_index().rename(columns={"team": "team"}), hide_index=True)
st.caption("`our_model_xg` is this project's model; `statsbomb_xg` is StatsBomb's own. "
           "Penalties count at the observed penalty conversion rate in our column. Over one match "
           "the numbers can differ a lot from the real score: xG is good at describing many shots "
           "on average, and a single match is only a handful of shots and some luck.")
with st.expander("Every shot in this match"):
    st.dataframe(ms[["team", "player", "minute", "x", "y", "xg", "sb_xg", "is_goal", "is_penalty"]]
                 .rename(columns={"xg": "our_model_xg", "sb_xg": "statsbomb_xg", "is_goal": "goal"})
                 .round(3), hide_index=True)

st.subheader("Model comparison (all 262 matches)")
st.caption("5-fold cross-validation grouped by match (no match in both train and test), "
           "mean ± std across folds. Lower log loss / Brier is better; higher ROC AUC is "
           "better. Accuracy is not shown: only ~9% of shots are goals, so it is misleading.")
if (R / "xg_results.csv").exists():
    st.dataframe(pd.read_csv(R / "xg_results.csv"), hide_index=True)
    with st.expander("More: pooled calibration error, paired differences, freeze-frame, "
                     "leave-one-tournament-out"):
        for name, title in [("xg_results_pooled_oof", "Pooled out-of-fold (ECE vs noise floor)"),
                            ("xg_results_paired_differences", "Paired log-loss differences vs "
                             "logreg_all"),
                            ("xg_results_freeze_frame", "With freeze-frame features"),
                            ("xg_results_leave_one_tournament_out", "Leave one tournament out")]:
            if (R / f"{name}.csv").exists():
                st.markdown(f"**{title}**")
                st.dataframe(pd.read_csv(R / f"{name}.csv"), hide_index=True)
else:
    st.info("Run `python -m scripts.reproduce_xg` to create the results tables.")

if (R / "xg_calibration.png").exists():
    st.subheader("Calibration: can the predictions be trusted?")
    st.caption("Each dot is a group of shots the model gave a similar xG. Across: what the model "
               "predicted. Up: how often those shots really scored. Dots on the dashed line mean "
               "the predictions are honest. The first plot (constant_rate) is the do-nothing "
               "baseline.")
    st.image(str(R / "xg_calibration.png"))
if (R / "xg_feature_importance.png").exists():
    st.subheader("What the model pays attention to")
    st.caption("Each row is an input. Dots to the right pushed a shot's xG up, to the left pushed "
               "it down; red = a high value of that input, blue = low. Distance and angle matter "
               "most: shots from close range and with a wide view of the goal are worth more.")
    st.image(str(R / "xg_feature_importance.png"), width=560)

st.subheader("Shot calculator: try your own shot")
st.caption("Not tied to any match. Move the sliders to invent a shot and see the model's xG. "
           "The goal is on the right. Positions use StatsBomb's pitch: x 0-120 (120 = goal line), "
           "y 0-80 (40 = middle).")
a, b = st.columns([1, 1])
with a:
    x = st.slider("x (distance up the pitch)", 60.0, 120.0, 108.0, 0.5)
    y = st.slider("y (across the pitch)", 0.0, 80.0, 40.0, 0.5)
    body = st.selectbox("Body part", ["Foot", "Head", "Other"])
    stype = st.selectbox("Shot type", ["Open Play", "Free Kick", "Other"])
    pattern = st.selectbox("Play pattern", ["Regular Play", "From Corner", "From Free Kick",
                                            "From Throw In", "From Counter", "From Goal Kick",
                                            "From Keeper", "From Kick Off", "Other"])
    first_time = st.checkbox("First-time shot")
    through = st.checkbox("Follows a through ball")
    cross = st.checkbox("Follows a cross")
    pressure = st.checkbox("Under pressure")
    minute = st.slider("Minute", 0, 120, 45)
    score_diff = st.slider("Score difference (shooter's team)", -3, 3, 0)
shot = {"x": x, "y": y, "body_part": body, "shot_type": stype, "play_pattern": pattern,
        "first_time": int(first_time), "follows_through_ball": int(through),
        "follows_cross": int(cross), "under_pressure": int(pressure),
        "minute": float(minute), "score_diff": score_diff}
xg = float(predict_xg(shot, model=common.xg_model())[0])
with b:
    st.metric("Predicted xG", f"{xg:.3f}")
    st.write(f"Distance to goal centre: **{float(shot_distance(x, y)):.1f}**  ·  "
             f"Goal angle: **{float(shot_angle(x, y)):.2f} rad**")
    fig, ax, pitch = draw_pitch(half=True, figsize=(6, 5))
    pitch.scatter([x], [y], s=250, color="#d95f02", edgecolors="black", ax=ax, zorder=3)
    pitch.lines(x, y, 120, 36, ax=ax, color="#999", lw=1)
    pitch.lines(x, y, 120, 44, ax=ax, color="#999", lw=1)
    st.pyplot(fig)
    plt.close(fig)
st.caption("The model uses only information known at the moment of the shot. It does not know "
           "about defenders or the goalkeeper (a freeze-frame variant is evaluated above).")
common.statsbomb_attribution()
