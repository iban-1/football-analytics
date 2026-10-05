"""Error analysis: where does the xG model go most wrong?

Works on out-of-fold predictions, so every shot was scored by a model that
never saw its match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def per_shot_log_loss(y: pd.Series, p: pd.Series) -> pd.Series:
    p = p.clip(EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def slice_table(df: pd.DataFrame, by: str, pred_col: str = "pred") -> pd.DataFrame:
    """Per-group count, goals, mean prediction, actual rate and share of total loss.

    `bias` = mean prediction - actual rate (positive = model over-predicts).
    """
    d = df.assign(loss=per_shot_log_loss(df["is_goal"], df[pred_col]))
    total = d["loss"].sum()
    g = d.groupby(by, observed=True).agg(
        shots=("is_goal", "size"), goals=("is_goal", "sum"),
        mean_pred=(pred_col, "mean"), goal_rate=("is_goal", "mean"),
        mean_log_loss=("loss", "mean"), loss_sum=("loss", "sum"))
    g["bias"] = g["mean_pred"] - g["goal_rate"]
    g["share_of_total_loss"] = g["loss_sum"] / total
    return g.drop(columns="loss_sum").round(4)


def error_report(data: pd.DataFrame, oof: np.ndarray) -> dict[str, pd.DataFrame]:
    """Slice tables by shot characteristics plus the single worst predictions."""
    df = data.assign(pred=oof)
    df["distance_band"] = pd.cut(df["distance"], [0, 6, 12, 18, 25, 100],
                                 labels=["<6", "6-12", "12-18", "18-25", "25+"])
    out = {f"by_{c}": slice_table(df, c) for c in
           ["distance_band", "body_part", "shot_type", "play_pattern", "under_pressure",
            "first_time", "follows_cross", "follows_through_ball"]}
    df["loss"] = per_shot_log_loss(df["is_goal"], df["pred"])
    cols = ["match_id", "team", "player", "x", "y", "distance", "body_part", "shot_type",
            "pred", "sb_xg", "is_goal", "loss"]
    out["worst_goals"] = df[df["is_goal"] == 1].nlargest(10, "loss")[cols].round(3)
    out["worst_misses"] = df[df["is_goal"] == 0].nlargest(10, "loss")[cols].round(3)
    return out
