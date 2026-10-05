"""Tracking-quality PROXIES computed without ground-truth labels.

No accuracy is claimed. Without hand-labelled frames we cannot measure ID switches or
missed players, so these numbers only describe the tracker's behaviour:

* players_per_frame      average on-grass persons detected per frame
* unique_track_ids       how many IDs the tracker issued for those persons
* mean_track_len_frames  average number of frames each ID lasted
* id_to_player_ratio     unique IDs / median players per frame. ~1 would mean every player
                         kept one ID; larger values mean tracks were broken or swapped
                         (a fragmentation proxy, not an ID-switch count)
* ball_frames_share      fraction of frames with at least one ball detection
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_quality(det: pd.DataFrame, n_frames: int) -> dict:
    """Proxy metrics for one shot's detections (`n_frames` = frames processed)."""
    players = det[(det["cls"] == 0) & det["on_grass"].astype(bool)]
    ids = players[players["track_id"] >= 0]
    per_frame = players.groupby("frame").size().reindex(range(det["frame"].min(),
                                                           det["frame"].min() + n_frames),
                                                       fill_value=0)
    lengths = ids.groupby("track_id").size()
    balls = det[det["cls"] == 32]["frame"].nunique()
    med = float(np.median(per_frame)) if len(per_frame) else 0.0
    return {"frames": n_frames,
            "players_per_frame": round(float(per_frame.mean()), 2),
            "median_players_per_frame": med,
            "unique_track_ids": int(ids["track_id"].nunique()),
            "mean_track_len_frames": round(float(lengths.mean()), 1) if len(lengths) else 0.0,
            "id_to_player_ratio": round(ids["track_id"].nunique() / med, 2) if med else None,
            "ball_frames_share": round(balls / n_frames, 3),
            "persons_removed_not_on_grass": int(((det["cls"] == 0)
                                                 & ~det["on_grass"].astype(bool)).sum())}
