"""Split tracked players into two teams by shirt colour (no labels, no training).

Each track gets one colour: the median of its per-frame torso colours in Lab space (a
colour space where distance roughly matches how different two colours look). K-means with
two clusters separates the two kits across the whole clip, so labels are consistent
between shots. Tracks far from both cluster centres (referees, goalkeepers in a third
colour) are labelled 'other'. The lighter cluster is called 'light', the other 'dark'
(here: light = Argentina's sky-blue/white, dark = France's navy; a visual check, not metadata).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

TEAM_LABELS = ("light", "dark", "other")


def assign_teams(det: pd.DataFrame, seed: int = 0, min_frames: int = 5,
                 outlier_factor: float = 2.5) -> pd.DataFrame:
    """Add a `team` column ('light' / 'dark' / 'other') to the detections table.

    Only on-grass persons with a valid colour take part; balls get 'ball'. A track with
    fewer than `min_frames` coloured detections gets 'other'. A track is an outlier if
    its distance to the nearest centre exceeds `outlier_factor` x the median distance of
    all tracks to their own centre.
    """
    det = det.copy()
    det["team"] = "other"
    det.loc[det["cls"] == 32, "team"] = "ball"
    pl = det[(det["cls"] == 0) & det["on_grass"].astype(bool) & det["L"].notna()
             & (det["track_id"] >= 0)]
    per = pl.groupby(["segment", "track_id"]).agg(L=("L", "median"), a=("a", "median"),
                                                 b=("b", "median"), n=("L", "size"))
    per = per[per["n"] >= min_frames]
    if len(per) < 4:
        return det
    X = per[["L", "a", "b"]].to_numpy()
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(X)
    dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    cutoff = outlier_factor * np.median(dist)
    light_cluster = int(np.argmax(km.cluster_centers_[:, 0]))
    team = np.where(km.labels_ == light_cluster, "light", "dark")
    team[dist > cutoff] = "other"
    mapping = dict(zip(per.index, team))
    keys = list(zip(det["segment"], det["track_id"]))
    det["team"] = [mapping.get(k, t) for k, t in zip(keys, det["team"])]
    det.attrs["kit_lab_centres"] = {"light": km.cluster_centers_[light_cluster].tolist(),
                                    "dark": km.cluster_centers_[1 - light_cluster].tolist()}
    return det
