"""Phase 1 data exploration: prints real numbers and saves a shot-location plot.

Run from the repo root:  python -m scripts.explore_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.statsbomb import load_all_events, load_all_matches
from src.utils.config import load_config, resolve
from src.viz.pitch import draw_pitch


def shot_table(events: pd.DataFrame) -> pd.DataFrame:
    """Shots with x/y split out of the `location` list column."""
    shots = events[events["type"] == "Shot"].copy()
    shots["x"] = shots["location"].map(lambda v: v[0])
    shots["y"] = shots["location"].map(lambda v: v[1])
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    return shots


def main() -> None:
    cfg = load_config()
    out = resolve(cfg["paths"]["results"])
    matches = load_all_matches()
    events = load_all_events(matches)
    shots = shot_table(events)

    print("== Matches per competition-season ==")
    names = {(c["competition_id"], c["season_id"]): c["name"]
             for c in cfg["statsbomb"]["competitions"]}
    per = matches.groupby(["competition_id", "season_id"]).size()
    for key, n in per.items():
        print(f"  {names[key]}: {n} matches")
    print(f"Total matches: {len(matches)}")
    print(f"Total events: {len(events):,}")

    print("\n== Events by type ==")
    print(events["type"].value_counts().to_string())

    print("\n== Shots ==")
    n_pen = int((shots["shot_type"] == "Penalty").sum())
    n_shootout = int((shots["period"] == 5).sum())
    print(f"Total shots (all): {len(shots):,}")
    print(f"  penalties in-match/shootout (shot_type=Penalty): {n_pen}")
    print(f"  shots in penalty shootouts (period 5): {n_shootout}")
    per_match = shots.groupby("match_id").size().reindex(matches["match_id"], fill_value=0)
    print(f"Shots per match: mean {per_match.mean():.2f}, median {per_match.median():.0f}, "
          f"min {per_match.min()}, max {per_match.max()}")
    print(f"Goals (all shots): {shots['is_goal'].sum()}  "
          f"goal rate {shots['is_goal'].mean():.4f}")
    open_play = shots[(shots["shot_type"] != "Penalty") & (shots["period"] != 5)]
    print(f"Excluding penalties & shootouts: {len(open_play):,} shots, "
          f"{open_play['is_goal'].sum()} goals, goal rate {open_play['is_goal'].mean():.4f}")
    print("\nShot outcomes:\n" + shots["shot_outcome"].value_counts().to_string())

    print("\n== Shot location distribution (x, y in StatsBomb units) ==")
    print(open_play[["x", "y"]].describe().round(2).to_string())
    dist = np.hypot(120 - open_play["x"], 40 - open_play["y"])
    print(f"Distance to goal centre: mean {dist.mean():.1f}, median {dist.median():.1f}")
    print(f"Share of shots inside the 18-yard box (x>=102, 18<=y<=62): "
          f"{((open_play['x'] >= 102) & open_play['y'].between(18, 62)).mean():.3f}")

    fig, ax, pitch = draw_pitch(half=True)
    pitch.kdeplot(open_play["x"], open_play["y"], ax=ax, fill=True, levels=30,
                  thresh=0.02, cmap="Reds", alpha=0.8)
    ax.set_title("Shot location density (excl. penalties/shootouts)")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "shot_location_density.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved {out / 'shot_location_density.png'}")


if __name__ == "__main__":
    main()
