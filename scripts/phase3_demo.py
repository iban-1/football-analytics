"""Phase 3 outputs for the demo match plus tournament-wide player table.

Run from the repo root:  python -m scripts.phase3_demo
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.events import (network_metrics, pass_network, penalty_conversion_rate,
                                 player_summary, shots_with_xg, touches)
from src.data.statsbomb import load_all_events, load_all_matches, load_nicknames
from src.utils.config import load_config, resolve, set_seed
from src.viz.match_plots import COLORS, heatmap, pass_network_plot, shot_map, xg_race


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    out = resolve(cfg["paths"]["results"]) / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    matches, events = load_all_matches(), None
    events = load_all_events(matches)
    mid = cfg["demo"]["match_id"]
    m = matches[matches["match_id"] == mid].iloc[0]
    teams = (m["home_team"], m["away_team"])
    pen_xg = penalty_conversion_rate(events)
    # saved so the dashboard can value penalties without loading every match
    (out / "penalty_rate.json").write_text(json.dumps(
        {"penalty_conversion_rate": pen_xg,
         "n_penalties": int(((events["shot_type"] == "Penalty") & (events["period"] != 5)).sum())}),
        encoding="utf-8")
    print(f"Penalty conversion rate used as penalty xG: {pen_xg:.4f} "
          f"(from {int(((events['shot_type'] == 'Penalty') & (events['period'] != 5)).sum())} in-match penalties)")

    shots_all = shots_with_xg(events, pen_xg)
    ev = events[events["match_id"] == mid]
    shots = shots_all[shots_all["match_id"] == mid]
    title = f"{teams[0]} {m['home_score']}-{m['away_score']} {teams[1]}"
    print(f"\n== {title} ==")
    print(shots.groupby("team").agg(shots=("xg", "size"), goals=("is_goal", "sum"),
                                    xg=("xg", "sum")).round(2).to_string())

    shot_map(shots, teams, title).savefig(out / "shot_map.png", dpi=150, bbox_inches="tight")
    xg_race(shots, teams, title).savefig(out / "xg_race.png", dpi=150, bbox_inches="tight")

    for team, color in zip(teams, COLORS):
        nodes, edges, minute = pass_network(ev, team)
        met = network_metrics(nodes, edges)
        print(f"\n-- {team} pass network (before first sub at minute {minute}) --")
        print(met.to_string(index=False))
        met.to_csv(out / f"centrality_{team}.csv", index=False)
        pass_network_plot(nodes, edges, color, title=f"{team} - {title}",
                          nicknames=load_nicknames(mid)).savefig(
            out / f"pass_network_{team}.png", dpi=150, bbox_inches="tight")
        heatmap(touches(ev, team=team), f"{team} touches - {title}").savefig(
            out / f"heatmap_{team}.png", dpi=150, bbox_inches="tight")
    summ_match = player_summary(ev, shots)
    print("\n-- player summary (this match, top 10 by xG) --")
    print(summ_match.head(10).to_string(index=False))
    summ_match.to_csv(out / "player_summary_match.csv", index=False)
    star = summ_match.sort_values("passes", ascending=False).iloc[0]["player"]
    heatmap(touches(ev, player=star), f"{star} - {title}").savefig(
        out / "heatmap_player.png", dpi=150, bbox_inches="tight")

    summ_all = player_summary(events, shots_all)
    summ_all.to_csv(out / "player_summary_tournaments.csv", index=False)
    print(f"\n-- player summary, all {events['match_id'].nunique()} matches: "
          f"{len(summ_all)} players; top 10 by goals - xG --")
    print(summ_all[summ_all["shots"] >= 10].sort_values("goals_minus_xg", ascending=False)
          .head(10).to_string(index=False))
    plt.close("all")
    print(f"\nSaved figures/tables to {out}")


if __name__ == "__main__":
    main()
