"""Phase 4: tracking analysis for Metrica sample games 1 and 2.

Run from the repo root:  python -m scripts.phase4_tracking
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.tracking import (formation_reliability, formation_windows, physical_table,
                                   possession_series, raw_kinematics, shape_by_possession,
                                   smooth_kinematics, team_shape)
from src.data.metrica import load_game
from src.utils.config import load_config, resolve, set_seed
from src.viz.tracking_plots import formation_plot, frame_plot, shape_timeline, voronoi_plot


def noise_comparison(game, team: str) -> dict:
    """Numbers behind 'why smooth': raw differencing vs Savitzky-Golay."""
    df = game.team(team)
    rs, ra = raw_kinematics(df, game.meta, game.fps)
    ss, sa = smooth_kinematics(df, game.meta, game.fps)

    def stats(speed, acc):
        a = acc.to_numpy().ravel()
        a = a[np.isfinite(a)]
        s = speed.to_numpy().ravel()
        s = s[np.isfinite(s)]
        return {"speed_p99_ms": round(float(np.percentile(s, 99)), 2),
                "max_speed_ms": round(float(s.max()), 2),
                "accel_p99_ms2": round(float(np.percentile(a, 99)), 2),
                "share_accel_over_10ms2": round(float((a > 10).mean()), 5)}
    return {"raw_differencing": stats(rs, ra), "savgol_smoothed": stats(ss, sa)}


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    out = resolve(cfg["paths"]["results"]) / "phase4"
    out.mkdir(parents=True, exist_ok=True)
    for g in (1, 2):
        game = load_game(g)
        print(f"\n================ Sample Game {g} ================")
        (out / f"quality_game{g}.json").write_text(
            json.dumps(game.quality, indent=1, default=int), encoding="utf-8")
        print("Data quality (raw):", json.dumps(game.quality, default=int))

        poss = possession_series(game)
        known = (poss != "").mean()
        print(f"Possession known for {known:.1%} of frames "
              f"(home {(poss == 'home').mean():.1%}, away {(poss == 'away').mean():.1%})")

        print("Noise comparison (home):", json.dumps(noise_comparison(game, "home")))

        phys, shapes = [], {}
        for team in ("home", "away"):
            speed, _ = smooth_kinematics(game.team(team), game.meta, game.fps)
            phys.append(physical_table(game, team, speed))
            shapes[team] = team_shape(game, team)
            # 1 Hz copy (every 5th row of the 5 Hz series) keeps the dashboard file small
            shapes[team].iloc[::5].round(2).to_csv(out / f"shape_game{g}_{team}_1hz.csv")
            sbp = shape_by_possession(shapes[team], poss, team)
            print(f"\n-- {team} shape by possession state --\n{sbp.to_string()}")
            sbp.to_csv(out / f"shape_by_possession_game{g}_{team}.csv")
        phys = pd.concat(phys, ignore_index=True)
        phys.to_csv(out / f"physical_game{g}.csv", index=False)
        print("\n-- physical table --")
        print(phys.to_string(index=False))

        shape_timeline(shapes, "hull_area", "convex hull area (m²)").savefig(
            out / f"shape_hull_game{g}.png", dpi=150)
        shape_timeline(shapes, "length", "team length (m)").savefig(
            out / f"shape_length_game{g}.png", dpi=150)
        shape_timeline(shapes, "width", "team width (m)").savefig(
            out / f"shape_width_game{g}.png", dpi=150)

        for variant, k_range in (("free_k_2to4", (2, 3, 4)), ("fixed_k3", (3,))):
            wins = []
            for team in ("home", "away"):
                w, pos = formation_windows(game, team, seed=cfg["seed"], poss=poss,
                                           k_range=k_range)
                wins.append(w)
                if pos and variant == "free_k_2to4":
                    key = sorted(pos)[0]
                    formation_plot(pos[key], f"{team}, game {g}, period {key[0]} window "
                                   f"from {key[1]} min, {key[2]}").savefig(
                        out / f"formation_game{g}_{team}.png", dpi=150, bbox_inches="tight")
            wins = pd.concat(wins, ignore_index=True)
            wins.to_csv(out / f"formation_windows_game{g}_{variant}.csv", index=False)
            rel = formation_reliability(wins)
            rel.to_csv(out / f"formation_reliability_game{g}_{variant}.csv", index=False)
            print(f"\n-- formation reliability [{variant}] --")
            print(rel.to_string(index=False))
            print("k chosen:", wins["k"].value_counts().to_dict(),
                  "| mean silhouette:", round(float(wins["silhouette"].mean()), 3))

        shots = game.events[game.events["type"] == "SHOT"]
        frame = int(shots["start_frame"].iloc[0])
        frame_plot(game, frame).savefig(out / f"frame_game{g}.png", dpi=150, bbox_inches="tight")
        fig, share = voronoi_plot(game, frame)
        fig.savefig(out / f"voronoi_game{g}.png", dpi=150, bbox_inches="tight")
        print(f"\nVoronoi at frame {frame}: home share of pitch {share:.3f}")
        plt.close("all")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
