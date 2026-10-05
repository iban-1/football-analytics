"""Reproduce the whole xG experiment from raw data to results/.

Run from the repo root:  python -m scripts.reproduce_xg
(Needs data/raw/ filled by `python -m scripts.download_data`.)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.analysis.xg_errors import error_report
from src.data.statsbomb import load_all_events
from src.features.shots import build_shot_dataset, exclusion_counts, feature_columns
from src.models.metrics import ece_noise_floor, score_all
from src.models.xg import (MODEL_NAMES, build_model, fit_final, match_group_splits,
                           nested_cv, save_model)
from src.utils.config import load_config, resolve, set_seed
from src.viz.xg_plots import calibration_plot, permutation_importance_plot, shap_summary

METRICS = ["log_loss", "brier", "roc_auc", "ece"]


def summarise(folds: pd.DataFrame) -> dict[str, str]:
    """mean +/- std across folds for each metric."""
    return {m: f"{folds[m].mean():.4f} +/- {folds[m].std(ddof=1):.4f}" for m in METRICS}


def statsbomb_folds(data: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb's own xG scored on the same match folds (benchmark, not a model of ours)."""
    y, p = data["is_goal"].to_numpy(), data["sb_xg"].to_numpy()
    rows = [score_all(y[te], p[te])
            for _, te in match_group_splits(data["match_id"].to_numpy())]
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    res = resolve(cfg["paths"]["results"])
    proc = resolve(cfg["paths"]["processed"])
    res.mkdir(exist_ok=True)
    proc.mkdir(exist_ok=True)

    events = load_all_events()
    print("Shot exclusions:", exclusion_counts(events))
    data = build_shot_dataset(events)
    data.to_csv(proc / "shots.csv", index=False)
    y = data["is_goal"].to_numpy()
    print(f"Modelling dataset: {len(data)} shots, {data['match_id'].nunique()} matches, "
          f"{y.sum()} goals ({y.mean():.4f})")
    print(f"Shots without freeze frame: {int(data['ff_gk_distance'].isna().sum())}")

    # ---- main comparison: base features, match-grouped nested CV ----
    table, oof_all, params_all, fold_tables = [], {}, {}, {}
    for name in MODEL_NAMES:
        folds, oof, params = nested_cv(name, data, False, seed)
        fold_tables[name], oof_all[name], params_all[name] = folds, oof, params
        table.append({"model": name, "features": "base", **summarise(folds)})
        print(f"[{time.time() - t0:5.0f}s] {name:16s}", table[-1])
    sb = statsbomb_folds(data)
    table.append({"model": "statsbomb_xg (benchmark)", "features": "own", **summarise(sb)})
    pd.DataFrame(table).to_csv(res / "xg_results.csv", index=False)
    pooled = {n: score_all(y, p) for n, p in oof_all.items()}
    pooled["statsbomb_xg (benchmark)"] = score_all(y, data["sb_xg"].to_numpy())
    pooled_df = pd.DataFrame(pooled).T
    preds_for_floor = {**oof_all, "statsbomb_xg (benchmark)": data["sb_xg"].to_numpy()}
    pooled_df["ece_noise_floor"] = [ece_noise_floor(preds_for_floor[n]) for n in pooled_df.index]
    pooled_df.round(4).to_csv(res / "xg_results_pooled_oof.csv")

    # Paired per-fold log-loss differences vs logreg_all (same folds, so differences
    # are far less noisy than the across-fold std of each model alone).
    ref = fold_tables["logreg_all"]["log_loss"]
    paired = [{"model": n, "mean_diff_vs_logreg_all": (fold_tables[n]["log_loss"] - ref).mean(),
               "std_diff": (fold_tables[n]["log_loss"] - ref).std(ddof=1),
               "folds_better_than_logreg_all": int((fold_tables[n]["log_loss"] < ref).sum())}
              for n in MODEL_NAMES if n != "logreg_all"]
    paired.append({"model": "statsbomb_xg (benchmark)",
                   "mean_diff_vs_logreg_all": (sb["log_loss"] - ref).mean(),
                   "std_diff": (sb["log_loss"] - ref).std(ddof=1),
                   "folds_better_than_logreg_all": int((sb["log_loss"] < ref).sum())})
    pd.DataFrame(paired).round(4).to_csv(res / "xg_results_paired_differences.csv", index=False)

    # ---- ablation: freeze-frame features ----
    ablation = []
    for name in ["logreg_all", "lightgbm", "mlp"]:
        folds, oof, _ = nested_cv(name, data, True, seed)
        ablation.append({"model": name, "features": "base+freeze_frame", **summarise(folds)})
        oof_all[f"{name}+ff"] = oof
        print(f"[{time.time() - t0:5.0f}s] {name}+ff", ablation[-1])
    pd.DataFrame(ablation).to_csv(res / "xg_results_freeze_frame.csv", index=False)

    # ---- generalisation: leave one tournament out ----
    loco = []
    for name in MODEL_NAMES:
        folds, _, _ = nested_cv(name, data, False, seed, scheme="competition")
        loco.append({"model": name, "features": "base", **summarise(folds)})
        print(f"[{time.time() - t0:5.0f}s] LOCO {name}", loco[-1])
    pd.DataFrame(loco).to_csv(res / "xg_results_leave_one_tournament_out.csv", index=False)

    # ---- settings record ----
    grids = {n: build_model(n, *feature_columns(False), seed)[1] for n in MODEL_NAMES}
    (res / "xg_settings.json").write_text(json.dumps(
        {"seed": seed, "outer_cv": "GroupKFold(5) by match_id",
         "inner_cv": "GroupKFold(3) by match_id, scoring=neg_log_loss",
         "grids": grids, "chosen_params_per_outer_fold": params_all}, indent=2, default=str),
        encoding="utf-8")

    # ---- calibration plots (pooled out-of-fold) ----
    preds = {n: oof_all[n] for n in MODEL_NAMES}
    preds["statsbomb_xg"] = data["sb_xg"].to_numpy()
    calibration_plot(y, preds).savefig(res / "xg_calibration.png", dpi=150)

    # ---- best model (by mean CV log loss, base features) and final fit ----
    scored = {n: fold_tables[n]["log_loss"].mean() for n in MODEL_NAMES}
    best = min(scored, key=lambda n: scored[n])
    print("Best by CV log loss:", best, round(scored[best], 4))
    num, cat = feature_columns(False)
    final, final_params = fit_final(best, data, False, seed)
    save_model(final, res / "xg_model.joblib",
               {"model": best, "features": num + cat, "params": final_params, "seed": seed,
                "trained_on_shots": len(data), "note": "fit on all shots; "
                "performance numbers come from nested CV, not this fit"})
    if best in ("lightgbm", "logreg_all"):
        fig, imp = shap_summary(final, data[num + cat], best, seed)
    else:
        fig, imp = permutation_importance_plot(final, data[num + cat], y, seed)
    fig.savefig(res / "xg_feature_importance.png", dpi=150, bbox_inches="tight")
    imp.round(4).to_csv(res / "xg_feature_importance.csv", index=False)
    print(imp.head(10).to_string(index=False))

    # ---- error analysis on out-of-fold predictions of the best model ----
    for key, tbl in error_report(data, oof_all[best]).items():
        tbl.to_csv(res / f"xg_errors_{key}.csv")
        print(f"\n== {key} ==\n{tbl.to_string()}")

    # ---- suspicion check ----
    sb_ll = pooled["statsbomb_xg (benchmark)"]["log_loss"]
    for n, m in pooled.items():
        if n.startswith("statsbomb"):
            continue
        if m["log_loss"] < sb_ll:
            print(f"WARNING: {n} beats StatsBomb xG on pooled log loss "
                  f"({m['log_loss']:.4f} < {sb_ll:.4f}) - investigate before claiming a win")
    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
