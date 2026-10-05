"""Plots for the xG model: calibration curves and feature importance (SHAP)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.models.metrics import expected_calibration_error, reliability_curve


def calibration_plot(y: np.ndarray, preds: dict[str, np.ndarray], n_bins: int = 10,
                     ) -> plt.Figure:
    """One reliability panel per model (pooled out-of-fold predictions).

    Points on the dashed diagonal = perfectly calibrated. Bins have equal
    shot counts; the title shows the ECE for the same bins.
    """
    cols = 3
    rows = int(np.ceil(len(preds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    for ax, (name, p) in zip(axes.ravel(), preds.items()):
        mean_p, obs, _ = reliability_curve(y, p, n_bins)
        ax.plot([0, 0.5], [0, 0.5], "--", color="grey", lw=1)
        ax.plot(mean_p, obs, "o-", color="#1f77b4")
        ax.set_title(f"{name}\nECE = {expected_calibration_error(y, p, n_bins):.4f}",
                     fontsize=10)
        ax.set_xlabel("mean predicted xG")
        ax.set_ylabel("observed goal rate")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.5)
    for ax in axes.ravel()[len(preds):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _feature_names(model: Pipeline) -> list[str]:
    return [n.replace("num__", "").replace("cat__", "")
            for n in model.named_steps["pre"].get_feature_names_out()]


def shap_summary(model: Pipeline, X: pd.DataFrame, model_name: str,
                 seed: int = 0) -> tuple[plt.Figure, pd.DataFrame]:
    """Importance of each (encoded) feature for a fitted pipeline.

    Tree model -> SHAP TreeExplainer; logistic regression -> SHAP
    LinearExplainer; anything else -> permutation importance (log loss).
    Returns the figure and a table of mean |importance|.
    """
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = _feature_names(model)
    if model_name == "lightgbm":
        vals = shap.TreeExplainer(clf).shap_values(Xt)
    elif model_name.startswith("logreg"):
        masker = shap.maskers.Independent(Xt, max_samples=len(Xt))
        vals = shap.LinearExplainer(clf, masker).shap_values(Xt)
    else:
        raise NotImplementedError("use permutation_importance_plot for this model type")
    vals = vals[1] if isinstance(vals, list) else vals
    table = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(vals).mean(0)})
    table = table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(vals, Xt, feature_names=names, show=False, max_display=15)
    fig = plt.gcf()
    fig.tight_layout()
    return fig, table


def permutation_importance_plot(model: Pipeline, X: pd.DataFrame, y: np.ndarray,
                                seed: int) -> tuple[plt.Figure, pd.DataFrame]:
    """Model-agnostic fallback: how much log loss rises when a column is shuffled."""
    res = permutation_importance(model, X, y, scoring="neg_log_loss", n_repeats=5,
                                 random_state=seed)
    table = pd.DataFrame({"feature": X.columns, "importance": res.importances_mean}
                         ).sort_values("importance", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(table["feature"][::-1], table["importance"][::-1])
    ax.set_xlabel("increase in log loss when shuffled")
    fig.tight_layout()
    return fig, table
