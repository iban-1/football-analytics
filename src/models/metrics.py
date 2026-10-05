"""Probability-model metrics. Accuracy is deliberately absent.

Only ~9% of shots are goals, so "always predict no goal" scores ~91% accuracy
while being useless. We judge probabilities instead:
  * log loss  - punishes confident wrong predictions heavily (main metric)
  * Brier     - mean squared error of the probability
  * ROC AUC   - ranking quality only (says nothing about calibration)
  * ECE       - do predicted probabilities match observed goal frequencies?
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-15


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean predicted prob, observed goal rate and count per quantile bin.

    Equal-FREQUENCY bins (not equal width): xG predictions pile up near 0.05,
    so equal-width bins would leave most bins empty.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    # Bin edges are quantiles of p; searchsorted puts identical predictions in the
    # same bin, so a constant-rate model collapses to one bin instead of being
    # split arbitrarily (which would make its "calibration error" meaningless).
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    labels = np.searchsorted(edges, p, side="right")
    groups = [np.where(labels == k)[0] for k in np.unique(labels)]
    mean_p = np.array([p[g].mean() for g in groups])
    obs = np.array([y[g].mean() for g in groups])
    counts = np.array([len(g) for g in groups])
    return mean_p, obs, counts


def ece_noise_floor(p: np.ndarray, n_bins: int = 10, n_sims: int = 200,
                    seed: int = 0) -> float:
    """ECE expected from sampling noise alone if the predictions were exactly right.

    Draws outcomes y ~ Bernoulli(p) from the model's own probabilities and
    averages the resulting ECE. An observed ECE near this value is
    indistinguishable from perfect calibration at this sample size.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    return float(np.mean([expected_calibration_error(rng.binomial(1, p), p, n_bins)
                          for _ in range(n_sims)]))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Weighted average gap |observed rate - mean prediction| over quantile bins."""
    mean_p, obs, counts = reliability_curve(y, p, n_bins)
    return float(np.sum(counts * np.abs(obs - mean_p)) / counts.sum())


def score_all(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """The four headline metrics for one set of predictions."""
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ece": expected_calibration_error(y, p),
    }
