"""xG models, match-grouped cross-validation, and the `predict_xg` API.

Design in plain words
---------------------
* Every model is a scikit-learn Pipeline (preprocessing + estimator) so the
  scaler/encoder are fitted on training shots only (no peeking at test data).
* Evaluation is NESTED: the outer GroupKFold measures performance; inside each
  outer training set a smaller GroupKFold picks hyper-parameters. The test
  fold therefore never influences tuning. Groups are match ids, so shots from
  one match never sit in both train and test.
* The grids are deliberately tiny ("tune lightly").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.geometry import shot_angle, shot_distance
from src.features.shots import BASE_FEATURES_CAT, BASE_FEATURES_NUM, feature_columns
from src.models.metrics import score_all

MODEL_NAMES = ["constant_rate", "logreg_distance", "logreg_all", "lightgbm", "mlp"]


# ---------- model construction ----------

def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Median-impute (+ missing flag) and scale numerics; one-hot categoricals.

    Imputation only matters for freeze-frame features (~4% of shots lack a
    frame); the missing flag lets the model learn "no frame" as its own signal.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    transformers: list[tuple[str, Any, list[str]]] = [("num", numeric, num)]
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(transformers)


def build_model(name: str, num: list[str], cat: list[str], seed: int,
                ) -> tuple[Pipeline, dict[str, list]]:
    """Return (untuned pipeline, hyper-parameter grid) for a model name."""
    if name == "constant_rate":
        # Predicts the training goal rate for every shot: the "know nothing" floor.
        return Pipeline([("clf", DummyClassifier(strategy="prior"))]), {}
    if name == "logreg_distance":
        pre = _preprocessor(["distance"], [])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        return Pipeline([("pre", pre), ("clf", clf)]), {}
    if name == "logreg_all":
        pre = _preprocessor(num, cat)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        return Pipeline([("pre", pre), ("clf", clf)]), {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "lightgbm":
        pre = _preprocessor(num, cat)
        clf = LGBMClassifier(learning_rate=0.03, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, n_jobs=1,
                             verbose=-1, deterministic=True, force_row_wise=True)
        grid = {"clf__num_leaves": [4, 8, 16], "clf__n_estimators": [100, 250],
                "clf__min_child_samples": [20, 60]}
        return Pipeline([("pre", pre), ("clf", clf)]), grid
    if name == "mlp":
        pre = _preprocessor(num, cat)
        clf = MLPClassifier(max_iter=800, random_state=seed)
        grid = {"clf__hidden_layer_sizes": [(16,), (32, 16)],
                "clf__alpha": [1e-3, 1e-2, 1e-1]}
        return Pipeline([("pre", pre), ("clf", clf)]), grid
    raise ValueError(f"unknown model: {name}")


# ---------- splitting ----------

def match_group_splits(groups: np.ndarray, n_splits: int = 5,
                       ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by match id. Asserts that no match is in both train and test."""
    groups = np.asarray(groups)
    for train, test in GroupKFold(n_splits=n_splits).split(groups, groups=groups):
        assert not set(groups[train]) & set(groups[test]), "match leaked across split"
        yield train, test


def competition_splits(competition: np.ndarray, match_id: np.ndarray,
                       ) -> Iterator[tuple[np.ndarray, np.ndarray, Any]]:
    """Leave-one-competition-out splits (a stricter test: unseen tournament)."""
    competition = np.asarray(competition)
    match_id = np.asarray(match_id)
    for comp in np.unique(competition):
        test = np.where(competition == comp)[0]
        train = np.where(competition != comp)[0]
        assert not set(match_id[train]) & set(match_id[test])
        yield train, test, comp


# ---------- cross-validation ----------

def _fit_tuned(name: str, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               num: list[str], cat: list[str], seed: int, inner_splits: int = 3,
               ) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one model; if it has a grid, pick params by grouped inner CV on log loss."""
    pipe, grid = build_model(name, num, cat, seed)
    if not grid:
        return pipe.fit(X, y), {}
    inner = list(GroupKFold(n_splits=inner_splits).split(X, y, groups))
    search = GridSearchCV(pipe, grid, scoring="neg_log_loss", cv=inner, n_jobs=1,
                          refit=True)
    search.fit(X, y)
    return search.best_estimator_, {k: _jsonable(v) for k, v in search.best_params_.items()}


def _jsonable(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def nested_cv(name: str, data: pd.DataFrame, use_freeze_frame: bool, seed: int,
              n_splits: int = 5, scheme: str = "match",
              ) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    """Outer CV for one model.

    Returns (per-fold metrics, out-of-fold predictions aligned with `data`,
    chosen hyper-parameters per fold). `scheme` is 'match' (GroupKFold on
    matches) or 'competition' (leave one tournament out).
    """
    num, cat = feature_columns(use_freeze_frame)
    X, y, groups = data[num + cat], data["is_goal"].to_numpy(), data["match_id"].to_numpy()
    if scheme == "match":
        splits = [(tr, te, k) for k, (tr, te) in enumerate(match_group_splits(groups, n_splits))]
    else:
        splits = list(competition_splits(data["competition_id"].to_numpy()
                                         * 1000 + data["season_id"].to_numpy(), groups))
    oof = np.full(len(data), np.nan)
    rows, params = [], []
    for tr, te, fold in splits:
        model, best = _fit_tuned(name, X.iloc[tr], y[tr], groups[tr], num, cat, seed)
        p = model.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        rows.append({"fold": fold, "n_test": len(te), "goal_rate": y[te].mean(),
                     **score_all(y[te], p)})
        params.append({"fold": fold, **best})
    return pd.DataFrame(rows), oof, params


def fit_final(name: str, data: pd.DataFrame, use_freeze_frame: bool, seed: int,
              ) -> tuple[Pipeline, dict[str, Any]]:
    """Fit on ALL shots; hyper-parameters chosen by 5-fold grouped CV on all data."""
    num, cat = feature_columns(use_freeze_frame)
    return _fit_tuned(name, data[num + cat], data["is_goal"].to_numpy(),
                      data["match_id"].to_numpy(), num, cat, seed, inner_splits=5)


# ---------- saving and prediction ----------

def save_model(model: Pipeline, path: Path, meta: dict[str, Any]) -> None:
    """Save the fitted pipeline (joblib) plus a human-readable JSON sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


_DEFAULTS = {"body_part": "Foot", "shot_type": "Open Play", "play_pattern": "Regular Play",
             "first_time": 0, "follows_through_ball": 0, "follows_cross": 0,
             "under_pressure": 0, "minute": 45.0, "score_diff": 0}


def predict_xg(shot_features: dict | pd.DataFrame, model: Pipeline | None = None,
               model_path: Path | None = None) -> np.ndarray:
    """Predict P(goal) for one or more shots.

    `shot_features` needs `x` and `y` (StatsBomb coordinates, attacking toward
    x = 120). Distance and angle are computed here so callers cannot get them
    inconsistent. Other fields default to a neutral open-play foot shot (see
    `_DEFAULTS`). Returns an array of probabilities.
    """
    if model is None:
        from src.utils.config import load_config, resolve
        path = model_path or resolve(load_config()["paths"]["results"]) / "xg_model.joblib"
        model = joblib.load(path)  # our own file; see pickle note in data/statsbomb.py
    df = pd.DataFrame([shot_features]) if isinstance(shot_features, dict) \
        else shot_features.copy()
    for col, default in _DEFAULTS.items():
        if col not in df:
            df[col] = default
    df["distance"] = shot_distance(df["x"], df["y"])
    df["angle"] = shot_angle(df["x"], df["y"])
    cols = BASE_FEATURES_NUM + BASE_FEATURES_CAT
    needed = [c for c in cols if c in model.feature_names_in_] if hasattr(
        model, "feature_names_in_") else cols
    return model.predict_proba(df[needed])[:, 1]
