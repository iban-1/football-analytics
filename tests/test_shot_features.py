import numpy as np
import pandas as pd
import pytest

from src.features.shots import (add_score_diff, build_shot_dataset, exclusion_counts,
                                freeze_frame_features, goal_events, key_pass_flags)
from src.models.metrics import expected_calibration_error, reliability_curve, score_all
from src.models.xg import competition_splits, match_group_splits, predict_xg


def _ev(rows):
    base = {"match_id": 1, "period": 1, "minute": 10, "team": "A", "player": "p",
            "shot_outcome": None, "shot_type": None, "location": None, "shot_body_part": None,
            "shot_key_pass_id": None, "shot_first_time": None, "under_pressure": None,
            "shot_freeze_frame": None, "shot_statsbomb_xg": None, "play_pattern": "Regular Play",
            "competition_id": 1, "season_id": 1, "id": None, "pass_through_ball": None,
            "pass_cross": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def _synthetic_events():
    return _ev([
        {"index": 1, "type": "Pass", "id": "k1", "pass_cross": True},
        {"index": 2, "type": "Shot", "team": "A", "shot_outcome": "Goal", "shot_type": "Open Play",
         "location": [108.0, 40.0], "shot_body_part": "Right Foot", "shot_key_pass_id": "k1",
         "shot_statsbomb_xg": 0.3, "shot_first_time": True},
        {"index": 3, "type": "Shot", "team": "B", "shot_outcome": "Saved", "shot_type": "Open Play",
         "location": [100.0, 30.0], "shot_body_part": "Head", "shot_statsbomb_xg": 0.05},
        {"index": 4, "type": "Own Goal For", "team": "B"},
        {"index": 5, "type": "Shot", "team": "A", "shot_outcome": "Goal", "shot_type": "Penalty",
         "location": [108.0, 40.0], "shot_body_part": "Right Foot", "shot_statsbomb_xg": 0.78},
        {"index": 6, "type": "Shot", "team": "A", "shot_outcome": "Off T", "shot_type": "Open Play",
         "location": [110.0, 45.0], "shot_body_part": "Left Foot", "period": 5,
         "shot_statsbomb_xg": 0.1},
        {"index": 7, "type": "Shot", "team": "A", "shot_outcome": "Off T", "shot_type": "Free Kick",
         "location": [95.0, 40.0], "shot_body_part": "Left Foot", "shot_statsbomb_xg": 0.04},
    ])


def test_single_match_missing_columns_are_filled():
    """A match with no through balls etc. lacks those columns in raw StatsBomb output."""
    from src.data.statsbomb import REQUIRED_EVENT_COLUMNS, ensure_event_columns
    ev = _synthetic_events().drop(columns=["pass_through_ball", "pass_cross", "shot_first_time"])
    fixed = ensure_event_columns(ev)
    assert set(REQUIRED_EVENT_COLUMNS) <= set(fixed.columns)
    d = build_shot_dataset(fixed.drop(columns=["competition_id", "season_id"]))
    assert len(d) == 3 and d["follows_cross"].sum() == 0 and d["first_time"].sum() == 0


def test_penalties_and_shootouts_excluded_and_counted():
    ev = _synthetic_events()
    counts = exclusion_counts(ev)
    assert counts == {"all_shots": 5, "excluded_penalties": 1, "excluded_shootout_kicks": 1,
                      "modelled_shots": 3}
    assert len(build_shot_dataset(ev)) == 3


def test_features_built_correctly():
    d = build_shot_dataset(_synthetic_events())
    first = d.iloc[0]
    assert first["distance"] == pytest.approx(12.0)
    assert first["angle"] == pytest.approx(2 * np.arctan(4 / 12))
    assert first["follows_cross"] == 1 and first["follows_through_ball"] == 0
    assert first["first_time"] == 1 and first["is_goal"] == 1
    assert first["sb_xg"] == 0.3
    assert d["body_part"].tolist() == ["Foot", "Head", "Foot"]


def test_score_diff_counts_only_earlier_goals_including_own_goals():
    d = build_shot_dataset(_synthetic_events())
    # shot idx2 (A): 0-0 -> 0 ; idx3 (B): A already scored -> -1 ;
    # idx7 (A): A goal idx2, own goal for B idx4, penalty goal idx5 is a shot-goal -> A=2,B=1 -> +1
    assert d["score_diff"].tolist() == [0, -1, 1]


def test_goal_events_exclude_shootout_goals():
    ev = _synthetic_events()
    ev.loc[ev["index"] == 5, "period"] = 5
    assert len(goal_events(ev)) == 2  # shot goal + own goal


def test_key_pass_flags_missing_pass_is_false():
    ev = _synthetic_events()
    shots = ev[ev["type"] == "Shot"]
    out = key_pass_flags(shots, ev)
    assert out["follows_cross"].tolist()[0] == 1
    assert out["follows_cross"].sum() == 1


def test_no_leaky_columns_in_feature_sets():
    from src.features.shots import BASE_FEATURES_CAT, BASE_FEATURES_NUM, FREEZE_FEATURES
    banned = {"sb_xg", "shot_statsbomb_xg", "is_goal", "shot_outcome", "shot_end_location",
              "shot_deflected"}
    assert not banned & set(BASE_FEATURES_NUM + BASE_FEATURES_CAT + FREEZE_FEATURES)


def test_freeze_frame_features_hand_checked():
    frame = [
        {"location": [115.0, 40.0], "teammate": False, "position": {"name": "Center Back"}},
        {"location": [115.0, 60.0], "teammate": False, "position": {"name": "Left Back"}},
        {"location": [118.0, 40.0], "teammate": False, "position": {"name": "Goalkeeper"}},
        {"location": [112.0, 40.0], "teammate": True, "position": {"name": "Center Forward"}},
    ]
    f = freeze_frame_features([110.0, 40.0], frame)
    assert f["ff_defenders_in_cone"] == 1.0  # teammate and wide defender not counted
    assert f["ff_gk_in_cone"] == 1.0
    assert f["ff_gk_distance"] == pytest.approx(2.0)
    assert f["ff_nearest_defender"] == pytest.approx(5.0)


def test_freeze_frame_missing_gives_nan():
    assert np.isnan(freeze_frame_features([110.0, 40.0], float("nan"))["ff_gk_distance"])


def test_match_split_has_no_match_in_both_train_and_test():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, size=500)  # many shots per match
    n_test_total = 0
    for tr, te in match_group_splits(groups, 5):
        assert not set(groups[tr]) & set(groups[te])
        assert set(tr).isdisjoint(te)
        n_test_total += len(te)
    assert n_test_total == len(groups)  # every shot is tested exactly once


def test_competition_split_keeps_matches_apart():
    comp = np.array([1, 1, 2, 2, 3, 3])
    match = np.array([10, 11, 20, 21, 30, 31])
    for tr, te, _ in competition_splits(comp, match):
        assert not set(match[tr]) & set(match[te])


def test_metrics_perfectly_calibrated_has_zero_ece():
    y = np.array([0, 0, 0, 1] * 25)
    p = np.full(100, 0.25)
    assert expected_calibration_error(y, p, 5) == pytest.approx(0.0)
    s = score_all(y, p)
    assert s["log_loss"] == pytest.approx(-(0.25 * np.log(0.25) + 0.75 * np.log(0.75)))
    mean_p, obs, counts = reliability_curve(y, p, 5)
    assert counts.sum() == 100


def test_constant_predictions_form_a_single_bin():
    y = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    mean_p, obs, counts = reliability_curve(y, np.full(10, 0.2), 10)
    assert len(counts) == 1 and obs[0] == pytest.approx(0.2)
    assert expected_calibration_error(y, np.full(10, 0.2), 10) == pytest.approx(0.0)


def test_predict_xg_orders_shots_sensibly():
    pytest.importorskip("lightgbm")
    from src.models.xg import fit_final
    rng = np.random.default_rng(1)
    n = 600
    x = rng.uniform(85, 119, n)
    y = rng.uniform(25, 55, n)
    from src.features.geometry import shot_distance
    p_true = 1 / (1 + np.exp(-(1.0 - 0.18 * shot_distance(x, y))))
    df = pd.DataFrame({"x": x, "y": y, "match_id": rng.integers(0, 60, n),
                       "is_goal": (rng.random(n) < p_true).astype(int)})
    df["distance"] = shot_distance(df["x"], df["y"])
    from src.features.geometry import shot_angle
    df["angle"] = shot_angle(df["x"], df["y"])
    for col, v in {"first_time": 0, "follows_through_ball": 0, "follows_cross": 0,
                   "under_pressure": 0, "minute": 30.0, "score_diff": 0,
                   "body_part": "Foot", "shot_type": "Open Play",
                   "play_pattern": "Regular Play"}.items():
        df[col] = v
    model, _ = fit_final("logreg_all", df, False, seed=0)
    near = predict_xg({"x": 114, "y": 40}, model=model)[0]
    far = predict_xg({"x": 88, "y": 40}, model=model)[0]
    assert 0 < far < near < 1
