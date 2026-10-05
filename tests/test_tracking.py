"""Phase 4 tests: parser, cleaning, kinematics and tactical metrics on tiny
synthetic data with known answers, plus checks on the real Metrica files."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.tracking import (_hull_area, formation_from_positions, goalkeepers,
                                   physical_table, possession_series, raw_kinematics,
                                   smooth_kinematics, voronoi_control)
from src.data.metrica import (TrackingGame, _repair, attack_direction_at_kickoff,
                              convert_to_metres, despike, flip_frames, read_events_csv,
                              read_tracking_csv)
from src.data.metrica_download import raw_dir

FPS = 25.0
L, W = 105.0, 68.0


# ---------- parser ----------

def _write_tracking(path):
    lines = [",,,Home,,Home,,Home,",
             ",,,11,,1,,,",
             "Period,Frame,Time [s],Player11,,Player1,,Ball,",
             "1,1,0.04,0.10,0.50,0.60,0.25,0.50,0.50",
             "1,2,0.08,0.10,0.50,NaN,NaN,0.52,0.50",
             "2,3,0.12,0.90,0.50,0.40,0.75,NaN,NaN"]
    path.write_text("\n".join(lines))


def test_read_tracking_csv_names_and_values(tmp_path):
    f = tmp_path / "t.csv"
    _write_tracking(f)
    meta, pos = read_tracking_csv(f)
    assert list(pos.columns) == ["Player11_x", "Player11_y", "Player1_x", "Player1_y",
                                 "Ball_x", "Ball_y"]
    assert meta["period"].tolist() == [1, 1, 2] and meta.index.tolist() == [1, 2, 3]
    assert pos.loc[1, "Player1_y"] == 0.25 and np.isnan(pos.loc[2, "Player1_x"])


def test_convert_to_metres_flips_y_to_bottom_left_origin(tmp_path):
    f = tmp_path / "t.csv"
    _write_tracking(f)
    _, pos = read_tracking_csv(f)
    m = convert_to_metres(pos, L, W)
    assert m.loc[1, "Player11_x"] == pytest.approx(10.5)
    assert m.loc[1, "Player1_y"] == pytest.approx(0.75 * 68)  # top-left y=0.25 -> 51 m up


def test_attack_direction_and_flip(tmp_path):
    f = tmp_path / "t.csv"
    _write_tracking(f)
    meta, pos = read_tracking_csv(f)
    m = convert_to_metres(pos, L, W)
    # period 1: players at x = 10.5 and 63 m, mean < 52.5 -> starts left -> attacks right
    assert attack_direction_at_kickoff(m, meta, 1, L) == 1
    flip_frames(m, (meta["period"] == 2).to_numpy(), L, W)
    assert m.loc[3, "Player11_x"] == pytest.approx(105 - 94.5)
    assert m.loc[3, "Player11_y"] == pytest.approx(68 - 34.0)


def test_read_events_renames_and_lowercases(tmp_path):
    f = tmp_path / "e.csv"
    f.write_text("Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],"
                 "From,To,Start X,Start Y,End X,End Y\nHome,PASS,,1,5,0.2,9,0.36,P1,P2,0.5,0.5,0.6,0.4")
    ev = read_events_csv(f)
    assert ev.loc[0, "team"] == "home" and ev.loc[0, "start_frame"] == 5


# ---------- cleaning ----------

def test_single_frame_spike_is_removed_and_repaired():
    n = 200
    x = np.linspace(10, 20, n)          # 0.05 m per frame = 1.25 m/s
    y = np.full(n, 30.0)
    x[100] += 15                        # one-frame glitch
    meta = pd.DataFrame({"period": 1, "time_s": np.arange(n) / FPS}, index=np.arange(1, n + 1))
    team = pd.DataFrame({"P1_x": x, "P1_y": y}, index=meta.index)
    fixed, rep = despike(team, meta, FPS)
    assert rep["glitch_points_flagged"] >= 1
    assert fixed["P1_x"].iloc[100] == pytest.approx(10 + 10 * 100 / (n - 1), abs=0.2)
    assert fixed["P1_x"].isna().sum() == 0


def test_permanent_jump_is_not_bridged_with_a_fake_ramp():
    # 30 m jump held afterwards, like a lost-and-reacquired track
    x = np.r_[np.full(50, 10.0), np.full(50, 40.0)]
    y = np.full(100, 30.0)
    bad = np.zeros(100, bool)
    bad[49:51] = True
    fx, fy, left = _repair(x, y, bad, FPS, max_gap=25)
    assert left == 2 and np.isnan(fx[49:51]).all()


def test_gap_longer_than_limit_is_left_missing():
    x = np.linspace(0, 10, 100)
    y = np.zeros(100)
    bad = np.zeros(100, bool)
    bad[40:70] = True  # 30 frames > 25
    _, _, left = _repair(x, y, bad, FPS, max_gap=25)
    assert left == 30


# ---------- kinematics ----------

def _team_one_player(x, y, period=None):
    n = len(x)
    meta = pd.DataFrame({"period": period if period is not None else 1,
                         "time_s": np.arange(n) / FPS}, index=np.arange(1, n + 1))
    return pd.DataFrame({"Player1_x": x, "Player1_y": y}, index=meta.index), meta


def test_constant_velocity_recovered_by_smoothing():
    t = np.arange(250) / FPS
    team, meta = _team_one_player(5.0 * t, np.full(250, 20.0))
    sp, ac = smooth_kinematics(team, meta, FPS)
    mid = sp["Player1"].iloc[20:-20]
    assert np.allclose(mid, 5.0, atol=1e-6)
    assert np.allclose(ac["Player1"].iloc[20:-20], 0.0, atol=1e-6)


def test_smoothing_reduces_noise_in_acceleration():
    rng = np.random.default_rng(0)
    t = np.arange(500) / FPS
    team, meta = _team_one_player(4.0 * t + rng.normal(0, 0.05, 500),
                                  20 + rng.normal(0, 0.05, 500))
    _, ra = raw_kinematics(team, meta, FPS)
    _, sa = smooth_kinematics(team, meta, FPS)
    # true acceleration is 0; smoothed estimate must be much closer to it than raw differencing
    assert np.nanmean(sa["Player1"]) < 0.25 * np.nanmean(ra["Player1"])


def test_edges_trimmed_and_gap_not_bridged():
    x = np.arange(200) * 0.1
    x[90:100] = np.nan
    team, meta = _team_one_player(x, np.zeros(200))
    sp, _ = smooth_kinematics(team, meta, FPS)
    s = sp["Player1"].to_numpy()
    assert np.isnan(s[90:100]).all()
    assert np.isnan(s[:6]).all() and np.isnan(s[-6:]).all()       # run-start / run-end trimmed
    assert np.isnan(s[84:90]).all() and np.isnan(s[100:106]).all()  # next to the gap too


def test_nothing_smoothed_across_period_boundary():
    x = np.r_[np.arange(100) * 0.1, 50 + np.arange(100) * 0.1]  # 40 m teleport at half-time
    team, meta = _team_one_player(x, np.zeros(200), period=np.r_[np.ones(100), np.full(100, 2)])
    sp, _ = smooth_kinematics(team, meta, FPS)
    assert np.nanmax(sp["Player1"]) < 3.0  # a smoothed spike would be enormous


# ---------- physical metrics ----------

def _fake_game(team_df, meta, events=None):
    cols = ["Player1_x", "Player1_y"]
    return TrackingGame(0, FPS, L, W, meta, team_df[cols], team_df[cols].copy(),
                        pd.DataFrame({"x": np.nan, "y": np.nan}, index=meta.index),
                        events if events is not None else pd.DataFrame(), [], {})


def test_distance_sprints_and_zones_on_known_run():
    # 3 s at 8 m/s (sprint, >= 1 s), then 4 s at 2 m/s (walking)
    v = np.r_[np.full(75, 8.0), np.full(100, 2.0)]
    x = np.cumsum(v / FPS)
    team, meta = _team_one_player(x, np.zeros(len(x)))
    game = _fake_game(team, meta)
    sp, _ = smooth_kinematics(team, meta, FPS)
    row = physical_table(game, "home", sp).iloc[0]
    assert row["sprints"] == 1
    assert row["top_speed_kmh"] == pytest.approx(8 * 3.6, abs=0.5)
    zone_sum = sum(row[c] for c in row.index if c.startswith("dist_"))
    assert zone_sum == pytest.approx(row["distance_m"], abs=5)  # each zone value is rounded
    # true distance of the VALID frames (trimmed edges lose ~12 frames) is close to 8*3+2*4=32 m
    assert 27 < row["distance_m"] < 33


def test_short_burst_is_not_a_sprint():
    v = np.r_[np.full(50, 2.0), np.full(15, 8.0), np.full(60, 2.0)]  # 0.6 s fast
    team, meta = _team_one_player(np.cumsum(v / FPS), np.zeros(len(v)))
    game = _fake_game(team, meta)
    sp, _ = smooth_kinematics(team, meta, FPS)
    assert physical_table(game, "home", sp).iloc[0]["sprints"] == 0


# ---------- tactical metrics ----------

def test_hull_area_of_square():
    assert _hull_area(np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5]], float)) == pytest.approx(100.0)


def test_formation_from_clear_442():
    xs = [20, 22, 21, 23, 45, 46, 44, 47, 70, 72]
    out = formation_from_positions(pd.DataFrame({"x": xs, "y": range(10)}))
    assert out["formation"] == "4-4-2" and out["k"] == 3


def test_possession_series_rules():
    meta = pd.DataFrame({"period": 1, "time_s": 0.0}, index=np.arange(1, 21))
    ev = pd.DataFrame({"team": ["home", "away", "away", "home"],
                       "type": ["PASS", "BALL OUT", "SET PIECE", "CHALLENGE"],
                       "start_frame": [3, 8, 12, 15]})
    game = _fake_game(pd.DataFrame({"Player1_x": 0.0, "Player1_y": 0.0}, index=meta.index),
                      meta, ev)
    p = possession_series(game)
    assert p.loc[1:2].eq("").all()           # before first event
    assert p.loc[3:7].eq("home").all()
    assert p.loc[8:11].eq("").all()          # ball out -> unknown
    assert p.loc[12:20].eq("away").all()     # CHALLENGE ignored


def test_voronoi_two_players_split_pitch_evenly():
    pos = pd.DataFrame({"team": ["home", "away"], "x": [26.25, 78.75], "y": [34.0, 34.0]})
    grid, home_share = voronoi_control(pos, L, W)
    assert home_share == pytest.approx(0.5, abs=0.01)


# ---------- real data (skipped if not downloaded) ----------

real = pytest.mark.skipif(not (raw_dir() / "Sample_Game_1_RawEventsData.csv").exists(),
                          reason="Metrica files not downloaded")


@pytest.fixture(scope="module")
def game1():
    from src.data.metrica import load_game
    return load_game(1)


@real
def test_real_game_basic_facts(game1):
    assert game1.fps == 25.0 and len(game1.meta) == 145006
    assert game1.quality["frame_gaps"] == 0
    gk = goalkeepers(game1)
    # after direction normalisation, home keeper is near x=0 end, away keeper near x=105 end
    assert game1.home[f"{gk['home']}_x"].mean() < 25
    assert game1.away[f"{gk['away']}_x"].mean() > 80


@real
def test_real_game_has_no_impossible_steps_after_cleaning(game1):
    per = game1.meta["period"].to_numpy()
    same = per[1:] == per[:-1]
    for team in (game1.home, game1.away):
        for p in {c[:-2] for c in team.columns}:
            step = np.hypot(np.diff(team[p + "_x"]), np.diff(team[p + "_y"])) * FPS
            assert not (step[same] > 12.0 + 1e-9).any(), p


@real
def test_real_game_speeds_are_physically_plausible(game1):
    sp, _ = smooth_kinematics(game1.home, game1.meta, game1.fps)
    assert np.nanmax(sp.to_numpy()) < 12.0
    table = physical_table(game1, "home", sp)
    outfield = table[~table["is_gk"] & (table["minutes"] > 90)]
    assert outfield["distance_m"].between(7000, 14000).all()
