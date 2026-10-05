"""Smoke tests: every dashboard page runs without an exception (needs cached data)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

from src.data.metrica_download import raw_dir  # noqa: E402
from src.data.statsbomb import _cache_dir  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

RES = resolve(load_config()["paths"]["results"])
have_sb = (_cache_dir() / "competitions.pkl").exists() and (RES / "xg_model.joblib").exists() \
    and (RES / "phase3" / "penalty_rate.json").exists()
have_mt = (raw_dir() / "Sample_Game_1_RawEventsData.csv").exists()

PAGES = ["Home.py", "pages/1_Team_Analysis.py", "pages/2_Player_Analysis.py",
         "pages/3_xG_Model.py"]


@pytest.mark.skipif(not have_sb, reason="StatsBomb cache / results missing")
@pytest.mark.parametrize("page", PAGES)
def test_statsbomb_pages_run(page):
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not have_sb, reason="StatsBomb cache / results missing")
def test_shot_calculator_responds_to_distance():
    at = AppTest.from_file(str(APP / "pages/3_xG_Model.py"), default_timeout=120).run()
    near = float(at.metric[0].value)
    at.slider[0].set_value(75.0).run()  # x: move the shot far from goal
    far = float(at.metric[0].value)
    assert 0 < far < near < 1


@pytest.mark.skipif(not (have_sb and (RES / "phase7").exists()), reason="video phase not run")
def test_video_page_runs():
    at = AppTest.from_file(str(APP / "pages/5_Video.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not (have_sb and have_mt), reason="data missing")
def test_tracking_page_runs():
    at = AppTest.from_file(str(APP / "pages/4_Tracking.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
