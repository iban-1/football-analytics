"""Shared helpers for every dashboard page: paths, cached loaders, attribution.

Caching rules (Streamlit):
  * st.cache_data     - pure data returned as copies (DataFrames, numbers)
  * st.cache_resource - large objects shared as-is (model, tracking game)
Heavy results (tracking tables, tournament player table, model comparison) are
read from files in results/ that the scripts in scripts/ produce.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.analysis.events import shots_with_xg  # noqa: E402
from src.data.statsbomb import load_all_matches, load_events, load_nicknames  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

RESULTS = resolve(load_config()["paths"]["results"])
LOGO = ROOT / "app" / "assets" / "statsbomb_logo.png"


def setup_page(title: str) -> None:
    st.set_page_config(page_title=f"{title} · Football analytics", layout="wide")
    st.title(title)


def statsbomb_attribution() -> None:
    """Required by StatsBomb: name the source and show their logo."""
    st.divider()
    cols = st.columns([1, 6])
    if LOGO.exists():
        cols[0].image(str(LOGO), width=110)
    cols[1].caption("Event data: **StatsBomb** open data "
                    "(https://github.com/statsbomb/open-data). See that repository for "
                    "StatsBomb's terms of use. The analysis and views here are not StatsBomb's.")
    if not LOGO.exists():
        cols[1].caption("(Add the StatsBomb logo from their Media Pack as "
                        "app/assets/statsbomb_logo.png to display it here.)")


def metrica_attribution() -> None:
    st.divider()
    st.caption("Tracking data: **Metrica Sports** sample data "
               "(https://github.com/metrica-sports/sample-data), anonymised, used responsibly "
               "with acknowledgement of the source as the repository requests.")


# ---------- cached loaders ----------

@st.cache_data(show_spinner="Loading match list…")
def matches() -> pd.DataFrame:
    """Matches of the configured competitions with a readable label."""
    cfg = load_config()["statsbomb"]["competitions"]
    names = {(c["competition_id"], c["season_id"]): c["name"] for c in cfg}
    m = load_all_matches()
    m["competition"] = [names[(a, b)] for a, b in zip(m["competition_id"], m["season_id"])]
    m["label"] = (m["home_team"] + " " + m["home_score"].astype(str) + "-"
                  + m["away_score"].astype(str) + " " + m["away_team"]
                  + "  (" + m["match_date"].astype(str) + ")")
    return m.sort_values("match_date").reset_index(drop=True)


@st.cache_data(show_spinner="Loading match events…", max_entries=12)
def match_events(match_id: int) -> pd.DataFrame:
    ev = load_events(match_id).copy()
    ev["match_id"] = match_id
    return ev


@st.cache_resource
def xg_model():
    return joblib.load(RESULTS / "xg_model.joblib")  # our own file, written by reproduce_xg


@st.cache_data
def penalty_rate() -> float:
    return float(json.loads((RESULTS / "phase3" / "penalty_rate.json").read_text())
                 ["penalty_conversion_rate"])


@st.cache_data(show_spinner="Computing shot xG…", max_entries=12)
def match_shots(match_id: int) -> pd.DataFrame:
    return shots_with_xg(match_events(match_id), penalty_rate(), xg_model())


@st.cache_data(max_entries=12)
def nicknames(match_id: int) -> dict[str, str]:
    try:
        return load_nicknames(match_id)
    except Exception:  # offline or missing lineup: fall back to short names
        return {}


def match_selector(container=None) -> pd.Series:
    """Competition + match pickers whose choice persists across pages.

    Shown in the sidebar by default; pass `container=st` to put them in the page body.
    """
    box = st.sidebar if container is None else container
    m = matches()
    comps = list(dict.fromkeys(m["competition"]))
    comp = box.selectbox("Competition", comps, key="competition")
    sub = m[m["competition"] == comp]
    default = len(sub) - 1  # final of the competition
    label = box.selectbox("Match", sub["label"].tolist(), index=default, key=f"match_{comp}")
    return sub[sub["label"] == label].iloc[0]
