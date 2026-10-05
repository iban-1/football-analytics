"""Download Metrica Sports sample games 1 and 2 (CSV format) into data/raw/metrica.

Source: https://github.com/metrica-sports/sample-data (no formal licence; the
repo asks users to be responsible and to acknowledge the source in public use).
Sample Game 3 uses a different format (EPTS tracking + JSON events) and is not
used here.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from src.utils.config import load_config, resolve

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
GAMES = (1, 2)


def raw_dir() -> Path:
    d = resolve(load_config()["paths"]["raw"]) / "metrica"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_names(game: int) -> dict[str, str]:
    g = f"Sample_Game_{game}"
    return {"events": f"{g}_RawEventsData.csv",
            "home": f"{g}_RawTrackingData_Home_Team.csv",
            "away": f"{g}_RawTrackingData_Away_Team.csv"}


def download_game(game: int) -> dict[str, Path]:
    """Fetch the three CSVs of one game if not already cached; return their paths."""
    paths = {}
    for kind, name in file_names(game).items():
        dest = raw_dir() / name
        if not dest.exists():
            url = f"{BASE}/Sample_Game_{game}/{name}"
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, dest)
        paths[kind] = dest
    return paths


def download_all() -> None:
    for g in GAMES:
        download_game(g)
