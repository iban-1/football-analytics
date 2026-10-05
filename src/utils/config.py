"""Load project configuration and set the global random seed."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config.yaml (one place for seed, competitions, pitch sizes)."""
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python and NumPy so every run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)


def resolve(rel: str) -> Path:
    """Turn a config path (relative to the repo root) into an absolute path."""
    return ROOT / rel
