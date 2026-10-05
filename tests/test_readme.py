"""The README's results block must be exactly what build_summary generates from results/."""
import re

import pytest

from src.utils.config import ROOT

README = ROOT / "README.md"
BLOCK = ROOT / "results" / "README_results_block.md"


def _readme_block() -> str:
    text = README.read_text(encoding="utf-8")
    m = re.search(r"<!-- RESULTS:START.*?<!-- RESULTS:END -->", text, re.S)
    assert m, "README has no RESULTS block"
    return m.group(0)


@pytest.mark.skipif(not BLOCK.exists(), reason="run scripts.build_summary first")
def test_readme_results_match_generated_block():
    assert _readme_block() == BLOCK.read_text(encoding="utf-8")


def test_readme_names_required_attribution():
    text = README.read_text(encoding="utf-8")
    assert "StatsBomb" in text and "Metrica" in text
    assert "states the source as StatsBomb and uses their logo" in text


def test_gitignore_excludes_raw_data_and_secrets():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/raw/*", "data/processed/*", ".env", ".venv/"):
        assert pattern in ignore
