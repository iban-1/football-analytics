"""One-command setup for a fresh clone:  python -m scripts.setup

1. downloads and caches the StatsBomb and Metrica data (data/raw/, git-ignored)
2. builds any result artefact the dashboard needs that is not already in results/
   (the saved xG model, Phase 3 tables, Phase 4 tracking tables)
Then start the dashboard with:  streamlit run app/Home.py
"""
from __future__ import annotations

from src.utils.config import load_config, resolve

NEEDED = {
    "xg_model.joblib": ("scripts.reproduce_xg", "train and evaluate the xG models (~10 min)"),
    "phase3/penalty_rate.json": ("scripts.phase3_demo", "event-analysis tables and figures"),
    "phase4/physical_game1.csv": ("scripts.phase4_tracking", "tracking tables and figures"),
}


def main() -> None:
    from scripts import download_data
    print("Step 1/2: downloading data (cached after the first run)")
    download_data.main()

    results = resolve(load_config()["paths"]["results"])
    print("\nStep 2/2: building missing result files")
    for rel, (module, what) in NEEDED.items():
        if (results / rel).exists():
            print(f"  have {rel}")
            continue
        print(f"  missing {rel}: running {module} to {what}")
        __import__(module, fromlist=["main"]).main()
    print("\nSetup complete. Start the dashboard with:  streamlit run app/Home.py")


if __name__ == "__main__":
    main()
