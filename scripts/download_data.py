"""Download and cache all configured StatsBomb data into data/raw/.

Run from the repo root:  python -m scripts.download_data
"""
from __future__ import annotations

from src.data.statsbomb import list_competitions, load_all_matches, load_events


def main() -> None:
    from src.data.metrica_download import download_all
    download_all()
    list_competitions()
    matches = load_all_matches()
    print(f"{len(matches)} matches across "
          f"{matches[['competition_id', 'season_id']].drop_duplicates().shape[0]} competition-seasons")
    for i, mid in enumerate(matches["match_id"], 1):
        load_events(mid)
        if i % 20 == 0 or i == len(matches):
            print(f"  cached events {i}/{len(matches)}")


if __name__ == "__main__":
    main()
