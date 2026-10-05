"""Make the annotated video playable in the dashboard.

Run from the repo root:  python -m scripts.make_web_videos
Writes results/phase7/annotated_web.mp4 (H.264) using the ffmpeg installed on this machine.
The file contains broadcast footage: local use only, git-ignored, never publish it.
"""
from __future__ import annotations

from src.utils.config import load_config, resolve
from src.video.web import to_h264


def main() -> None:
    out = resolve(load_config()["paths"]["results"]) / "phase7"
    if to_h264(out / "annotated.mp4", out / "annotated_web.mp4"):
        print(f"annotated_web.mp4 written ({(out / 'annotated_web.mp4').stat().st_size / 1e6:.1f} MB)")
    else:
        print("ffmpeg not found: install it (https://ffmpeg.org) or open annotated.mp4 directly")


if __name__ == "__main__":
    main()
