"""Make the annotated Phase 7 video playable in a browser.

OpenCV writes MPEG-4 Part 2 ('mp4v'), which desktop players open but browsers do not. This
re-encodes it to H.264 with the system `ffmpeg` if one is installed. The video contains
broadcast footage, so the result is for LOCAL viewing only and is git-ignored.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def to_h264(src: Path, dst: Path, width: int = 960) -> bool:
    """Re-encode with the system ffmpeg (H.264, web-friendly). Returns False if no ffmpeg."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(src), "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-crf", "28", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dst)], check=True)
    return True
