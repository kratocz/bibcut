"""Video metadata via the ffprobe CLI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: str
    codec: str
    width: int
    height: int
    duration: float
    # Nominal rate from the container; time itself always comes from PTS.
    fps: float
    # True when the stream declares a varying frame rate (avg differs from the base rate).
    variable_frame_rate: bool
    creation_time: str | None


def _rate(value: str) -> Fraction:
    num, _, den = value.partition("/")
    return Fraction(int(num), int(den or 1)) if den != "0" else Fraction(0)


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip
    data = json.loads(result.stdout)
    video = next((s for s in data["streams"] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    base = _rate(video["r_frame_rate"])
    avg = _rate(video.get("avg_frame_rate", "0/0"))
    fmt = data["format"]
    return VideoInfo(
        path=str(path),
        codec=video["codec_name"],
        width=int(video["width"]),
        height=int(video["height"]),
        duration=float(fmt.get("duration") or video.get("duration") or 0.0),
        fps=float(avg or base),
        variable_frame_rate=bool(avg) and avg != base,
        creation_time=fmt.get("tags", {}).get("creation_time"),
    )
