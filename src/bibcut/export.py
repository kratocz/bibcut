"""Output files (SPEC §5): intervals.json, intervals.csv, youtube.txt and one ffmpeg cut script
per rider."""

from __future__ import annotations

import csv
import json
import math
import shlex
from pathlib import Path

from bibcut.models import Appearance, RiderTimeline
from bibcut.probe import VideoInfo
from bibcut.timeline import Unresolved


def write_outputs(
    out_dir: Path,
    info: VideoInfo,
    timelines: list[RiderTimeline],
    unresolved: list[Unresolved],
    *,
    encoder: list[str] | None = None,
    youtube_offset: float = 0.0,
) -> list[Path]:
    """Write all outputs into `out_dir`; returns the paths written.

    `encoder` is ffmpeg output options for the video of the cuts (ffmpeg's default without).
    `youtube_offset` shifts the YouTube timestamps, for an upload that does not start where this
    video starts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "video": info.path,
        "riders": [
            {
                "number": t.number,
                "name": t.name,
                "appearances": [
                    {"start": a.start, "end": a.end, "confidence": a.confidence, "source": a.source}
                    for a in t.appearances
                ],
            }
            for t in timelines
        ],
        "unresolved": [
            {"tracklet_id": u.tracklet, "start": u.start, "end": u.end, "votes": u.votes}
            for u in unresolved
        ],
    }
    paths = [out_dir / "intervals.json", out_dir / "intervals.csv"]
    paths[0].write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with paths[1].open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["number", "name", "start", "end", "confidence", "source"])
        for t in timelines:
            for a in t.appearances:
                writer.writerow([t.number, t.name or "", a.start, a.end, a.confidence, a.source])
    paths.append(out_dir / "youtube.txt")
    paths[-1].write_text(
        youtube_description(timelines, start_time=info.start_time, offset=youtube_offset),
        encoding="utf-8",
    )

    cuts = out_dir / "cuts"
    cuts.mkdir(exist_ok=True)
    for t in timelines:
        script = cuts / f"rider-{t.number}.sh"
        command = cut_command(info, t.appearances, Path(f"rider-{t.number}.mp4"), encoder)
        script.write_text(f'#!/bin/sh\ncd "$(dirname "$0")" || exit 1\n{shlex.join(command)}\n')
        script.chmod(0o755)
        paths.append(script)
    return paths


def youtube_description(
    timelines: list[RiderTimeline], *, start_time: float = 0.0, offset: float = 0.0
) -> str:
    """Timestamp lines for a YouTube video description, one line per rider.

    Each timestamp is where an appearance starts, in whole seconds rounded down so the link
    lands just before the rider shows up. Times are PTS minus the container start (what a
    player shows) plus `offset`. YouTube turns `h:mm:ss` in a description into links.
    """
    lines = ["Riders in this video:"]
    for t in timelines:
        stamps = []
        for a in t.appearances:
            stamp = _hms(a.start - start_time + offset)
            if stamp not in stamps:
                stamps.append(stamp)
        who = f"#{t.number} {t.name}" if t.name else f"#{t.number}"
        lines.append(f"- {who}: {' , '.join(stamps)}")
    return "\n".join(lines) + "\n"


def _hms(seconds: float) -> str:
    s = max(0, math.floor(seconds))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def cut_command(
    info: VideoInfo,
    appearances: list[Appearance],
    output: Path,
    encoder: list[str] | None = None,
) -> list[str]:
    """ffmpeg command that joins a rider's appearances into one file.

    Each appearance is its own input trimmed by input seeking, which ffmpeg makes frame-accurate
    when re-encoding and which decodes only the appearance, not the whole video. Times are PTS;
    `-ss` counts from the container start, hence the offset.
    """
    video = str(Path(info.path).resolve())
    command = ["ffmpeg", "-hide_banner", "-nostdin"]
    for a in appearances:
        start = a.start - info.start_time
        end = a.end - info.start_time
        command += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video]
    n = len(appearances)
    audio = info.has_audio
    streams = "".join(f"[{i}:v:0]" + (f"[{i}:a:0]" if audio else "") for i in range(n))
    graph = f"{streams}concat=n={n}:v=1:a={int(audio)}[v]" + ("[a]" if audio else "")
    command += ["-filter_complex", graph, "-map", "[v]"]
    if audio:
        command += ["-map", "[a]"]
    return [*command, *(encoder or []), str(output)]


def videotoolbox_encoder(info: VideoInfo) -> list[str]:
    """HEVC on Apple's media engine at the source's bit rate: about 6x faster than libx264 on
    4K footage (SPEC §12). `hvc1` lets Apple players open the file."""
    rate = info.bit_rate or 20_000_000
    return ["-c:v", "hevc_videotoolbox", "-b:v", str(rate), "-tag:v", "hvc1"]
