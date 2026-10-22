"""Frame decoding through the ffmpeg CLI (SPEC §6, step 1).

ffmpeg runs as a subprocess rather than a linked library, so the core stays free of the GPL
components that binary FFmpeg bundles (e.g. PyAV wheels) ship with.

Raw RGB frames come from ffmpeg's stdout. Their timestamps come from the `showinfo` filter on
stderr, which logs every frame in the same order, so time is always the stream's PTS and never
frame index x fps.
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from bibcut.models import Frame
from bibcut.probe import probe

_SHOWINFO = re.compile(r"Parsed_showinfo.*\bpts_time:\s*(-?[\d.]+)")
_GOPRO = re.compile(r"G[XH](\d{2})(\d{4})", re.IGNORECASE)
# Half a millisecond of slack so a frame exactly on the sampling grid is not skipped by rounding.
_GRID_SLACK = 5e-4
_DONE = object()


def proxy_for(video: Path) -> Path | None:
    """The GoPro low-resolution proxy (`GL…LRV`) next to `video`, if there is one.

    Proxies share the exact frame timeline of their MP4 (SPEC §12), so detection and tracking can
    run on them and only number reading needs the full-resolution file.
    """
    m = _GOPRO.fullmatch(video.stem)
    if m is None:
        return None
    proxy = video.with_name(f"GL{m[1]}{m[2]}.LRV")
    return proxy if proxy.is_file() else None


def iter_frames(
    video: Path,
    *,
    sample_fps: float | None = None,
    size: tuple[int, int] | None = None,
    start: float | None = None,
    end: float | None = None,
    hwaccel: str | None = None,
) -> Iterator[Frame]:
    """Yield RGB frames of `video` in presentation order.

    sample_fps: keep a frame only once at least 1/sample_fps s have passed since the last kept
        one; timestamps stay those of the source frames.
    size: (width, height) to scale to; defaults to the stream's size.
    start, end: limit to [start, end) seconds on the source timeline.
    hwaccel: ffmpeg hardware decoder, e.g. "videotoolbox".
    """
    width, height = size or _native_size(video)
    filters = []
    if sample_fps is not None:
        step = 1 / sample_fps - _GRID_SLACK
        filters.append(f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step})'")
    filters += [f"scale={width}:{height}", "format=rgb24", "showinfo"]

    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-v", "info"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-copyts", "-i", str(video), "-an", "-sn", "-dn"]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-vf", ",".join(filters), "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    stamps: queue.Queue[object] = queue.Queue()
    log: list[str] = []
    reader = threading.Thread(target=_read_stderr, args=(proc.stderr, stamps, log), daemon=True)
    reader.start()

    frame_bytes = width * height * 3
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            pts = stamps.get()
            if pts is _DONE:
                raise RuntimeError(f"ffmpeg wrote a frame without a timestamp: {video}")
            image = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            yield Frame(index=index, pts=float(pts), image=image)  # type: ignore[arg-type]
            index += 1
        if proc.wait() != 0:
            tail = "\n".join(log[-10:])
            raise RuntimeError(f"ffmpeg failed on {video} (exit {proc.returncode}):\n{tail}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        reader.join()
        proc.stderr.close()


def _native_size(video: Path) -> tuple[int, int]:
    try:
        info = probe(video)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg/ffprobe cannot read {video}:\n{e.stderr.strip()}") from e
    return info.width, info.height


def _read_stderr(stream, stamps: queue.Queue[object], log: list[str]) -> None:
    for raw in stream:
        line = raw.decode(errors="replace").rstrip()
        m = _SHOWINFO.search(line)
        if m:
            stamps.put(float(m[1]))
        elif line:
            log.append(line)
    stamps.put(_DONE)
