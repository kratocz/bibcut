"""Cached pipeline stages (SPEC §6). Each writes JSON under `<workdir>/<stage>/`."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bibcut.cache import StageCache, fingerprint_file, stage_key
from bibcut.crop import crop, torso_box
from bibcut.decode import iter_frames, proxy_for
from bibcut.interfaces import NumberRecognizer, PersonDetector
from bibcut.models import Detection, Tracklet
from bibcut.track import Tracker, TrackerConfig

Progress = Callable[[str], None]


def detect_stage(
    video: Path,
    workdir: Path,
    detector: PersonDetector,
    detector_key: dict[str, Any],
    *,
    sample_fps: float,
    start: float | None = None,
    end: float | None = None,
    use_proxy: bool = True,
    hwaccel: str | None = None,
    progress: Progress | None = None,
) -> tuple[Path, str]:
    """Detect people on sampled frames; returns (detections.json, stage key).

    With `use_proxy`, a GoPro LRV proxy is read instead of the video when one exists; boxes are
    normalized, so they apply to the full-resolution file unchanged.
    """
    source = (proxy_for(video) if use_proxy else None) or video
    key = stage_key(
        stage="detect",
        source=fingerprint_file(source),
        detector=detector_key,
        input_size=list(detector.input_size),
        sample_fps=sample_fps,
        start=start,
        end=end,
    )
    cache = StageCache(workdir, "detect", key)
    out = cache.dir / "detections.json"
    if cache.is_valid():
        return out, key

    cache.begin()
    frames = []
    began = time.monotonic()
    for frame in iter_frames(
        source, sample_fps=sample_fps, size=detector.input_size, start=start, end=end,
        hwaccel=hwaccel,
    ):  # fmt: skip
        dets = detector.detect(frame)
        frames.append({"pts": _r(frame.pts), "detections": [_det_row(d) for d in dets]})
        if progress and len(frames) % 50 == 0:
            progress(f"detect: {len(frames)} frames, t={frame.pts:.1f}s, "
                     f"{time.monotonic() - began:.0f}s elapsed")  # fmt: skip
    _write_json(out, {"video": str(video), "source": str(source), "frames": frames})
    cache.commit()
    return out, key


def track_stage(workdir: Path, detect_key: str, config: TrackerConfig) -> tuple[Path, str]:
    """Track people across the cached detections; returns (tracklets.json, stage key)."""
    key = stage_key(stage="track", detect=detect_key, config=dataclasses.asdict(config))
    cache = StageCache(workdir, "track", key)
    out = cache.dir / "tracklets.json"
    if cache.is_valid():
        return out, key

    data = json.loads((workdir / "detect" / "detections.json").read_text())
    tracker = Tracker(config)
    for f in data["frames"]:
        tracker.update(f["pts"], [_det_from_row(f["pts"], row) for row in f["detections"]])
    cache.begin()
    tracklets = [
        {"id": t.id, "detections": [[d.pts, *_det_row(d)] for d in t.detections]}
        for t in tracker.finish()
    ]
    _write_json(out, {"tracklets": tracklets})
    cache.commit()
    return out, key


def read_stage(
    video: Path,
    workdir: Path,
    track_key: str,
    recognizer: NumberRecognizer,
    recognizer_key: dict[str, Any],
    *,
    sample_fps: float,
    start: float | None = None,
    end: float | None = None,
    min_height_px: int = 100,
    hwaccel: str | None = None,
    progress: Progress | None = None,
) -> tuple[Path, str]:
    """Read start numbers from torso crops of the full-resolution video; returns (readings.json,
    stage key).

    Frames are decoded on the sampling grid detection used, so every tracked detection's PTS comes
    round again. Detections shorter than `min_height_px` in the full-resolution frame are skipped:
    their number is too small to read.
    """
    key = stage_key(
        stage="read",
        track=track_key,
        source=fingerprint_file(video),
        recognizer=recognizer_key,
        sample_fps=sample_fps,
        start=start,
        end=end,
        min_height_px=min_height_px,
    )
    cache = StageCache(workdir, "read", key)
    out = cache.dir / "readings.json"
    if cache.is_valid():
        return out, key

    wanted: dict[float, list[tuple[int, Detection]]] = {}
    for t in load_tracklets(workdir / "track" / "tracklets.json"):
        for d in t.detections:
            wanted.setdefault(_r(d.pts), []).append((t.id, d))
    cache.begin()
    readings = []
    began = time.monotonic()
    for frame in iter_frames(video, sample_fps=sample_fps, start=start, end=end, hwaccel=hwaccel):
        for tracklet_id, d in wanted.pop(_r(frame.pts), []):
            height_px = round((d.bbox[3] - d.bbox[1]) * frame.image.shape[0])
            if height_px < min_height_px:
                continue
            candidates = recognizer.read(crop(frame.image, torso_box(d.bbox)))
            readings.append({
                "tracklet": tracklet_id,
                "pts": d.pts,
                "height_px": height_px,
                "candidates": [[text, _r(c)] for text, c in candidates],
            })  # fmt: skip
        if progress and frame.index % 50 == 49:
            progress(f"read: t={frame.pts:.1f}s, {len(readings)} crops, "
                     f"{time.monotonic() - began:.0f}s elapsed")  # fmt: skip
    if wanted:
        raise RuntimeError(
            f"{video}: {len(wanted)} tracked frames (first at {min(wanted)}s) not decoded; "
            "was it tracked with the same sample rate and time range?"
        )
    readings.sort(key=lambda r: (r["tracklet"], r["pts"]))
    _write_json(out, {"video": str(video), "readings": readings})
    cache.commit()
    return out, key


def load_readings(path: Path) -> dict[int, list[list[tuple[str, float]]]]:
    """Candidates of every read crop, grouped by tracklet."""
    crops: dict[int, list[list[tuple[str, float]]]] = {}
    for r in json.loads(path.read_text())["readings"]:
        crops.setdefault(r["tracklet"], []).append([(t, c) for t, c in r["candidates"]])
    return crops


def load_tracklets(path: Path) -> list[Tracklet]:
    data = json.loads(path.read_text())
    return [
        Tracklet(id=t["id"], detections=[_det_from_row(row[0], row[1:]) for row in t["detections"]])
        for t in data["tracklets"]
    ]


def stderr_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _r(x: float) -> float:
    # Rounded so cached output is compact and stable across platforms.
    return round(float(x), 6)


def _det_row(d: Detection) -> list[float]:
    return [_r(v) for v in d.bbox] + [_r(d.score)]


def _det_from_row(pts: float, row: list[float]) -> Detection:
    x1, y1, x2, y2, score = row
    return Detection(pts=pts, bbox=(x1, y1, x2, y2), score=score)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
