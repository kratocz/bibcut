import json
from pathlib import Path

import numpy as np
import pytest

from bibcut.models import Detection, Frame
from bibcut.pipeline import detect_stage, load_readings, load_tracklets, read_stage, track_stage
from bibcut.track import TrackerConfig


class FakeDetector:
    """One person walking left to right at 0.2 frame widths per second."""

    input_size = (16, 12)

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        assert frame.image.shape == (12, 16, 3)
        x = 0.1 + 0.2 * frame.pts
        return [Detection(pts=frame.pts, bbox=(x, 0.4, x + 0.05, 0.6), score=0.9)]


def test_detect_then_track(tiny_video: Path, tmp_path: Path) -> None:
    det = FakeDetector()
    detections, key = detect_stage(tiny_video, tmp_path, det, {"name": "fake"}, sample_fps=10)
    frames = json.loads(detections.read_text())["frames"]
    assert len(frames) == 10
    assert frames[1]["pts"] == 0.12
    tracklets_path, _ = track_stage(tmp_path, key, TrackerConfig())
    (t,) = load_tracklets(tracklets_path)
    assert (t.id, len(t.detections), t.start, t.end) == (1, 10, 0.0, 0.92)


def test_detect_stage_is_cached(tiny_video: Path, tmp_path: Path) -> None:
    det = FakeDetector()
    first = detect_stage(tiny_video, tmp_path, det, {"name": "fake"}, sample_fps=10)
    calls = det.calls
    assert detect_stage(tiny_video, tmp_path, det, {"name": "fake"}, sample_fps=10) == first
    assert det.calls == calls
    detect_stage(tiny_video, tmp_path, det, {"name": "fake"}, sample_fps=5)
    assert det.calls == calls + 5


def test_track_stage_follows_its_config(tiny_video: Path, tmp_path: Path) -> None:
    _, key = detect_stage(tiny_video, tmp_path, FakeDetector(), {"name": "fake"}, sample_fps=10)
    a = track_stage(tmp_path, key, TrackerConfig())[0].read_text()
    b = track_stage(tmp_path, key, TrackerConfig(new_track=0.95))[0].read_text()
    assert json.loads(a)["tracklets"] and not json.loads(b)["tracklets"]


class FakeRecognizer:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def read(self, crop: np.ndarray) -> list[tuple[str, float]]:
        self.shapes.append(crop.shape)
        return [("7", 0.9), ("1", 0.4)]


def _tracked(video: Path, workdir: Path) -> str:
    _, key = detect_stage(video, workdir, FakeDetector(), {"name": "fake"}, sample_fps=10)
    return track_stage(workdir, key, TrackerConfig())[1]


def test_read_stage_reads_every_tall_enough_detection(tiny_video: Path, tmp_path: Path) -> None:
    key = _tracked(tiny_video, tmp_path)
    rec = FakeRecognizer()
    path, _ = read_stage(
        tiny_video, tmp_path, key, rec, {"name": "fake"}, sample_fps=10, min_height_px=5
    )
    # 10 frames of a 0.2-tall box in a 48 px frame: 10 px tall, torso 4 px of a widened 0.05 box.
    assert len(rec.shapes) == 10
    assert rec.shapes[0] == (4, 4, 3)
    assert load_readings(path) == {1: [[("7", 0.9), ("1", 0.4)]] * 10}


def test_read_stage_skips_small_people_and_is_cached(tiny_video: Path, tmp_path: Path) -> None:
    key = _tracked(tiny_video, tmp_path)
    rec = FakeRecognizer()
    path, _ = read_stage(tiny_video, tmp_path, key, rec, {"name": "fake"}, sample_fps=10)
    assert load_readings(path) == {}
    read_stage(tiny_video, tmp_path, key, rec, {"name": "fake"}, sample_fps=10)
    read_stage(tiny_video, tmp_path, key, rec, {"name": "fake"}, sample_fps=10, min_height_px=5)
    assert len(rec.shapes) == 10


def test_read_stage_needs_the_detection_sampling_grid(tiny_video: Path, tmp_path: Path) -> None:
    key = _tracked(tiny_video, tmp_path)
    with pytest.raises(RuntimeError, match="not decoded"):
        read_stage(tiny_video, tmp_path, key, FakeRecognizer(), {"name": "fake"}, sample_fps=5)
