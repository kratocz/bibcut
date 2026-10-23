import json
from pathlib import Path

from bibcut.models import Detection, Frame
from bibcut.pipeline import detect_stage, load_tracklets, track_stage
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
    tracklets_path = track_stage(tmp_path, key, TrackerConfig())
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
    a = track_stage(tmp_path, key, TrackerConfig()).read_text()
    b = track_stage(tmp_path, key, TrackerConfig(new_track=0.95)).read_text()
    assert json.loads(a)["tracklets"] and not json.loads(b)["tracklets"]
