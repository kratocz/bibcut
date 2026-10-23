import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from bibcut.export import videotoolbox_encoder, write_outputs
from bibcut.models import Appearance, RiderTimeline
from bibcut.probe import probe
from bibcut.timeline import Unresolved


def _timelines() -> list[RiderTimeline]:
    return [
        RiderTimeline("7", "Řidič Sedm", [Appearance(0.5, 1.5, 0.9), Appearance(2.5, 3.0, 0.8)]),
        RiderTimeline("12", None, [Appearance(1.0, 2.0, 1.0)]),
    ]


def test_writes_json_and_csv(av_video: Path, tmp_path: Path) -> None:
    unresolved = [Unresolved(4, 1.0, 2.5, (("98", 1.2),))]
    write_outputs(tmp_path, probe(av_video), _timelines(), unresolved)
    data = json.loads((tmp_path / "intervals.json").read_text(encoding="utf-8"))
    assert data["riders"][0]["name"] == "Řidič Sedm"
    assert data["riders"][0]["appearances"][1] == {
        "start": 2.5, "end": 3.0, "confidence": 0.8, "source": "auto"
    }  # fmt: skip
    assert data["unresolved"] == [
        {"tracklet_id": 4, "start": 1.0, "end": 2.5, "votes": [["98", 1.2]]}
    ]
    rows = list(csv.DictReader((tmp_path / "intervals.csv").open(encoding="utf-8")))
    assert [(r["number"], r["start"], r["end"]) for r in rows] == [
        ("7", "0.5", "1.5"), ("7", "2.5", "3.0"), ("12", "1.0", "2.0")
    ]  # fmt: skip


@pytest.mark.parametrize("fixture", ["av_video", "tiny_video"])
def test_cut_scripts_produce_the_appearances(
    fixture: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    video: Path = request.getfixturevalue(fixture)
    info = probe(video)
    short = [RiderTimeline("12", None, [Appearance(0.2, 0.8, 1.0)])]
    timelines = _timelines() if info.duration > 3 else short
    write_outputs(tmp_path, info, timelines, [])
    number = timelines[0].number
    subprocess.run([str(tmp_path / "cuts" / f"rider-{number}.sh")], check=True, capture_output=True)
    out = probe(tmp_path / "cuts" / f"rider-{number}.mp4")
    expected = sum(a.end - a.start for a in timelines[0].appearances)
    assert out.duration == pytest.approx(expected, abs=0.1)
    assert out.has_audio == info.has_audio


@pytest.mark.skipif(sys.platform != "darwin", reason="VideoToolbox is macOS only")
def test_cut_with_videotoolbox(av_video: Path, tmp_path: Path) -> None:
    info = probe(av_video)
    timelines = [RiderTimeline("7", None, [Appearance(0.5, 1.5, 1.0)])]
    write_outputs(tmp_path, info, timelines, [], encoder=videotoolbox_encoder(info))
    subprocess.run([str(tmp_path / "cuts" / "rider-7.sh")], check=True, capture_output=True)
    out = probe(tmp_path / "cuts" / "rider-7.mp4")
    assert out.codec == "hevc"
    assert out.duration == pytest.approx(1.0, abs=0.1)
