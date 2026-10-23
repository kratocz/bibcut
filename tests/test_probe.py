import json
from pathlib import Path

import pytest

from bibcut.cli import main
from bibcut.probe import probe


def test_probe_reads_basic_stream_info(tiny_video: Path) -> None:
    info = probe(tiny_video)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(25)
    assert info.duration == pytest.approx(1, abs=0.1)
    assert not info.variable_frame_rate
    assert info.start_time == 0.0
    assert not info.has_audio


def test_probe_sees_audio(av_video: Path) -> None:
    assert probe(av_video).has_audio


def test_cli_probe_prints_json(tiny_video: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe", str(tiny_video)]) == 0
    assert json.loads(capsys.readouterr().out)["width"] == 64


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("bibcut ")
