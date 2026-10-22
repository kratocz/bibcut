import shutil
import subprocess
from pathlib import Path

import pytest


def _make_video(path: Path, *args: str) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25",
         *args, "-c:v", "mpeg4", "-q:v", "2", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1 s, 64x48, 25 fps synthetic clip, generated rather than committed."""
    return _make_video(tmp_path_factory.mktemp("video") / "tiny.mp4", "-t", "1")


@pytest.fixture(scope="session")
def vfr_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """10 frames whose timestamps are n*n/25 s — anything but index x fps.

    The values stay on mpeg4's 1/25 s encoder grid, so the encoder writes them unchanged.
    """
    return _make_video(
        tmp_path_factory.mktemp("video") / "vfr.mp4",
        "-frames:v", "10",
        "-vf", "setpts=N*N/25/TB",
        "-fps_mode", "passthrough",
        "-video_track_timescale", "1000",
    )  # fmt: skip
