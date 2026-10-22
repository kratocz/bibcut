import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1 s, 64x48, 25 fps synthetic clip, generated rather than committed."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25",
            "-t", "1", "-c:v", "mpeg4",
            str(path),
        ],
        check=True,
    )  # fmt: skip
    return path
