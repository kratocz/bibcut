from pathlib import Path

import pytest

from bibcut.decode import iter_frames, proxy_for


def test_cfr_frames_carry_pts(tiny_video: Path) -> None:
    frames = list(iter_frames(tiny_video))
    assert len(frames) == 25
    assert [f.index for f in frames] == list(range(25))
    assert [f.pts for f in frames] == pytest.approx([i / 25 for i in range(25)], abs=1e-3)
    assert frames[0].image.shape == (48, 64, 3)


def test_vfr_time_comes_from_pts_not_index(vfr_video: Path) -> None:
    pts = [f.pts for f in iter_frames(vfr_video)]
    assert pts == pytest.approx([n * n / 25 for n in range(10)], abs=1e-3)


def test_sample_fps_keeps_a_time_grid(tiny_video: Path) -> None:
    pts = [f.pts for f in iter_frames(tiny_video, sample_fps=5)]
    assert pts == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8], abs=1e-3)


def test_sample_fps_on_vfr_keeps_original_timestamps(vfr_video: Path) -> None:
    pts = [f.pts for f in iter_frames(vfr_video, sample_fps=1)]
    # 0, then 5*5/25 = 1.0 (exactly on the grid), then 8*8/25 = 2.56 (first >= 2.0)
    assert pts == pytest.approx([0.0, 1.0, 2.56], abs=1e-3)


def test_resize(tiny_video: Path) -> None:
    frame = next(iter(iter_frames(tiny_video, size=(32, 24))))
    assert frame.image.shape == (24, 32, 3)


def test_start_end_keep_the_source_timeline(tiny_video: Path) -> None:
    pts = [f.pts for f in iter_frames(tiny_video, start=0.4, end=0.6)]
    assert pts == pytest.approx([0.4, 0.44, 0.48, 0.52, 0.56], abs=1e-3)


def test_stopping_early_does_not_hang(tiny_video: Path) -> None:
    frames = iter_frames(tiny_video)
    next(frames)
    frames.close()


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="ffmpeg"):
        list(iter_frames(tmp_path / "nope.mp4", size=(8, 8)))


@pytest.mark.parametrize(
    ("video", "proxy"),
    [
        ("GX010217.MP4", "GL010217.LRV"),
        ("GX020207.MP4", "GL020207.LRV"),
        ("GH010001.MP4", "GL010001.LRV"),
    ],
)
def test_proxy_for_gopro_names(tmp_path: Path, video: str, proxy: str) -> None:
    (tmp_path / video).touch()
    assert proxy_for(tmp_path / video) is None
    (tmp_path / proxy).touch()
    assert proxy_for(tmp_path / video) == tmp_path / proxy


def test_proxy_for_other_names(tmp_path: Path) -> None:
    (tmp_path / "race.mp4").touch()
    assert proxy_for(tmp_path / "race.mp4") is None
