from pathlib import Path

import pytest

from bibcut.config import Config, load_config


def test_no_file_gives_defaults() -> None:
    assert load_config(None) == Config()


def test_loads_roi_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "bibcut.toml"
    path.write_text("merge_gap = 3.5\nroi = [[0, 0], [100, 0], [100, 50]]\n")
    cfg = load_config(path)
    assert cfg.merge_gap == 3.5
    assert cfg.roi == ((0.0, 0.0), (100.0, 0.0), (100.0, 50.0))
    assert cfg.padding == Config().padding


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bibcut.toml"
    path.write_text("merge_gpa = 3\n")
    with pytest.raises(ValueError, match="merge_gpa"):
        load_config(path)
