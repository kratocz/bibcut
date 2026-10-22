from pathlib import Path

from bibcut.cache import StageCache, fingerprint_file, stage_key


def test_stage_key_ignores_argument_order() -> None:
    assert stage_key(a=1, b=[2, 3]) == stage_key(b=[2, 3], a=1)
    assert stage_key(a=1) != stage_key(a=2)


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    f = tmp_path / "v.bin"
    f.write_bytes(b"x" * 3_000_000)
    before = fingerprint_file(f)
    f.write_bytes(b"x" * 2_999_999 + b"y")
    assert fingerprint_file(f) != before


def test_stage_valid_only_after_commit(tmp_path: Path) -> None:
    cache = StageCache(tmp_path, "decode", key="k1")
    assert not cache.is_valid()
    out = cache.begin()
    (out / "frames.json").write_text("[]")
    assert not cache.is_valid()
    cache.commit()
    assert cache.is_valid()
    assert not StageCache(tmp_path, "decode", key="k2").is_valid()


def test_begin_clears_stale_output(tmp_path: Path) -> None:
    StageCache(tmp_path, "decode", key="k1").begin().joinpath("old.json").write_text("{}")
    out = StageCache(tmp_path, "decode", key="k2").begin()
    assert list(out.iterdir()) == []
