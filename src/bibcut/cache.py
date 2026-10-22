"""Per-stage output cache (AGENTS.md: every stage caches its output).

A stage writes into `<workdir>/<stage>/` and records the key it was computed from in
`key.txt`. The key covers the input video, the stage's parameters and the keys of the stages it
read from, so changing any of them invalidates everything downstream.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

_EDGE = 1 << 20  # bytes hashed from each end of a video


def fingerprint_file(path: Path) -> str:
    """Cheap, stable identity of a large file: its size plus hashes of its first and last MiB."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(_EDGE))
        if size > _EDGE:
            f.seek(max(_EDGE, size - _EDGE))
            h.update(f.read(_EDGE))
    return h.hexdigest()


def stage_key(**parts: Any) -> str:
    """Hash of JSON-serialisable parts; order of keyword arguments does not matter."""
    blob = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class StageCache:
    def __init__(self, workdir: Path, stage: str, key: str) -> None:
        self.dir = workdir / stage
        self.key = key

    def is_valid(self) -> bool:
        marker = self.dir / "key.txt"
        return marker.is_file() and marker.read_text().strip() == self.key

    def begin(self) -> Path:
        """Clear any stale output and return the directory the stage should write into."""
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        return self.dir

    def commit(self) -> None:
        """Mark the output complete. Written last, so an interrupted stage stays invalid."""
        (self.dir / "key.txt").write_text(self.key + "\n")
