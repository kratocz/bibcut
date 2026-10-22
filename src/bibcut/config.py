"""Run configuration (SPEC §4), loaded from an optional TOML file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

Point = tuple[float, float]


@dataclass(frozen=True)
class Config:
    # Track region of interest in frame pixels; empty means the whole frame.
    roi: tuple[Point, ...] = ()
    # Placeholder defaults, to be tuned on real footage.
    merge_gap: float = 2.0
    padding: float = 0.5
    min_appearance: float = 1.0


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as f:
        data = tomllib.load(f)
    known = {f.name for f in fields(Config)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {', '.join(sorted(unknown))}")
    if "roi" in data:
        data["roi"] = tuple((float(x), float(y)) for x, y in data["roi"])
    return Config(**data)
