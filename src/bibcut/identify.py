"""Who is in each tracklet: start list and per-tracklet voting (SPEC §6, steps 6-7)."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def load_startlist(path: Path) -> dict[str, str]:
    """Start numbers to names from a CSV with `number` and `name` columns (extra columns ok)."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "number" not in reader.fieldnames:
            raise ValueError(f"{path}: needs a 'number' column")
        return {row["number"].strip(): (row.get("name") or "").strip() for row in reader}


@dataclass(frozen=True)
class Identity:
    tracklet: int
    # None when the vote is too weak to name anyone.
    number: str | None
    # Summed confidence per number, strongest first.
    votes: tuple[tuple[str, float], ...]


def vote(
    tracklet: int,
    crops: Iterable[list[tuple[str, float]]],
    *,
    valid: set[str] | None = None,
    min_votes: float = 2.0,
) -> Identity:
    """Confidence-weighted vote over a tracklet's crops.

    Each crop gives its most confident reading that is on the start list (any reading without
    one); a number needs `min_votes` summed confidence to be credited. The 2.0 default was set
    on one labelled segment (SPEC §12) and is a hypothesis, not a measured optimum.
    """
    totals: dict[str, float] = defaultdict(float)
    for candidates in crops:
        for text, confidence in candidates:
            if valid is None or text in valid:
                totals[text] += confidence
                break
    votes = tuple(sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))
    number = votes[0][0] if votes and votes[0][1] >= min_votes else None
    return Identity(tracklet, number, votes)
