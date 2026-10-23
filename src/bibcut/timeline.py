"""Per-rider timelines from identified tracklets (SPEC §6, step 8)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from bibcut.config import Config
from bibcut.identify import Identity
from bibcut.models import Appearance, RiderTimeline, Tracklet


@dataclass(frozen=True)
class Unresolved:
    """A tracklet whose number was read but not confidently enough to credit anyone."""

    tracklet: int
    start: float
    end: float
    votes: tuple[tuple[str, float], ...]


@dataclass
class _Span:
    start: float
    end: float
    # Summed confidence for the credited number, and for all numbers, over its tracklets.
    support: float
    total: float


def build_timelines(
    tracklets: Iterable[Tracklet],
    identities: dict[int, Identity],
    config: Config,
    *,
    bounds: tuple[float, float],
    names: dict[str, str] | None = None,
) -> tuple[list[RiderTimeline], list[Unresolved]]:
    """Group tracklets by credited number into appearances.

    Tracklets of one number closer than `merge_gap` join into one span; spans shorter than
    `min_appearance` are dropped; the rest are widened by `padding` (clamped to `bounds`, the
    processed part of the video) and joined again where they now overlap. An appearance's
    confidence is the share of all votes in its tracklets that went to its number.
    """
    spans: dict[str, list[_Span]] = defaultdict(list)
    unresolved = []
    for t in tracklets:
        identity = identities.get(t.id)
        if identity is None or not identity.votes:
            continue
        if identity.number is None:
            unresolved.append(Unresolved(t.id, t.start, t.end, identity.votes))
            continue
        votes = dict(identity.votes)
        spans[identity.number].append(
            _Span(t.start, t.end, votes[identity.number], sum(votes.values()))
        )

    timelines = []
    for number in sorted(spans, key=_number_order):
        merged = _merge(spans[number], config.merge_gap)
        kept = [s for s in merged if s.end - s.start >= config.min_appearance]
        padded = [
            _Span(
                max(bounds[0], s.start - config.padding),
                min(bounds[1], s.end + config.padding),
                s.support,
                s.total,
            )
            for s in kept
        ]
        appearances = [
            Appearance(round(s.start, 3), round(s.end, 3), round(s.support / s.total, 3))
            for s in _merge(padded, 0.0)
        ]
        if appearances:
            name = (names or {}).get(number) or None
            timelines.append(RiderTimeline(number, name, appearances))
    return timelines, sorted(unresolved, key=lambda u: (u.start, u.tracklet))


def _merge(spans: list[_Span], gap: float) -> list[_Span]:
    out: list[_Span] = []
    for s in sorted(spans, key=lambda s: (s.start, s.end)):
        if out and s.start - out[-1].end <= gap:
            last = out[-1]
            last.end = max(last.end, s.end)
            last.support += s.support
            last.total += s.total
        else:
            out.append(_Span(s.start, s.end, s.support, s.total))
    return out


def _number_order(number: str) -> tuple[int, str]:
    return (int(number), number) if number.isdigit() else (1 << 30, number)
