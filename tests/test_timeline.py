import pytest

from bibcut.config import Config
from bibcut.identify import Identity
from bibcut.models import Appearance, Detection, Tracklet
from bibcut.timeline import Unresolved, build_timelines


def _tracklet(id: int, start: float, end: float) -> Tracklet:
    box = (0.1, 0.1, 0.2, 0.5)
    return Tracklet(id, [Detection(start, box, 0.9), Detection(end, box, 0.9)])


CONFIG = Config(merge_gap=2.0, padding=0.5, min_appearance=1.0)


def test_close_tracklets_merge_and_get_padded() -> None:
    tracklets = [_tracklet(1, 10.0, 12.0), _tracklet(2, 13.5, 15.0), _tracklet(3, 30.0, 33.0)]
    ids = {
        1: Identity(1, "7", (("7", 3.0), ("1", 1.0))),
        2: Identity(2, "7", (("7", 2.0),)),
        3: Identity(3, "7", (("7", 4.0),)),
    }
    (rider,), unresolved = build_timelines(
        tracklets, ids, CONFIG, bounds=(0, 100), names={"7": "A"}
    )
    assert (rider.number, rider.name) == ("7", "A")
    assert rider.appearances == [Appearance(9.5, 15.5, 0.833), Appearance(29.5, 33.5, 1.0)]
    assert unresolved == []


def test_short_appearances_are_dropped_before_padding() -> None:
    ids = {1: Identity(1, "7", (("7", 3.0),))}
    timelines, _ = build_timelines([_tracklet(1, 10.0, 10.8)], ids, CONFIG, bounds=(0, 100))
    assert timelines == []


def test_padding_is_clamped_and_overlaps_join() -> None:
    tracklets = [_tracklet(1, 0.2, 2.0), _tracklet(2, 4.5, 9.9)]
    ids = {1: Identity(1, "7", (("7", 2.0),)), 2: Identity(2, "7", (("7", 2.0),))}
    config = Config(merge_gap=2.0, padding=1.5, min_appearance=1.0)
    (rider,), _ = build_timelines(tracklets, ids, config, bounds=(0, 10))
    assert rider.appearances == [Appearance(0.0, 10.0, 1.0)]


def test_unread_and_unresolved_tracklets() -> None:
    tracklets = [_tracklet(1, 5.0, 8.0), _tracklet(2, 1.0, 2.0), _tracklet(3, 3.0, 4.0)]
    ids = {
        1: Identity(1, None, (("98", 1.2),)),
        2: Identity(2, None, ()),
    }
    timelines, unresolved = build_timelines(tracklets, ids, CONFIG, bounds=(0, 100))
    assert timelines == []
    assert unresolved == [Unresolved(1, 5.0, 8.0, (("98", 1.2),))]


def test_riders_are_ordered_by_number() -> None:
    tracklets = [_tracklet(1, 0.0, 5.0), _tracklet(2, 0.0, 5.0), _tracklet(3, 0.0, 5.0)]
    ids = {i: Identity(i, n, ((n, 3.0),)) for i, n in ((1, "100"), (2, "9"), (3, "23"))}
    timelines, _ = build_timelines(tracklets, ids, CONFIG, bounds=(0, 100))
    assert [t.number for t in timelines] == ["9", "23", "100"]
    assert timelines[0].appearances[0].confidence == pytest.approx(1.0)
