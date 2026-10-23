from bibcut.models import Detection
from bibcut.track import Tracker, TrackerConfig, track


def _box(cx: float, cy: float, w: float = 0.04, h: float = 0.12) -> tuple[float, ...]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _det(pts: float, cx: float, cy: float, score: float = 0.9) -> Detection:
    return Detection(pts=pts, bbox=_box(cx, cy), score=score)  # type: ignore[arg-type]


def _frames(n: int, step: float, objects, start: float = 0.0):
    """objects: (x0, y, vx, score) moving linearly; returns [(pts, detections)]."""
    out = []
    for i in range(n):
        t = start + i * step
        out.append((t, [_det(t, x0 + vx * t, y, s) for x0, y, vx, s in objects]))
    return out


def test_two_riders_moving_apart_give_two_tracklets() -> None:
    frames = _frames(20, 0.2, [(0.3, 0.3, 0.05, 0.9), (0.7, 0.7, -0.05, 0.9)])
    tracklets = track(frames)
    assert [t.id for t in tracklets] == [1, 2]
    assert [len(t.detections) for t in tracklets] == [20, 20]
    assert tracklets[0].detections[0].bbox == frames[0][1][0].bbox
    assert tracklets[0].start == 0.0 and tracklets[0].end == frames[-1][0]


def test_short_gap_keeps_identity() -> None:
    frames = _frames(20, 0.2, [(0.3, 0.5, 0.05, 0.9)])
    del frames[8:11]  # 0.8 s without a detection
    (t,) = track(frames, TrackerConfig(max_age=1.0))
    assert len(t.detections) == 17


def test_long_gap_starts_a_new_tracklet() -> None:
    frames = _frames(20, 0.2, [(0.3, 0.5, 0.0, 0.9)])
    del frames[8:16]  # 1.8 s
    assert [len(t.detections) for t in track(frames, TrackerConfig(max_age=1.0))] == [8, 4]


def test_weak_detections_continue_but_never_start_a_track() -> None:
    frames = _frames(10, 0.2, [(0.3, 0.5, 0.05, 0.9)])
    weak = _frames(10, 0.2, [(0.3, 0.5, 0.05, 0.2)], start=2.0)
    ghost = _frames(10, 0.2, [(0.8, 0.2, 0.0, 0.2)])
    merged = [(t, a + b) for (t, a), (_, b) in zip(frames + weak, ghost + ghost, strict=True)]
    (t,) = track(merged)
    assert len(t.detections) == 20


def test_single_frame_blip_is_not_a_tracklet() -> None:
    frames = _frames(10, 0.2, [(0.3, 0.5, 0.0, 0.9)])
    frames[5][1].append(_det(1.0, 0.8, 0.2))
    assert len(track(frames)) == 1


def test_irregular_time_steps_use_real_time() -> None:
    # Constant speed sampled at uneven intervals: the prediction must scale with dt, or the
    # long steps land outside the predicted box.
    times = [0.0, 0.1, 0.2, 0.3, 0.9, 1.0, 1.1, 1.7, 1.8]
    frames = [(t, [_det(t, 0.1 + 0.1 * t, 0.5)]) for t in times]
    (t,) = track(frames, TrackerConfig(max_age=1.0))
    assert len(t.detections) == len(times)


def test_fast_rider_moving_more_than_its_width_per_sample() -> None:
    # Measured on the sample footage at 5 fps: consecutive boxes of a racing rider rarely overlap
    # (IoU 0 in the median case) while the centres are ~0.7 box heights apart.
    frames = _frames(15, 0.2, [(0.1, 0.5, 0.3, 0.9)])  # 0.06 per step, box 0.04 wide
    (t,) = track(frames)
    assert len(t.detections) == 15


def test_distance_respects_frame_aspect() -> None:
    # 0.12 of the height is one box height; 0.12 of the (16:9) width is 1.8 box heights.
    vertical = [(t, [_det(t, 0.5, 0.2 + 0.6 * t)]) for t in (0.0, 0.2, 0.4)]
    horizontal = [(t, [_det(t, 0.2 + 0.6 * t, 0.5)]) for t in (0.0, 0.2, 0.4)]
    cfg = TrackerConfig(max_distance=1.5)
    assert len(track(vertical, cfg)) == 1
    assert track(horizontal, cfg) == []


def test_size_mismatch_is_not_matched() -> None:
    frames = [
        (0.0, [Detection(pts=0.0, bbox=_box(0.5, 0.5), score=0.9)]),  # type: ignore[arg-type]
        (0.2, [Detection(pts=0.2, bbox=_box(0.5, 0.5, 0.15, 0.45), score=0.9)]),  # type: ignore[arg-type]
    ]
    assert track(frames) == []


def test_deterministic() -> None:
    objects = [(0.3, 0.3, 0.02, 0.9), (0.32, 0.34, 0.015, 0.7), (0.6, 0.6, -0.03, 0.8)]
    frames = _frames(30, 0.2, objects)
    first = [(t.id, [d.pts for d in t.detections]) for t in track(frames)]
    assert first == [(t.id, [d.pts for d in t.detections]) for t in track(frames)]


def test_incremental_api_matches_batch() -> None:
    frames = _frames(10, 0.2, [(0.3, 0.5, 0.05, 0.9)])
    tracker = Tracker()
    for pts, dets in frames:
        tracker.update(pts, dets)
    assert [len(t.detections) for t in tracker.finish()] == [10]
