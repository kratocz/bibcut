"""Multi-person tracking, ByteTrack-style (SPEC §6, step 2).

Per frame: predict every track with a constant-velocity Kalman filter, match confident
detections to tracks, then give unmatched confirmed tracks a second chance with weak detections
(occlusion, blur). Only confident detections start tracks, and a track counts once it has been
matched twice in a row.

Two departures from the original:

- Time is real time (PTS), not frame counts: prediction scales with the time step and tracks
  expire after `max_age` seconds, so sampling rate and variable frame rate do not change behaviour.
- Matching uses the distance between predicted and detected box centres, in box heights, instead
  of IoU. Detection is sampled at a few fps, and a racing rider then moves further than their own
  width between samples (SPEC §12), so consecutive boxes rarely overlap. Pairs whose heights differ
  too much are never matched. Matching is greedy by distance: deterministic and enough at race
  density.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from bibcut.models import Detection, Tracklet

# ByteTrack's noise weights are per frame at 30 fps; noise variance grows with elapsed frames.
_REF_FPS = 30.0
_W_POS = 1 / 20
_W_VEL = 1 / 160


@dataclass(frozen=True)
class TrackerConfig:
    high: float = 0.4  # detections at or above take part in the first matching pass
    low: float = 0.1  # weaker ones are ignored entirely
    new_track: float = 0.5  # minimum score to start a track
    max_distance: float = 1.5  # first pass: centre distance in box heights
    weak_max_distance: float = 0.75  # second pass, weak detections
    max_size_ratio: float = 1.6  # matched box heights differ at most by this factor
    min_hits: int = 2
    max_age: float = 1.0  # seconds without a match before a track ends
    frame_aspect: float = 16 / 9  # width / height, so x and y distances share one unit


def track(
    frames: Iterable[tuple[float, list[Detection]]], config: TrackerConfig | None = None
) -> list[Tracklet]:
    tracker = Tracker(config)
    for pts, detections in frames:
        tracker.update(pts, detections)
    return tracker.finish()


class Tracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._active: list[_Track] = []
        self._ended: list[_Track] = []
        self._next_id = 1

    def update(self, pts: float, detections: list[Detection]) -> None:
        cfg = self.config
        for t in self._active:
            t.kf.predict(pts)
        expired = [t for t in self._active if pts - t.last_seen > cfg.max_age]
        self._ended += [t for t in expired if t.id is not None]
        self._active = [t for t in self._active if t not in expired]

        strong = [d for d in detections if d.score >= cfg.high]
        weak = [d for d in detections if cfg.low <= d.score < cfg.high]

        matched, free_tracks, free_strong = _greedy(self._active, strong, cfg.max_distance, cfg)
        confirmed_free = [t for t in free_tracks if t.id is not None]
        matched_weak, _, _ = _greedy(confirmed_free, weak, cfg.weak_max_distance, cfg)

        for t, d in matched + matched_weak:
            t.kf.update(_xyah(d.bbox))
            t.detections.append(d)
            t.last_seen = pts
            t.hits += 1
        # Ids follow track creation order, not match order, so they do not hinge on float ties.
        for t in self._active:
            if t.id is None and t.hits >= cfg.min_hits:
                t.id = self._next_id
                self._next_id += 1

        hit = {id(t) for t, _ in matched + matched_weak}
        # A tentative track must be matched on every frame until confirmed.
        self._active = [t for t in self._active if t.id is not None or id(t) in hit]
        self._active += [
            _Track(_Kalman(_xyah(d.bbox), pts), [d], pts)
            for d in free_strong
            if d.score >= cfg.new_track
        ]

    def finish(self) -> list[Tracklet]:
        done = [t for t in self._ended + self._active if t.id is not None]
        return [Tracklet(id=t.id, detections=t.detections) for t in sorted(done, key=_by_id)]


@dataclass(eq=False)
class _Track:
    kf: _Kalman
    detections: list[Detection]
    last_seen: float
    hits: int = 1
    id: int | None = None  # assigned on confirmation


def _by_id(t: _Track) -> int:
    assert t.id is not None
    return t.id


def _greedy(tracks: list[_Track], dets: list[Detection], max_dist: float, cfg: TrackerConfig):
    """Match closest pairs first; returns (pairs, unmatched tracks, unmatched detections)."""
    if not tracks or not dets:
        return [], list(tracks), list(dets)
    pred = np.array([t.kf.box() for t in tracks])
    boxes = np.array([d.bbox for d in dets])
    dist, ratio = _distance_matrix(pred, boxes, cfg.frame_aspect)
    ti, di = np.nonzero((dist <= max_dist) & (ratio <= cfg.max_size_ratio))
    order = np.lexsort((di, ti, dist[ti, di]))
    used_t: set[int] = set()
    used_d: set[int] = set()
    pairs = []
    for k in order:
        a, b = int(ti[k]), int(di[k])
        if a in used_t or b in used_d:
            continue
        used_t.add(a)
        used_d.add(b)
        pairs.append((tracks[a], dets[b]))
    free_t = [t for i, t in enumerate(tracks) if i not in used_t]
    free_d = [d for i, d in enumerate(dets) if i not in used_d]
    return pairs, free_t, free_d


def _distance_matrix(
    pred: npt.NDArray[np.float64], boxes: npt.NDArray[np.float64], aspect: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Centre distance in predicted box heights, and the height ratio (>= 1), track x detection."""
    pc = (pred[:, :2] + pred[:, 2:]) / 2
    bc = (boxes[:, :2] + boxes[:, 2:]) / 2
    ph = np.maximum(pred[:, 3] - pred[:, 1], 1e-6)
    bh = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
    dx = (pc[:, None, 0] - bc[None, :, 0]) * aspect
    dy = pc[:, None, 1] - bc[None, :, 1]
    dist = np.hypot(dx, dy) / ph[:, None]
    ratio = np.maximum(ph[:, None] / bh[None, :], bh[None, :] / ph[:, None])
    return dist, ratio


def _xyah(bbox) -> npt.NDArray[np.float64]:
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1e-6)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / h, h])


class _Kalman:
    """Constant-velocity filter on (cx, cy, aspect, height); velocities are per second."""

    def __init__(self, z: npt.NDArray[np.float64], t: float) -> None:
        h = z[3]
        self.x = np.concatenate([z, np.zeros(4)])
        vel = _W_VEL * _REF_FPS
        std = [2 * _W_POS * h, 2 * _W_POS * h, 1e-2, 2 * _W_POS * h,
               10 * vel * h, 10 * vel * h, 1e-5, 10 * vel * h]  # fmt: skip
        self.P = np.diag(np.square(std))
        self.t = t

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        F = np.eye(8)
        F[:4, 4:] = dt * np.eye(4)
        h = self.x[3]
        vel = _W_VEL * _REF_FPS
        std = [_W_POS * h, _W_POS * h, 1e-2, _W_POS * h, vel * h, vel * h, 1e-5, vel * h]
        Q = np.diag(np.square(std)) * (dt * _REF_FPS)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z: npt.NDArray[np.float64]) -> None:
        h = self.x[3]
        R = np.diag(np.square([_W_POS * h, _W_POS * h, 1e-1, _W_POS * h]))
        H = np.eye(4, 8)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S, H @ self.P).T
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    def box(self) -> tuple[float, float, float, float]:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
