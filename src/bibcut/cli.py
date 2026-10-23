"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bibcut import __version__
from bibcut.probe import probe


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.video)
    print(json.dumps(dataclasses.asdict(info), indent=2))
    return 0


def _hwaccel(args: argparse.Namespace) -> str | None:
    # VideoToolbox decodes on Apple's media engine; its frames are bit-identical to software
    # decoding on the sample footage (SPEC §12), so it is safe as the macOS default.
    if args.hwaccel is None:
        return "videotoolbox" if sys.platform == "darwin" else None
    return None if args.hwaccel == "none" else args.hwaccel


def _workdir(args: argparse.Namespace) -> Path:
    return args.workdir or Path("data") / "work" / args.video.stem


def _tracked(args: argparse.Namespace) -> tuple[Path, str]:
    """Run (or reuse) detection and tracking; returns (tracklets.json, track stage key)."""
    from bibcut import weights
    from bibcut.detect import RFDETR
    from bibcut.pipeline import detect_stage, stderr_progress, track_stage
    from bibcut.track import TrackerConfig

    workdir = _workdir(args)
    model = weights.ensure(weights.RFDETR_S)
    detector = RFDETR(model, threshold=args.min_score)
    _, key = detect_stage(
        args.video,
        workdir,
        detector,
        {"weights": weights.RFDETR_S.sha256, "threshold": args.min_score},
        sample_fps=args.sample_fps,
        start=args.start,
        end=args.end,
        use_proxy=not args.full_res,
        hwaccel=_hwaccel(args),
        progress=stderr_progress,
    )
    return track_stage(workdir, key, TrackerConfig())


def _cmd_track(args: argparse.Namespace) -> int:
    from bibcut.pipeline import load_tracklets

    tracklets = load_tracklets(_tracked(args)[0])
    summary = {
        "workdir": str(_workdir(args)),
        "tracklets": [
            {"id": t.id, "start": t.start, "end": t.end, "detections": len(t.detections)}
            for t in tracklets
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_identify(args: argparse.Namespace) -> int:
    from bibcut import weights
    from bibcut.identify import load_startlist, vote
    from bibcut.ocr import PPOCR
    from bibcut.pipeline import load_readings, load_tracklets, read_stage, stderr_progress

    startlist = load_startlist(args.startlist) if args.startlist else None
    tracklets_path, track_key = _tracked(args)
    model = weights.ensure(weights.PPOCRV5_EN_REC)
    readings_path, _ = read_stage(
        args.video,
        _workdir(args),
        track_key,
        PPOCR(model),
        {"weights": weights.PPOCRV5_EN_REC.sha256},
        sample_fps=args.sample_fps,
        start=args.start,
        end=args.end,
        min_height_px=args.min_height,
        hwaccel=_hwaccel(args),
        progress=stderr_progress,
    )
    crops = load_readings(readings_path)
    valid = set(startlist) if startlist is not None else None
    rows = []
    for t in load_tracklets(tracklets_path):
        result = vote(t.id, crops.get(t.id, []), valid=valid, min_votes=args.min_votes)
        rows.append({
            "id": t.id,
            "start": t.start,
            "end": t.end,
            "number": result.number,
            "name": (startlist or {}).get(result.number or "") or None,
            "crops": len(crops.get(t.id, [])),
            "votes": [[n, round(v, 3)] for n, v in result.votes[:3]],
        })  # fmt: skip
    print(json.dumps({"workdir": str(_workdir(args)), "tracklets": rows}, indent=2))
    return 0


def _add_tracking_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("video", type=Path)
    p.add_argument("--workdir", type=Path, help="stage cache (default: data/work/<video stem>)")
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--start", type=float, help="seconds on the video timeline")
    p.add_argument("--end", type=float, help="seconds on the video timeline")
    p.add_argument("--min-score", type=float, default=0.1, help="detector score floor")
    p.add_argument("--full-res", action="store_true", help="ignore a GoPro LRV proxy")
    p.add_argument(
        "--hwaccel",
        help='ffmpeg hardware decoder (default: "videotoolbox" on macOS), "none" for software',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bibcut", description="Per-rider timelines from race footage, keyed by start number."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="print video metadata as JSON")
    p.add_argument("video", type=Path)
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("track", help="detect and track people; print tracklets as JSON")
    _add_tracking_args(p)
    p.set_defaults(func=_cmd_track)

    p = sub.add_parser(
        "identify", help="track people and read their start numbers; print tracklets as JSON"
    )
    _add_tracking_args(p)
    p.add_argument("--startlist", type=Path, help="CSV with number,name; restricts readings")
    p.add_argument("--min-height", type=int, default=100, help="person height in px to read")
    p.add_argument(
        "--min-votes", type=float, default=2.0, help="summed confidence to credit a number"
    )
    p.set_defaults(func=_cmd_identify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
