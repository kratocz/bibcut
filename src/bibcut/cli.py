"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path

from bibcut import __version__
from bibcut.probe import probe


def _cmd_probe(args: argparse.Namespace) -> int:
    info = probe(args.video)
    print(json.dumps(dataclasses.asdict(info), indent=2))
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    from bibcut import weights
    from bibcut.detect import RFDETR
    from bibcut.pipeline import detect_stage, load_tracklets, stderr_progress, track_stage
    from bibcut.track import TrackerConfig

    workdir = args.workdir or Path("data") / "work" / args.video.stem
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
        hwaccel=args.hwaccel,
        progress=stderr_progress,
    )
    tracklets = load_tracklets(track_stage(workdir, key, TrackerConfig()))
    summary = {
        "workdir": str(workdir),
        "tracklets": [
            {"id": t.id, "start": t.start, "end": t.end, "detections": len(t.detections)}
            for t in tracklets
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


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
    p.add_argument("video", type=Path)
    p.add_argument("--workdir", type=Path, help="stage cache (default: data/work/<video stem>)")
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--start", type=float, help="seconds on the video timeline")
    p.add_argument("--end", type=float, help="seconds on the video timeline")
    p.add_argument("--min-score", type=float, default=0.1, help="detector score floor")
    p.add_argument("--full-res", action="store_true", help="ignore a GoPro LRV proxy")
    p.add_argument("--hwaccel", help='ffmpeg hardware decoder, e.g. "videotoolbox"')
    p.set_defaults(func=_cmd_track)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
