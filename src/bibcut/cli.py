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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bibcut", description="Per-rider timelines from race footage, keyed by start number."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="print video metadata as JSON")
    p.add_argument("video", type=Path)
    p.set_defaults(func=_cmd_probe)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
