"""Torso crops of every tracked person tall enough to carry a legible number.

Reads the tracklets of a `bibcut track`/`identify`/`intervals` run and cuts the torso of each
detection from the full-resolution video, as JPEGs plus an `index.json`. The crops are what
`sheets.py` puts in front of the labellers.

    uv run --with pillow python tools/eval/crops.py VIDEO --workdir data/work/<stem> \
        --out data/explore/crops-<stem>

Use the same --sample-fps, --start and --end as the run, or tracked frames are not found.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from bibcut.crop import crop, torso_box
from bibcut.decode import iter_frames
from bibcut.pipeline import load_tracklets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("video", type=Path)
    p.add_argument("--workdir", type=Path, required=True, help="the run's stage cache")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-height", type=int, default=150, help="person height in px")
    p.add_argument("--sample-fps", type=float, default=5.0)
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--hwaccel", default="videotoolbox", help='"none" for software decoding')
    args = p.parse_args()

    wanted: dict[float, list] = {}
    for t in load_tracklets(args.workdir / "track" / "tracklets.json"):
        for d in t.detections:
            wanted.setdefault(round(d.pts, 6), []).append((t.id, d))
    args.out.mkdir(parents=True, exist_ok=True)
    index, began = [], time.monotonic()
    hwaccel = None if args.hwaccel == "none" else args.hwaccel
    for frame in iter_frames(
        args.video, sample_fps=args.sample_fps, start=args.start, end=args.end, hwaccel=hwaccel
    ):
        for tid, d in wanted.get(round(frame.pts, 6), []):
            height_px = round((d.bbox[3] - d.bbox[1]) * frame.image.shape[0])
            if height_px < args.min_height:
                continue
            name = f"{tid:04d}_{frame.pts:09.3f}.jpg"
            Image.fromarray(crop(frame.image, torso_box(d.bbox))).save(args.out / name, quality=92)
            index.append({"file": name, "tracklet": tid, "pts": frame.pts, "bbox": d.bbox,
                          "score": d.score, "height_px": height_px})  # fmt: skip
    (args.out / "index.json").write_text(json.dumps(index))
    print(f"{len(index)} crops in {time.monotonic() - began:.0f}s")


if __name__ == "__main__":
    main()
