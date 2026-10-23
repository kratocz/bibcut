"""Blind labelling sheets: per tracklet its two tallest and two sharpest torso crops in one row.

Rows carry only the tracklet id, never a reading, so whoever labels them cannot anchor on
bibcut's output. Writes sheet-NNN.jpg and tracklets.json (the ids on the sheets).

    uv run --with pillow python tools/eval/sheets.py data/explore/crops-<stem> OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, LABEL = 290, 170, 110


def sharpness(path: Path) -> float:
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("crops", type=Path, help="output of crops.py")
    p.add_argument("out", type=Path)
    p.add_argument("--rows", type=int, default=8, help="tracklets per sheet")
    p.add_argument("--min-height", type=int, default=150, help="tallest person in px to include")
    args = p.parse_args()

    by: dict[int, list[dict]] = {}
    for c in json.loads((args.crops / "index.json").read_text()):
        by.setdefault(c["tracklet"], []).append(c)
    picks = []
    for tid in sorted(by):
        cs = by[tid]
        if max(c["height_px"] for c in cs) < args.min_height:
            continue
        tall = sorted(cs, key=lambda c: -c["height_px"])[:2]
        rest = [c for c in cs if c not in tall]
        sharp = sorted(rest, key=lambda c: -sharpness(args.crops / c["file"]))[:2]
        picks.append((tid, sorted(tall + sharp, key=lambda c: c["pts"])))

    args.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=26)  # big ids: a small one was misread (T6 as T8)
    for s in range(0, len(picks), args.rows):
        chunk = picks[s : s + args.rows]
        sheet = Image.new("RGB", (LABEL + 4 * W, len(chunk) * (H + 6)), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        for r, (tid, cs) in enumerate(chunk):
            y = r * (H + 6)
            d.text((6, y + H // 2 - 14), f"T{tid}", fill=(255, 255, 0), font=font)
            for k, c in enumerate(cs):
                im = Image.open(args.crops / c["file"])
                scale = min((W - 6) / im.width, H / im.height)  # up- or downscale to fit
                im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
                sheet.paste(im, (LABEL + k * W, y))
        sheet.save(args.out / f"sheet-{s // args.rows + 1:03d}.jpg", quality=88)
    (args.out / "tracklets.json").write_text(json.dumps([tid for tid, _ in picks]))
    print(len(picks), "tracklets on", (len(picks) + args.rows - 1) // args.rows, "sheets")


if __name__ == "__main__":
    main()
