"""Score `bibcut identify` output against blind labels of the sheets from sheets.py.

    python3 tools/eval/score.py IDENTIFY_JSON SHEETS_DIR/labels-*.json \
        [--sheets SHEETS_DIR] [--exclude FROM TO]

Labels are JSON objects {"T<id>": {"number": "218" | "?" | "-", "note": "..."}}; several files
(one per labeller) are merged. With --sheets, the merged ids must match the sheets exactly.
--exclude leaves out tracklets overlapping a time window on both sides (held-out scoring of a
parameter tuned on that window).

Rider recall: labelled numbers bibcut credited to at least one tracklet.
Tracklet errors: credited tracklets whose label is another number or "-" (no bib).
Unverifiable: credited tracklets labelled "?" or not on the sheets.
A wrong rider: a credited number none of whose tracklets is confirmed and one is contradicted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("identify", type=Path, help="JSON printed by `bibcut identify`")
    p.add_argument("labels", type=Path, nargs="+")
    p.add_argument("--sheets", type=Path, help="sheets dir, to check the ids")
    p.add_argument("--exclude", type=float, nargs=2, metavar=("FROM", "TO"))
    args = p.parse_args()

    labels: dict[int, str] = {}
    for path in args.labels:
        for key, value in json.loads(path.read_text()).items():
            tid = int(key[1:])
            if tid in labels:
                sys.exit(f"{path}: {key} labelled twice")
            labels[tid] = value["number"]
    if args.sheets:
        ids = set(json.loads((args.sheets / "tracklets.json").read_text()))
        if ids != set(labels):
            sys.exit(f"ids differ: missing {sorted(ids - set(labels))}, "
                     f"extra {sorted(set(labels) - ids)}")  # fmt: skip

    rows = json.loads(args.identify.read_text())["tracklets"]
    if args.exclude:
        a, b = args.exclude
        rows = [r for r in rows if r["end"] < a or r["start"] > b]
    system = {r["id"]: r["number"] for r in rows}
    labels = {k: v for k, v in labels.items() if k in system}

    gt_riders = {n for n in labels.values() if n not in ("?", "-")}
    credited = {tid: n for tid, n in system.items() if n}
    sys_riders = set(credited.values())
    found = gt_riders & sys_riders
    right, wrong, unverifiable = [], [], []
    status: dict[str, set[str]] = defaultdict(set)
    for tid, n in sorted(credited.items()):
        label = labels.get(tid)
        if label == n:
            right.append(tid)
        elif label in (None, "?"):
            unverifiable.append(tid)
        else:
            wrong.append((tid, n, label))
        status[n].add("ok" if label == n else "?" if label in (None, "?") else "bad")
    wrong_riders = sorted((n for n, s in status.items() if "ok" not in s and "bad" in s), key=int)

    count = defaultdict(int)
    for v in labels.values():
        count["number" if v not in ("?", "-") else v] += 1
    print(f"labelled tracklets: {len(labels)} ({count['number']} with a number, "
          f"{count['?']} '?', {count['-']} '-')")  # fmt: skip
    print(
        f"rider recall: {len(found)}/{len(gt_riders)} = {len(found) / max(1, len(gt_riders)):.0%}"
    )
    print(f"  missed: {sorted(gt_riders - sys_riders, key=int)}")
    print(f"credited tracklets: {len(credited)}; right {len(right)}, wrong {len(wrong)}, "
          f"unverifiable {len(unverifiable)}")  # fmt: skip
    print(f"  wrong (tracklet, credited, label): {wrong}")
    print(f"wrong riders: {len(wrong_riders)}/{len(sys_riders)} = "
          f"{len(wrong_riders) / max(1, len(sys_riders)):.0%} {wrong_riders}")  # fmt: skip


if __name__ == "__main__":
    main()
