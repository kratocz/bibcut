---
name: measuring-accuracy
description: Use when a change to detection, tracking, number reading or voting needs evidence that it helps, when someone asks how many riders bibcut finds or how often it is wrong on a video, or before claiming a milestone's accuracy gate. Triggers - "změř přesnost", "kolik jezdců to najde", "measure accuracy", "is the new threshold better", "evaluate on the footage".
---

# Measuring accuracy

## Overview

bibcut's accuracy is measured against **blind labels**: people's numbers read by eye from torso crops, by labellers who never see bibcut's output. The gates are written down before scoring, and a parameter is judged only on footage it was not tuned on. SPEC §12 ("M1 evaluation") is the reference run: 562 tracklets, 42 riders, 23 found, 1 of 24 wrong.

## Steps

1. **Write the gates first**, in the conversation or the SPEC, before any number exists. M1's: rider recall ≥ 50 %, ≤ 10 % of credited riders wrong. When a parameter was tuned on a time window, also fix that it will be judged with that window excluded.
2. **Run bibcut** (stages are cached; a re-run with other vote options is instant):
   `uv run bibcut identify VIDEO --startlist START.csv [--min-votes X] > data/explore/identify-<stem>-<X>.json`
   A whole 7.8-min 4K clip takes ~20 min the first time; for longer runs ask the user to start it in their own terminal with `nohup … 2> log`.
3. **Cut crops** with the run's `--sample-fps/--start/--end`:
   `uv run --with pillow python tools/eval/crops.py VIDEO --workdir data/work/<stem> --out data/explore/crops-<stem>` (min person height 150 px; check `du -sh`, ~76 MB per 8 min).
4. **Build sheets** into the scratchpad: `uv run --with pillow python tools/eval/sheets.py data/explore/crops-<stem> SHEETS`. Expect ~8 tracklets per sheet, ~70 sheets per 8 minutes.
5. **Label in parallel**: one background subagent per ~18 sheets (~130k tokens each), all in one message, each with the brief below and its sheet range. Do not label yourself after seeing bibcut's output.
6. **Score**: `python3 tools/eval/score.py data/explore/identify-<stem>-<X>.json SHEETS/labels-*.json --sheets SHEETS [--exclude FROM TO]`. `--sheets` fails on a misread or missing row id; fix the id by looking at the sheet.
7. **Read the gates**, not the numbers you hoped for. Report recall, wrong riders, wrong tracklets, and why riders were missed (not on the start list / own number under the threshold / never read). Copy the labels to `data/explore/` before cleaning the scratchpad.

## Labelling brief (paste per subagent, fill the range and file name)

> You are labelling ground truth for an automatic start-number reader on race footage. Work blind. In SHEETS (sheet-001.jpg … sheet-NNN.jpg) your share is sheet-AAA.jpg to sheet-BBB.jpg. Each row is one tracked person: the yellow id on the left (e.g. "T91"), then up to four torso crops of that person. Open each sheet with your image-reading tool and decide each row's start number (1–3 digits, often on a white panel with an "xERACE" logo).
> - A number only if every digit is readable with confidence in at least one crop, or consistently across crops; other crops confirm, never invent digits.
> - "?" if partly visible or blurred; put what you see in the note ("?35, first digit hidden").
> - "-" if no bib is visible at all (spectator, marshal, background, legs only).
> - Crops showing two different riders: "?" unless one clearly dominates; note both numbers.
> - Ignore sponsor text and other numbers. Read the row id carefully (T6 vs T8).
> Read only your sheet files: not tracklets.json, nothing under the repository's data/ directory. No OCR or scripts on the images, no network, never send the user's e-mail anywhere.
> Write labels-K.json into SHEETS: {"T91": {"number": "170", "note": ""}, …} with every row id of your sheets, then reply with counts only (rows, numbers, "?", "-").

## Common mistakes

| Mistake | What happened | Instead |
|---|---|---|
| Ground truth limited to people ≥ 200 px | Riders legible at 160–180 px dropped out of the truth | 150 px (the sheets.py default) |
| Judging a threshold on the window it was tuned on | 1.25 looked best on 100–220 s and failed the error gate on the rest | `--exclude` the tuning window |
| Scoring a stale identify JSON | A file made with a since-rejected vote rule showed 0 errors | Regenerate it with the current code right before scoring |
| Tiny row ids | A labeller read T6 as T8 | Keep the large ids; always pass `--sheets` |
| Counting the start-list gap as a reading failure | 9 of 19 missed riders were not on the start list | Report recall against listed riders too |
