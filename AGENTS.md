# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, …) working in this repository.

## Project

bibcut identifies race participants in video by the number on their jersey/bib and outputs per-rider in-frame intervals (JSON/CSV, YouTube timestamps, ffmpeg cuts).

**Source of truth:** [docs/SPEC.md](docs/SPEC.md). Read it before making design decisions. If an implementation choice contradicts the spec, update the spec in the same change or ask first.

## Status

Scaffolded; working towards milestone **M1** (see SPEC §10). Do not build features beyond the current milestone without asking.

## Stack

- Python 3.12+, managed with `uv` (build backend `uv_build`, `src/` layout); `ruff` for lint/format; `pytest` for tests
- Video decoding, metadata and cutting all via the `ffmpeg`/`ffprobe` CLI as subprocesses. Do not switch decoding to PyAV: its binary wheels bundle `libx264`/`libx265` (GPL), which the licensing rule below keeps out of the core. Hardware decoding: the CLI uses `videotoolbox` by default on macOS (bit-identical to software decoding on the sample footage, SPEC §12); `--hwaccel none` turns it off
- Detection/tracking (with pose) and number recognition each behind a protocol in `src/bibcut/interfaces.py`, so backends can be swapped

## Commands

```sh
uv sync                      # create .venv and install
uv run pytest                # tests (needs ffmpeg on PATH; video tests skip without it)
uv run ruff check            # lint
uv run ruff format           # format
uv run bibcut probe VIDEO    # print video metadata as JSON
uv run bibcut track VIDEO --start 200 --end 320   # detect + track a segment; cache in data/work/<stem>
uv run bibcut identify VIDEO --start 200 --end 320 --startlist START.csv   # + read numbers, vote per tracklet
uv run bibcut intervals VIDEO --start 200 --end 320 --startlist START.csv  # + per-rider intervals and cut scripts in data/out/<stem>
```

Model weights download on first use to `~/.cache/bibcut/models` (override with `BIBCUT_MODELS_DIR`).

## Layout

- `src/bibcut/models.py` — data passed between stages (frames, detections, tracklets, readings, appearances)
- `src/bibcut/interfaces.py` — backend protocols
- `src/bibcut/cache.py` — per-stage output cache keyed by input fingerprint + stage parameters
- `src/bibcut/config.py` — optional TOML run config (SPEC §4)
- `src/bibcut/decode.py` — streaming RGB frames with PTS from `ffmpeg` (sampling, resize, time range, hwaccel) and GoPro proxy lookup
- `src/bibcut/weights.py` — pinned model weights (URL, SHA-256, license, provenance) and their download
- `src/bibcut/detect.py` — RF-DETR person detector on onnxruntime (normalized boxes)
- `src/bibcut/track.py` — own ByteTrack-style tracker on real time (PTS), numpy only
- `src/bibcut/crop.py` — torso band of a person box, pixel crops of normalized boxes
- `src/bibcut/ocr.py` — PP-OCRv5 start-number recognizer: horizontal bands, digit-only CTC, numpy resize matching Pillow's bilinear
- `src/bibcut/identify.py` — start list loading and confidence-weighted voting per tracklet
- `src/bibcut/pipeline.py` — cached stages: `detect_stage` → `detect/detections.json`, `track_stage` → `track/tracklets.json`, `read_stage` → `read/readings.json` (full-resolution crops of tracked detections)
- `src/bibcut/timeline.py` — per-rider appearances from identified tracklets (merge gap, minimum appearance, padding) and the unresolved list
- `src/bibcut/export.py` — `intervals.json`, `intervals.csv` and one ffmpeg cut script per rider
- `src/bibcut/probe.py`, `src/bibcut/cli.py` — ffprobe wrapper and CLI

## Sample footage

The development footage (SPEC §12) is read in place from an external disk and never copied into the repository. The local disk is tight: keep the whole working tree, including `data/` and `.venv`, under ~1 GiB, and check `du -sh` before extracting frames or caching stage output in bulk. Prefer the `.LRV` proxies for anything that does not need 4K.

## Hard rules

- **Never commit data:** videos, extracted frames, rider crops, start lists with real names, or model weights. They are personal data and/or large. Everything lives under `data/` (gitignored). Tests use tiny synthetic fixtures in `tests/fixtures/`.
- **Licensing:** the project is MIT. Do not add dependencies or copied code under GPL/AGPL or non-commercial licenses to the core. `mkoshkina/jersey-number-pipeline` (CC BY-NC 3.0) is a design reference only — do not copy its code. An AGPL backend (e.g. Ultralytics) may only be added as an optional, separately installed extra, and only after asking. A dependency's license includes the native libraries its wheels bundle: PyAV, OpenCV and packages requiring them are out (SPEC §9). Before adding any dependency with native code, list the libraries in its macOS and Linux wheels.
- **Model weights** are never committed or redistributed. They are downloaded from a pinned URL, verified against a pinned SHA-256, and need a stated permissive license.
- **No face recognition.** Riders are identified by their number only.
- **Time comes from frame timestamps (PTS)**, not frame index × nominal fps — phone footage is often variable frame rate.
- Outputs must be deterministic for the same input and config.

## Conventions

- Code, comments, commit messages and docs in English.
- Small, focused commits with descriptive messages.
- Every pipeline stage should be runnable and testable on its own and cache its output, so later stages can be re-run without re-processing the video.
