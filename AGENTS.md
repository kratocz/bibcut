# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, …) working in this repository.

## Project

bibcut identifies race participants in video by the number on their jersey/bib and outputs per-rider in-frame intervals (JSON/CSV, YouTube timestamps, ffmpeg cuts).

**Source of truth:** [docs/SPEC.md](docs/SPEC.md). Read it before making design decisions. If an implementation choice contradicts the spec, update the spec in the same change or ask first.

## Status

Pre-implementation. The current goal is milestone **M1** (see SPEC §10). Do not build features beyond the current milestone without asking.

## Stack (proposed — confirm before scaffolding)

- Python 3.12+, managed with `uv`; `ruff` for lint/format; `pytest` for tests
- Video decoding via PyAV or OpenCV; cutting via the `ffmpeg` CLI
- Detection/tracking, pose and number recognition each behind a small interface so backends can be swapped

Fill in the real commands here once the project is scaffolded.

## Hard rules

- **Never commit data:** videos, extracted frames, rider crops, start lists with real names, or model weights. They are personal data and/or large. Everything lives under `data/` (gitignored). Tests use tiny synthetic fixtures in `tests/fixtures/`.
- **Licensing:** the project is MIT. Do not add dependencies or copied code under GPL/AGPL or non-commercial licenses to the core. `mkoshkina/jersey-number-pipeline` (CC BY-NC 3.0) is a design reference only — do not copy its code. An AGPL backend (e.g. Ultralytics) may only be added as an optional, separately installed extra, and only after asking.
- **No face recognition.** Riders are identified by their number only.
- **Time comes from frame timestamps (PTS)**, not frame index × nominal fps — phone footage is often variable frame rate.
- Outputs must be deterministic for the same input and config.

## Conventions

- Code, comments, commit messages and docs in English.
- Small, focused commits with descriptive messages.
- Every pipeline stage should be runnable and testable on its own and cache its output, so later stages can be re-run without re-processing the video.
