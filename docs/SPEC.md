# bibcut — specification

> Draft v0.1 (2026-10-22). Living document — update it together with design decisions.

## 1. Background

At an amateur race in Prague, about 20 riders on electric longboards, OneWheels and similar boards raced on a circuit. Each rider wore a jersey with a start number on the front (often hard to see — arms, posture, angle) and on the back. A camera filmed the riders as they passed.

Finding by hand when each rider was on screen is tedious. bibcut automates it.

## 2. Goals

1. **Identify riders by start number** (front or back of the jersey/bib).
2. **Per-rider in-frame intervals** covering the whole appearance — not only the frames where the number is legible.
   *Example:* a rider enters the frame at 0:01:20 with the chest number hidden by an arm; at 0:01:23 they pass the camera and the back number becomes legible; they leave the frame at 0:01:26. The rider's list gets **0:01:20 – 0:01:26**.
3. **Optional start list** mapping number → name (and team, category, …).
4. **Exports:** machine-readable intervals (JSON, CSV), YouTube description text with per-rider timestamps, and ffmpeg commands (or rendered clips) with a per-rider highlight cut.
5. **Manual review** of appearances whose number could not be read, so a human can resolve them in a few clicks.

## 3. Non-goals (for now)

- Real-time / live processing.
- Timing or lap-time measurement — bibcut is not a timing system.
- Face recognition or any other biometric identification.
- Synchronising multiple cameras (each video is processed independently).
- Training detection models from scratch (fine-tuning OCR on a few hundred crops is acceptable if needed).

## 4. Inputs

- **Video** — any format ffmpeg can decode. A fixed (tripod) trackside camera is the primary case; handheld footage is secondary.
- **Start list** (optional) — CSV: `number,name[,extra columns]`.
- **Config** (optional) — track region of interest (polygon), merge gap, padding, minimum appearance length.

## 5. Outputs

`intervals.json` (sketch):

```json
{
  "video": "race.mp4",
  "riders": [
    {
      "number": "7",
      "name": "Rider Name",
      "appearances": [
        { "start": 80.0, "end": 86.0, "confidence": 0.93, "source": "auto" }
      ]
    }
  ],
  "unresolved": [
    { "tracklet_id": 42, "start": 95.2, "end": 97.1, "thumbnail": "review/42.jpg" }
  ]
}
```

- `intervals.csv` — one row per appearance.
- `youtube.txt` — per-rider timestamp lines for the video description. YouTube *chapters* need a single non-overlapping timeline (first at 0:00, at least three, each at least 10 s), which overlapping riders cannot satisfy, so chapters are generated only for laps/heats, if at all.
- `cuts/` — one ffmpeg command (or rendered file) per rider, e.g.:

```sh
ffmpeg -i race.mp4 \
  -vf  "select='between(t,80,86)+between(t,221,229)',setpts=N/FRAME_RATE/TB" \
  -af "aselect='between(t,80,86)+between(t,221,229)',asetpts=N/SR/TB" \
  rider-07.mp4
```

## 6. Pipeline

1. **Decode** frames with presentation timestamps (PTS); support variable-frame-rate footage.
2. **Detect & track** people → *tracklets* (track id, per-frame bbox, timestamps). A pose model provides keypoints in the same pass.
3. **Filter** tracklets outside the track region of interest or too short (spectators, marshals).
4. **Torso crop** from pose keypoints (shoulders → hips), for both front and back views.
5. **Legibility** — start with an OCR-confidence threshold; add a dedicated classifier only if needed.
6. **Number recognition** per frame (scene-text recognition / OCR). If a start list is given, constrain candidates to its numbers.
7. **Tracklet consolidation** — confidence-weighted voting across frames. If readings change mid-tracklet (likely a tracker ID switch in a pack), split the tracklet at that point.
8. **Identity merge** — tracklets with the same number form the rider's timeline; merge gaps shorter than *merge gap*, apply *padding*.
9. **Review** — unresolved tracklets with thumbnails for manual labelling; labels feed back into step 8.
10. **Export** (section 5).

Each stage caches its output so later stages can be re-run without re-processing the video.

## 7. Known hard cases

- Board riders stand sideways: depending on stance (regular/goofy) and direction, a trackside camera sees the chest of some riders and the back of others, so both numbers must be supported.
- Chest number occluded by arms or posture.
- Digit confusions (6/9, 1/7), partially visible numbers, motion blur.
- Riders overlapping in a pack → tracker ID switches.
- A rider leaving and re-entering the frame (new tracklet, same number).
- People without numbers in the frame.

## 8. Prior art

- **Bib recognition in photos / at the finish line:** `gheinrich/bibnumber`, `ericBayless/bib-detector`, Roboflow's marathon bib workflow (line crossing + OCR). None of them produces per-participant timelines.
- **Tracklet-level jersey number recognition:** SoccerNet Jersey Number challenge; Koshkina & Elder, *A General Framework for Jersey Number Recognition in Sports Video* (CVPR Workshops 2024), code in `mkoshkina/jersey-number-pipeline`. Closest to bibcut's core (stages 4–7). Licensed CC BY-NC 3.0: a design reference only, do not copy code.
- **The gap bibcut fills:** end-to-end from raw video to per-rider intervals, YouTube timestamps and cuts, plus manual review.

## 9. Constraints

- **License:** MIT. No GPL/AGPL/non-commercial code or dependencies in the core; AGPL backends (e.g. Ultralytics) only as an optional extra.
- **Privacy:** race footage and start lists are personal data and never go into the repository. Publishing per-rider clips is the publisher's responsibility (riders' consent). No face recognition.
- Runs locally; GPU optional (CPU must work, just slower).

## 10. Milestones

- **M0** — repository, spec, agent guidance.
- **M1** — CLI: on one real video, generates intervals for most riders and ffmpeg makes cuts from them. → **switch the repository to public.**
- **M2** — start list mapping, YouTube export, manual review flow.
- **M3** — robustness: handheld footage, multiple videos per event, packaging.

## 11. Open questions

- Camera setup at target races: tripod vs handheld, one or several cameras, resolution and fps?
- Number format: digits only, range (1–99?), font and jersey design.
- OCR backend: PARSeq vs PaddleOCR vs a small local VLM.
- Detector/tracker backend and its license.
- Review UI: CLI/TUI vs a small local web page.
- Sample footage for development — source and riders' consent.
