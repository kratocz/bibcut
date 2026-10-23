# bibcut — specification

> Draft v0.2 (2026-10-23). Living document — update it together with design decisions.

## 1. Background

At an amateur race in Prague (xErace, 2025-05-17, an indoor karting hall), 74 riders raced on a circuit in several categories: OneWheel, electric unicycles (EUC, incl. juniors), Open e-skate (incl. 2WD) and Street; some riders entered more than one category. Each rider wore a jersey with a start number on the front (often hard to see — arms, posture, angle) and on the back. A pair of cameras filmed the race, mostly from a gallery overlooking almost the whole track (see section 12).

Finding by hand when each rider was on screen is tedious. bibcut automates it.

## 2. Goals

1. **Identify riders by start number** (front or back of the jersey/bib).
2. **Per-rider intervals** covering the whole appearance — not only the frames where the number is legible. Two interval modes:
   - **Participation** (M1): when the rider was on the track. With a camera that sees the whole circuit this is roughly the rider's heat, so the timeline answers "in which heats did #250 ride, and when".
   - **Near camera** (M2, switchable): when the rider was close to the camera and large in frame — the moments worth cutting into a per-rider highlight.

   *Example (near camera):* a rider enters the near part of the frame at 0:01:20 with the chest number hidden by an arm; at 0:01:23 the back number becomes legible; they leave at 0:01:26. The rider's list gets **0:01:20 – 0:01:26**.
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

1. **Decode** frames with presentation timestamps (PTS); support variable-frame-rate footage. Decoding runs the `ffmpeg` CLI as a subprocess (licensing: binary FFmpeg bundled in Python wheels such as PyAV's includes GPL encoders). When a GoPro proxy exists, detection and tracking read the proxy and only number reading touches the full-resolution file.
2. **Detect & track** people → *tracklets* (track id, per-frame bbox, timestamps). M1 detector: RF-DETR Small (ONNX via `onnxruntime`, CPU) on the whole proxy frame scaled to 512×512 by ffmpeg. Tracker: bibcut's own ByteTrack-style implementation (constant-velocity Kalman filter, greedy IoU matching in two passes by score) on `numpy` only. No pose model in M1 — no pose weights with a stated permissive license were found.
3. **Filter** tracklets outside the track region of interest or too short (spectators, marshals).
4. **Torso crop**, for both front and back views. M1 estimates it from the person box; pose keypoints (shoulders → hips) can replace the estimate later if it is not good enough.
5. **Legibility** — start with an OCR-confidence threshold; add a dedicated classifier only if needed.
6. **Number recognition** per frame (scene-text recognition / OCR). If a start list is given, constrain candidates to its numbers.
7. **Tracklet consolidation** — confidence-weighted voting across frames. If readings change mid-tracklet (likely a tracker ID switch in a pack), split the tracklet at that point.
8. **Identity merge** — tracklets with the same number form the rider's timeline; merge gaps shorter than *merge gap*, apply *padding*.
9. **Review** — unresolved tracklets with thumbnails for manual labelling; labels feed back into step 8.
10. **Export** (section 5).

Each stage caches its output so later stages can be re-run without re-processing the video.

## 7. Known hard cases

- Board riders stand sideways: depending on stance (regular/goofy) and direction, a trackside camera sees the chest of some riders and the back of others, so both numbers must be supported. In the middle of a pass a board rider may show only their side, with no number visible at all. EUC riders face forward.
- Riders off the track wear numbered jerseys too — waiting at the barrier, in the pit, walking around — and stand closer to the camera than the racers, so their numbers are often *more* legible. Without the track region filter they would be credited with participation.
- Riders on the far side of the track are too small to read (number patch visible, digits not); legibility comes from the parts of the lap closer to the camera.
- Partially hidden numbers read as a valid shorter number (e.g. "53" from "253" or "153"); constraining to the start list does not resolve this on its own.
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
  - "Dependency" includes what a package's published wheels bundle. This excludes PyAV (wheels bundle `libx264`/`libx265`), OpenCV (`opencv-python` and `-headless` 5.0 macOS wheels bundle `libx264`, `libx265`, `libpostproc`, `libvidstab`, `librubberband`), and anything requiring them (`supervision` → PyAV; Roboflow `trackers` → `supervision`, OpenCV). Checked 2026-10-23 by listing the wheels' native libraries.
  - External programs run as subprocesses (the `ffmpeg`/`ffprobe` CLI) are not linked or shipped and are fine, whatever their build's license.
  - Model weights need a stated permissive license. bibcut never commits or redistributes them: it downloads them from a pinned URL and verifies a pinned SHA-256.
- **Privacy:** race footage and start lists are personal data and never go into the repository. Publishing per-rider clips is the publisher's responsibility (riders' consent). No face recognition.
- Runs locally; GPU optional (CPU must work, just slower).

## 10. Milestones

- **M0** — repository, spec, agent guidance.
- **M1** — CLI: on one real video, generates participation intervals for most riders and ffmpeg makes cuts from them. → **switch the repository to public.**
- **M2** — start list mapping, YouTube export, manual review flow, near-camera interval mode.
- **M3** — robustness: handheld footage, multiple videos per event, packaging.

## 11. Open questions

- ~~Camera setup at target races~~ — answered by the sample footage (section 12).
- ~~Number format~~ — answered by the sample footage (section 12).
- ~~Does the GoPro LRV proxy share the exact timeline of its MP4?~~ — yes (section 12).
- Heat structure: the results show six columns per rider and category, presumably rounds. Can a heat schedule be recovered, to cross-check participation intervals?
- OCR backend: PP-OCRv5 English mobile (official ONNX) is the working candidate (section 12); alternatives found: PP-OCRv6 tiny, PP-OCRv4 English, PARSeq via OnnxTR (96 MB, fixed 32×128 input). Constraint from section 9: run as ONNX through `onnxruntime` (the PaddleOCR and RapidOCR packages require OpenCV). Weight licenses are Apache-2.0 per model cards only; training-data terms are not stated.
- How to credit participation when racing riders' numbers are rarely visible from the camera (section 12, number reading)?
- ~~Detector/tracker backend and its license~~ — RF-DETR Small + own tracker (section 6, step 2).
- RF-DETR ONNX provenance: the official weights are PyTorch only; M1 uses a third-party ONNX export (Hugging Face `saifkhichi96/opendetect`, card license apache-2.0) pinned by SHA-256. It works on our footage, but equality with the official weights is not verified. Fallback if it disappears: export from the official weights (needs PyTorch) and mirror the file in a release of this repository (Apache 2.0 allows it).
- Review UI: CLI/TUI vs a small local web page.
- Sample footage for development — riders' consent for anything that leaves the local machine (test fixtures stay synthetic).

## 12. Sample footage (xErace 2025-05-17)

Observed on 2026-10-22 from contact sheets and a handful of full-resolution frames. The footage itself stays outside the repository.

- **Cameras:** two GoPros as a stereo pair (left/right, for 3D), recording almost continuously; 7.3 h per camera, ~380 GB each.
- **Format:** 3840×2160 HEVC, 119.88 fps until ~15:30, then 100 fps. Constant frame rate per file, but time must still come from PTS.
- **Chapters:** GoPro splits long recordings into chapter files (`GX01nnnn`, `GX02nnnn`, `GX03nnnn` = one recording), so one recording can span several files.
- **Proxies:** every MP4 has an `.LRV` proxy (768×432 H.264). Its timeline matches the MP4 frame for frame: identical PTS lists (7805 frames in `GX010206`), and LRV frame *n* matches the downscaled MP4 frame *n* at both ends of the file (pixel MSE ~17 at offset 0 vs ≥ 360 at ±1).
- **Decoding speed** (M1 Max, 10 s of 4K at 119.88 fps): software HEVC 57 fps, VideoToolbox 239 fps, LRV 1421 fps. One camera's day (7.3 h) is therefore ~15 h, ~3.7 h and ~37 min of pure decoding. Through `bibcut.decode` with 10 fps sampling: LRV ~15× real time, 4K via VideoToolbox scaled to 1080p ~1.6× real time.
- **Person detection** (RF-DETR, one gallery frame of `GX010217` with ~10 riders on track and ~7 people off it): the whole frame at 512×512 finds every rider on the track, including the far ones (~60 px tall at 4K); the LRV proxy gives the same on-track result as the downscaled 4K frame. Tiling (2×2, 3×2) adds only a few very distant people plus false positives on a banner showing motorcyclists. RF-DETR Small takes ~0.45 s per frame on the M1 Max CPU; the CoreML execution provider is slower (the graph splits into ~100 partitions) and takes ~2 min to compile.
- **Tracking** (`GX010217`, 200–320 s, 5 fps, 600 frames; `bibcut track` took 9 min, ~0.9 s per frame with another workload on the machine): with IoU matching, 95 % of the confident detections left untracked had no overlapping box in the next sample — a racing rider moves further than their own width in 0.2 s — while the nearest next detection was a median 0.7 (p90 1.3) box heights away. Matching by centre distance in box heights fixed it: 1 % of detections with score ≥ 0.5 stay outside a tracklet (24 % at ≥ 0.3, mostly distant people and a banner). The segment still yields 147 tracklets (median 3.6 s, 48 under 2 s): camera pans, pillars and barriers break them. Joining fragments is left to the identity merge by number (section 6, step 8).
- **Number reading** (same segment; 4,648 torso crops cut from the 4K file where the person box is ≥ 100 px tall; torso = 15–55 % of the box height): PP-OCRv5 English mobile recognizer (official PaddlePaddle ONNX, Apache-2.0 per its model card) run on seven overlapping horizontal bands of each crop, output masked to digits, readings outside the start list dropped, confidence-weighted vote per tracklet. On 21 hand-labelled tracklets (numbers clearly visible): 18 right by vote, 58 % of single crops exact; the three misses had vote sums below 1.5. Tesseract (digits only) as a baseline: 1 right of 19 readings. **But:** of 65 tracklets that move (≥ 1 box height/s, i.e. riders racing), only 1 got a reading with vote sum ≥ 2.0, against 21 of 72 still ones (riders waiting at the barrier or in the pit queue, spectators). From the gallery, racing riders are mostly side-on, crouched and far, and their number is simply not in view. A rider's number is reliably readable before and after the heat, not during it.
- **Camera placement:** mostly on a gallery above the track, moved between files. It sees almost the whole circuit. A few files are from track level, the pit or the registration area. *Correction (2026-10-23):* the camera is **not** static within a file — in `GX010217` the view pans between 240 s and 263 s and is reframed by 315 s. A fixed track region per file will not hold; the region filter must follow the camera (or work per static segment).
- **Lighting:** indoor hall, warm cast, low light; frames are soft (compression, some motion blur).
- **Numbers:** 1–3 digits, 5–268 at this event (58 of the 74 have three digits). Black digits on a white rectangle on a black jersey, with the "xERACE" logo above and below — high contrast.
- **Number size at 4K:** riders on the near side of the track, digits ~35–50 px tall, legible. Middle of the hall, often side-on with no number visible. Far side, digits not legible.
- **Results:** per-category results with start numbers exist (published as screenshots on the organiser's website), so a start list can be built for the event.
