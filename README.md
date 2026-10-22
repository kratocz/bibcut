# bibcut

Find every rider in race footage by their start number — and turn it into per-rider timelines, YouTube timestamps and highlight cuts.

> **Status:** early development (pre-alpha). Nothing works yet; see [docs/SPEC.md](docs/SPEC.md) for what is planned.

## What it will do

Point bibcut at a video of a race on a circuit (electric skateboards, OneWheels, bikes, runners — anyone wearing a number on the chest or back) and it will:

1. detect and track every rider in the footage,
2. read the number on their jersey or bib — front or back — whenever it becomes legible,
3. credit the rider for the **whole** time they were in frame, including the seconds before the number could be read,
4. optionally map numbers to names from a start list,
5. export per-rider intervals as JSON/CSV, YouTube description timestamps, and ffmpeg cuts of each rider's moments.

```text
#7   Rider Name
     00:01:20 – 00:01:26
     00:03:41 – 00:03:49
#12  Another Rider
     00:01:22 – 00:01:27
```

## How it works (in short)

Person detection and multi-object tracking split the video into *tracklets* (one rider, one continuous appearance). A number read anywhere inside a tracklet labels the entire tracklet; per-frame readings are combined by confidence-weighted voting. Tracklets with the same number are then merged into a per-rider timeline. Tracklets whose number is never legible go to a short manual review.

## License

[MIT](LICENSE) © 2026 Petr Kratochvil
