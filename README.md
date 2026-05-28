# ltc-embed

Extract LTC (Linear/Longitudinal Timecode) from the audio track of video files
and embed it as QuickTime/MP4 timecode metadata — without re-encoding the video
stream.

Designed for GoPro MP4 recordings synced to an external SMPTE LTC generator
(e.g. Cubase, Tentacle Sync, Ambient).

## Requirements

- macOS
- [ffmpeg](https://ffmpeg.org)
- [libltc](https://github.com/x42/libltc) (recommended for robust decoding)
- Python 3 with [NumPy](https://numpy.org) (pure-Python fallback decoder)

## Quickstart

```bash
git clone ...
cd ltc-embed
make install    # brew deps + venv + pip
make run INPUT_DIR=/path/to/videos
```

Files with a valid LTC signal get a copy with `_tc` suffix and embedded timecode.
Files without LTC are skipped with a log message.

## Make targets

```bash
make venv         # create .venv/ with numpy
make install      # brew ffmpeg libltc + venv
make check-deps   # verify tools and libraries
make run          # process videos (configurable via vars)
make clean        # remove all *_tc output files
```

### Configuration

| Variable      | Default  | Description                                    |
|---------------|----------|------------------------------------------------|
| `INPUT_DIR`   | `.`      | Directory with video files                     |
| `FPS`         | *auto*   | Force frame rate (e.g. `FPS=25`)               |
| `SUFFIX`      | `_tc`    | Output filename suffix                         |
| `DURATION`    | `10`     | Seconds of audio to scan for LTC               |
| `OFFSET`      | `0`      | Manual frame offset (±N frames)                |
| `DRIFT_AUTO`  | *unset*  | Set to `1` to auto-correct clock drift         |
| `STRIP_LTC`   | *unset*  | Set to `1` to replace stereo with mono         |
| `OVERWRITE`   | *unset*  | Set to `1` to modify in-place                  |

Examples:

```bash
make run INPUT_DIR=./gopro FPS=25
make run INPUT_DIR=./videos SUFFIX=_timecoded
make run INPUT_DIR=./ OVERWRITE=1 FPS=25
make run INPUT_DIR=./ DRIFT_AUTO=1 STRIP_LTC=1 OVERWRITE=1
```

## CLI usage

```bash
.venv/bin/python3 ltc_embed.py /path/to/videos/
.venv/bin/python3 ltc_embed.py video.mp4 --suffix _tc
.venv/bin/python3 ltc_embed.py video.mov --overwrite --verbose
.venv/bin/python3 ltc_embed.py video.mp4 --fps 25                # force frame rate
.venv/bin/python3 ltc_embed.py video.mp4 --offset -3             # manual offset
.venv/bin/python3 ltc_embed.py video.mp4 --drift-auto            # auto drift correction
.venv/bin/python3 ltc_embed.py video.mp4 --strip-ltc-audio       # mono audio output
.venv/bin/python3 ltc_embed.py /videos/ --duration 30
```

## Features

### FPS auto-detection & offset correction

The script tries multiple zero-crossing hysteresis thresholds and frame rates,
scoring each combination by how many consecutive valid LTC frames it produces.
The winning (threshold, fps) pair is used for decoding.

Once the first valid LTC frame is found, the script measures its **sample offset**
from audio start and back-calculates the actual start timecode:

```
start_tc = first_ltc_tc − round(sample_offset / sample_rate × fps)
```

This compensates for the audio ramp-up time before the LTC signal becomes
detectable — no more manual offset guessing.

### Clock drift detection (head/tail)

After decoding the head segment (first 10 s), the script extracts the **last
10 s** of audio as well and decodes LTC there.  Both timecodes are compared and
the drift is reported:

```
Drift: −5fr over 413s  →  auto-offset −2fr applied, TC now 00:00:11:14
```

Clips shorter than 60 s are excluded (too little measurement distance).

### Auto drift correction (`--drift-auto`)

When enabled, the script automatically centers the timecode so that clock drift
error is symmetrically distributed across the clip duration.  Optimal offset =
`round(drift_frames / 2)`.  Maximum sync error is halved compared to no
correction.

### Manual frame offset (`--offset N`)

Fine-tune the timecode by ±N frames.  `--offset -3` shifts the timecode 3
frames earlier; `+3` shifts it 3 frames later.  Works alongside `--drift-auto`
(offsets are additive).

### LTC audio stripping (`--strip-ltc-audio`)

Replaces the stereo audio track with a mono track carrying **only the right
channel** — the LTC signal on the left channel has already been extracted and
is no longer needed.  Audio is re-encoded as AAC at the original bitrate.

This eliminates the manual per-clip work in DaVinci Resolve (clip attributes →
dual mono → map channels → mute LTC channel).  The output file imports as a
clean single-track mono clip.

## How it works

1. **Audio extraction** — `ffmpeg` extracts the first audio track (left channel
   when stereo, carrying the SMPTE LTC signal) as mono 48 kHz 16-bit WAV.
2. **LTC decoding** — the biphase-mark-encoded signal is decoded via
   Schmitt-trigger zero-crossing detection with hysteresis. Two consecutive
   short periods → bit `1`, one long period → bit `0`. The 80-bit LTC frame
   is identified by its sync word (`0xBFFC`, byte-swapped due to LSB-first
   packing) and the timecode digits are read from the BCD nibble positions.
3. **Sample-offset correction** — the decoder tracks which intervals produced
   each bit and computes the audio-sample offset of the first valid LTC frame.
   The start timecode is back-calculated from this offset.
4. **Head/tail drift check** — tail audio (last 10 s) is decoded and compared
   against the expected timecode. Warnings are issued for clock drift.
5. **Metadata write** — `ffmpeg -timecode HH:MM:SS:FF` embeds the timecode.
   Existing tmcd tracks are stripped to avoid duplicates.

Two decoder backends are supported:

| Backend | Install               | Notes                           |
|---------|-----------------------|---------------------------------|
| libltc  | `brew install libltc` | Primary, production-grade       |
| NumPy   | `pip install numpy`   | Pure Python, auto-FPS detection |

## Supported formats

MP4, MOV, M4V, AVI, MKV, MTS, M2T, MXF
