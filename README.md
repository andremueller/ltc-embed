# ltc-embed

Extract LTC (Linear/Longitudinal Timecode) from the audio track of video files
and embed it as QuickTime/MP4 timecode metadata — without re-encoding.

Designed for GoPro MP4 recordings synced to an external LTC generator.

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

The frame rate is auto-detected — the script tries common rates (25, 30, 24, 29.97)
and picks the one that produces the most consecutive valid LTC frames.  If your
video metadata says 25 fps but the timecode generator runs at 30 fps, the script
detects the discrepancy and logs it.

## Make targets

```bash
make venv         # create .venv/ with numpy
make install      # brew ffmpeg libltc + venv
make check-deps   # verify tools and libraries
make run          # process videos (configurable via vars)
make clean        # remove all *_tc output files
```

### Configuration

| Variable   | Default | Description                               |
|------------|---------|-------------------------------------------|
| `INPUT_DIR` | `.`     | Directory with video files                |
| `FPS`      | *auto*  | Override frame rate (e.g. `FPS=25`)       |
| `SUFFIX`   | `_tc`   | Output filename suffix                    |
| `DURATION` | `10`    | Seconds of audio to scan for LTC          |
| `OVERWRITE`| *unset* | Set to `1` to modify in-place             |

Examples:

```bash
make run INPUT_DIR=./gopro
make run INPUT_DIR=./videos SUFFIX=_timecoded
make run INPUT_DIR=./ OVERWRITE=1                 # overwrite originals
make run INPUT_DIR=./ FPS=30                      # force 30 fps
```

## CLI usage

```bash
.venv/bin/python3 ltc_embed.py /path/to/videos/
.venv/bin/python3 ltc_embed.py video.mp4 --suffix _tc
.venv/bin/python3 ltc_embed.py video.mov --overwrite --verbose
.venv/bin/python3 ltc_embed.py video.mp4 --fps 30        # force frame rate
.venv/bin/python3 ltc_embed.py /videos/ --duration 30
```

## How it works

1. **Audio extraction** — `ffmpeg` extracts the first audio track (left channel
   when stereo, carrying the SMPTE LTC signal) as mono 48 kHz 16-bit WAV.
2. **LTC decoding** — the biphase-mark-encoded signal is decoded via
   Schmitt-trigger zero-crossing detection with hysteresis.  Two consecutive
   short periods → bit `1`, one long period → bit `0`.  The 80-bit LTC frame
   is identified by its sync word (`0xBFFC`, byte-swapped due to LSB-first
   packing) and the timecode digits are read from the BCD nibble positions.
   The frame rate is auto-detected by scoring candidate FPS values against
   how many consecutive valid frames each produces.
3. **Metadata write** — `ffmpeg -c copy -timecode HH:MM:SS:FF` embeds the
   timecode without re-encoding the video stream.

Two decoder backends are supported:

| Backend    | Install                   | Notes                              |
|------------|---------------------------|------------------------------------|
| libltc     | `brew install libltc`     | Primary, production-grade          |
| NumPy      | `pip install numpy`       | Pure Python, auto-FPS detection    |

## Supported formats

MP4, MOV, M4V, AVI, MKV, MTS, M2T, MXF
