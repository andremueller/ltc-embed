#!/usr/bin/env python3
"""
Extract LTC (Linear/Longitudinal Timecode) from the first audio track of video files
and embed it as QuickTime/MP4 timecode metadata using ffmpeg.

Works with GoPro MP4 files that have an external LTC timecode generator recording
onto the audio track.

Requirements (macOS):
    brew install ffmpeg
    pip install numpy

Optional (for more robust LTC decoding):
    brew install libltc

Usage:
    python ltc_embed.py /path/to/videos/
    python ltc_embed.py video.mp4 --fps 25 --suffix _tc
"""

import subprocess
import sys
import os
import struct
import json
import shutil
import argparse
import logging
import tempfile
import re
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ltc_embed")

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".m2t", ".mxf"}

# ---------------------------------------------------------------------------
# Biphase mark decoder (pure Python, numpy-accelerated)
# ---------------------------------------------------------------------------

LTC_SYNC_WORD = 0xBFFC
LTC_FRAME_BITS = 80


def find_zero_crossings(samples, threshold=0.02):
    """Detect zero-crossing indices with Schmitt-trigger hysteresis.

    Handles signals that start at zero (silence) by latching onto the
    first excursion beyond the threshold in either direction.
    """
    state = 0
    crossings = []

    for i in range(len(samples)):
        s = samples[i]
        if state == 0:
            if s > threshold:
                state = 1
                crossings.append(i)
            elif s < -threshold:
                state = -1
                crossings.append(i)
        elif state == 1 and s < -threshold:
            state = -1
            crossings.append(i)
        elif state == -1 and s > threshold:
            state = 1
            crossings.append(i)
    return crossings


def intervals_from_crossings(crossings):
    """Compute sample-distance between consecutive zero crossings."""
    return [(crossings[i + 1] - crossings[i]) for i in range(len(crossings) - 1)]


def decode_biphase(intervals, bit_period_samples):
    """Decode biphase-mark encoded bits from interval lengths.

    A '0' bit produces one long period (≈ bit_period).
    A '1' bit produces two short periods (≈ bit_period/2 each).
    """
    threshold = bit_period_samples * 0.75

    shorts = [True if p < threshold else False for p in intervals]
    bits = []
    i = 0
    while i < len(shorts):
        if shorts[i] and i + 1 < len(shorts) and shorts[i + 1]:
            bits.append(1)
            i += 2
        else:
            bits.append(0)
            i += 1
    return bits


def bits_to_bytes(bits):
    """Pack bit stream into bytes (LSB-first: bit 0 → byte bit 0)."""
    data = []
    for i in range(0, len(bits) - len(bits) % 8, 8):
        byte = 0
        for j in range(8):
            if bits[i + j]:
                byte |= 1 << j
        data.append(byte)
    return bytes(data)


def find_sync_offset(data):
    """Search for the LTC sync word in a byte buffer.

    With LSB-first byte packing the sync word appears byte-swapped
    compared to libltc's MSB-first packing: [0xFC, 0xBF] instead of [0xBF, 0xFC].

    Returns the byte offset where the sync pair *starts* (lower byte), or -1.
    """
    for i in range(len(data) - 1):
        if ((data[i + 1] << 8) | data[i]) == LTC_SYNC_WORD:
            return i
    return -1


def parse_timecode(frame_bytes):
    """Extract timecode from an 80-bit LTC frame (10 bytes).

    Byte layout (libltc-compatible, LSB-first nibble packing):
      byte 0 bits 3-0 : frame units (BCD)
      byte 1 bits 1-0 : frame tens  (BCD)
      byte 2 bits 3-0 : seconds units (BCD)
      byte 3 bits 2-0 : seconds tens  (BCD)
      byte 4 bits 3-0 : minutes units (BCD)
      byte 5 bits 2-0 : minutes tens  (BCD)
      byte 6 bits 3-0 : hours units   (BCD)
      byte 7 bits 1-0 : hours tens    (BCD)
      bytes 8-9       : sync word
    """
    if len(frame_bytes) < 10:
        return None

    frame_units = frame_bytes[0] & 0x0F
    frame_tens = frame_bytes[1] & 0x03
    secs_units = frame_bytes[2] & 0x0F
    secs_tens = frame_bytes[3] & 0x07
    mins_units = frame_bytes[4] & 0x0F
    mins_tens = frame_bytes[5] & 0x07
    hours_units = frame_bytes[6] & 0x0F
    hours_tens = frame_bytes[7] & 0x03

    ff = frame_tens * 10 + frame_units
    ss = secs_tens * 10 + secs_units
    mm = mins_tens * 10 + mins_units
    hh = hours_tens * 10 + hours_units

    if ff > 99 or ss > 59 or mm > 59 or hh > 23:
        return None

    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


CANDIDATE_FPS = [25, 30, 24, 29.97]
LTC_SYNC_WORD = 0xBFFC
LTC_FRAME_BITS = 80
LTC_FRAME_BYTES = 10


def _timecode_to_frame_number(tc_str, fps):
    """Convert HH:MM:SS:FF to absolute frame number."""
    hh, mm, ss, ff = (int(x) for x in tc_str.split(":"))
    return ((hh * 3600) + (mm * 60) + ss) * fps + ff


def _decode_frame_from_bits(bits, start_byte, fps):
    """Decode a single LTC frame from a byte buffer at the given offset.

    Returns (timecode_str, frame_number) or (None, None).
    """
    if start_byte + LTC_FRAME_BYTES > len(bits):
        return None, None
    f = bits[start_byte : start_byte + LTC_FRAME_BYTES]
    tc = parse_timecode(f)
    if tc is None:
        return None, None
    fn = _timecode_to_frame_number(tc, fps)
    return tc, fn


def _score_fps(intervals, sample_rate, fps):
    """Score how well a given FPS explains the biphase intervals.

    Returns (first_timecode, consecutive_frame_count).
    """
    bit_period = sample_rate / (fps * LTC_FRAME_BITS)
    threshold = bit_period * 0.75

    bits = decode_biphase(intervals, bit_period)
    data = bits_to_bytes(bits)
    if len(data) < LTC_FRAME_BYTES:
        return None, 0

    sync_idx = find_sync_offset(data)
    if sync_idx < 0:
        return None, 0

    frame_start = sync_idx - (LTC_FRAME_BYTES - 2)
    if frame_start < 0:
        return None, 0

    first_tc, _ = _decode_frame_from_bits(data, frame_start, fps)
    if first_tc is None:
        return None, 0

    # Count consecutive frames at 10-byte spacings
    consecutive = 1
    for offset in range(
        frame_start + LTC_FRAME_BYTES,
        len(data) - LTC_FRAME_BYTES + 1,
        LTC_FRAME_BYTES,
    ):
        tc, _ = _decode_frame_from_bits(data, offset, fps)
        if tc is None:
            break
        consecutive += 1

    return first_tc, consecutive


def decode_ltc_numpy(samples, sample_rate, fps=None):
    """Decode LTC from raw audio samples using numpy.

    If fps is None, auto-detects by scoring all candidate frame rates
    and picking the one that produces the most consecutive valid frames.
    Returns (timecode_str, used_fps) or (None, None).
    """
    crossings = find_zero_crossings(samples, threshold=0.02)
    if len(crossings) < 20:
        return None, None

    intervals = intervals_from_crossings(crossings)
    if len(intervals) == 0:
        return None, None

    fps_list = [fps] if fps is not None else list(CANDIDATE_FPS)

    best_tc, best_fps, best_score = None, None, 0
    for try_fps in fps_list:
        tc, score = _score_fps(intervals, sample_rate, try_fps)
        if tc and score > best_score:
            best_tc, best_fps, best_score = tc, try_fps, score

    return best_tc, best_fps


def decode_ltc_from_wav(wav_path, fps=None, max_seconds=10):
    """Read a WAV file and decode LTC from it.

    If fps is None, the LTC decoder will auto-detect the frame rate.
    Returns (timecode_str, actual_fps) or (None, None).
    """
    with open(wav_path, "rb") as f:
        # Parse minimal WAV header
        riff = f.read(4)
        if riff != b"RIFF":
            log.error("Not a valid RIFF/WAV file")
            return None, None
        f.read(4)  # file size
        wave = f.read(4)
        if wave != b"WAVE":
            log.error("Not a valid WAVE file")
            return None, None

        # Find fmt chunk
        sample_rate = 48000
        num_channels = 1
        bits_per_sample = 16
        fmt_found = False
        while not fmt_found:
            chunk_id = f.read(4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                if len(fmt_data) >= 10:
                    audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                    num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                    sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                    bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                fmt_found = True
            else:
                f.read(chunk_size)

        # Find data chunk
        while True:
            try:
                chunk_id = f.read(4)
                chunk_size = struct.unpack("<I", f.read(4))[0]
                if chunk_id == b"data":
                    break
                f.read(chunk_size)
            except struct.error:
                log.error("Could not find data chunk in WAV")
                return None, None

        bytes_per_sample = bits_per_sample // 8
        max_frames = int(sample_rate * max_seconds)
        bytes_to_read = min(chunk_size, max_frames * num_channels * bytes_per_sample)
        raw = f.read(bytes_to_read)

    # Convert to numpy array
    if bits_per_sample == 16:
        dtype = np.int16
    elif bits_per_sample == 24:
        dtype = np.int32
    elif bits_per_sample == 32:
        dtype = np.int32
    else:
        log.error(f"Unsupported bit depth: {bits_per_sample}")
        return None, None

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    # Extract first channel if multi-channel
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels)
        audio = audio[:, 0]

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    log.debug(
        f"WAV: {len(audio)} samples, {sample_rate} Hz, {num_channels} ch, {bits_per_sample} bit"
    )

    return decode_ltc_numpy(audio, sample_rate, fps)


# ---------------------------------------------------------------------------
# External LTC decoder (libltc / ltcdecode)
# ---------------------------------------------------------------------------


def libltc_available():
    return shutil.which("ltcdecode") is not None


def decode_with_ltcdecode(wav_path, fps=None):
    """Use libltc's ltcdecode CLI to decode LTC from a WAV file.

    Returns (timecode_str, fps) or (None, None).
    """
    cmd = ["ltcdecode", str(wav_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return None, None
    except subprocess.TimeoutExpired:
        log.warning("ltcdecode timed out")
        return None, None

    if result.returncode != 0:
        log.debug(f"ltcdecode error: {result.stderr.strip()}")
        return None, None

    # Parse timecode from first output line
    for line in result.stdout.strip().split("\n"):
        m = re.search(r"(\d{2})[:;.,](\d{2})[:;.,](\d{2})[:;.,](\d{2})", line)
        if m:
            return f"{m.group(1)}:{m.group(2)}:{m.group(3)}:{m.group(4)}", fps

    return None, None


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def get_video_info(video_path):
    """Extract fps and other metadata from a video via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    # Find video stream fps
    fps = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "")
            if "/" in r_frame_rate:
                num, denom = r_frame_rate.split("/")
                if int(denom) > 0:
                    fps_val = float(num) / float(denom)
                    if fps_val > 0:
                        fps = round(fps_val)
            elif r_frame_rate:
                try:
                    fps = int(float(r_frame_rate))
                except ValueError:
                    pass
            break

    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    return {"fps": fps, "has_audio": has_audio}


def extract_audio_to_wav(video_path, wav_path, max_seconds=10):
    """Extract first audio track as mono 48kHz 16-bit WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "48000",
        "-sample_fmt",
        "s16",
        "-t",
        str(max_seconds),
        "-f",
        "wav",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    return wav_path


def find_tmcd_streams(video_path):
    """Return a list of stream indices that carry existing QuickTime tmcd tracks.

    These must be excluded when re-writing timecode to avoid duplicate
    timecode tracks in the output.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    tmcd = []
    for s in info.get("streams", []):
        if s.get("codec_type") == "data" and s.get("codec_tag_string") == "tmcd":
            tmcd.append(s["index"])
    return tmcd


def write_timecode_to_video(video_path, timecode_str, output_path):
    """Embed timecode metadata into video without re-encoding.

    Existing tmcd tracks are stripped first so the new timecode is the
    only one — avoids duplicate timecode entries that confuse NLEs.
    Uses -map 0 to preserve all streams (including GoPro gpmd metadata).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0",
    ]
    for idx in find_tmcd_streams(video_path):
        cmd += ["-map", f"-0:{idx}"]
    cmd += [
        "-c",
        "copy",
        "-timecode",
        timecode_str,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg metadata write failed: {result.stderr.strip()}"
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced empty or missing output file")
    return output_path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

OUTPUT_SUFFIX = "_tc"


def process_file(video_path, fps=None, suffix=OUTPUT_SUFFIX, overwrite=False, max_duration=10):
    """Process a single video file: extract LTC audio, decode timecode, embed.

    Returns True on success, False if skipped, raises on error.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        log.error(f"File not found: {video_path}")
        return False

    ext = video_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        log.info(f"SKIP (unsupported format): {video_path.name}")
        return False

    info = get_video_info(video_path)
    if info is None:
        log.error(f"Could not read video metadata: {video_path.name}")
        return False

    if not info["has_audio"]:
        log.info(f"SKIP (no audio track): {video_path.name}")
        return False

    if fps is None:
        fps = info.get("fps")
        if fps is None:
            log.warning(f"No FPS info, defaulting to 25: {video_path.name}")
            fps = 25

    log.info(f"Processing: {video_path.name} (fps={fps})")

    # Extract audio to temporary WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        extract_audio_to_wav(video_path, wav_path, max_duration)

        # Try libltc first, fall back to pure Python
        tc = None
        detected_fps = fps
        if libltc_available():
            log.debug("  Using ltcdecode (libltc)")
            tc, detected_fps = decode_with_ltcdecode(wav_path, fps)

        if tc is None:
            if not HAS_NUMPY:
                log.error(
                    "libltc not installed and numpy missing. "
                    "Install one:  brew install libltc   OR   pip install numpy"
                )
                return False
            log.debug("  Using pure Python LTC decoder")
            tc, detected_fps = decode_ltc_from_wav(wav_path, fps)

        if tc is None:
            log.info(f"SKIP (no LTC signal found): {video_path.name}")
            return False

        fps_str = f" @{detected_fps}fps" if detected_fps and detected_fps != fps else ""
        log.info(f"  Timecode: {tc}{fps_str}")

        # Determine output path
        if overwrite:
            temp_output = video_path.parent / f"._ltc_tmp_{video_path.name}"
            try:
                write_timecode_to_video(video_path, tc, temp_output)
                temp_output.replace(video_path)
                log.info(f"  Updated (in-place): {video_path.name}")
            finally:
                if temp_output.exists():
                    temp_output.unlink()
        else:
            stem = video_path.stem
            output_path = video_path.parent / f"{stem}{suffix}{ext}"
            write_timecode_to_video(video_path, tc, output_path)
            log.info(f"  Written: {output_path.name}")

        return True

    finally:
        if wav_path.exists():
            wav_path.unlink()


def process_directory(
    directory, fps=None, suffix=OUTPUT_SUFFIX, overwrite=False, max_duration=10
):
    """Scan directory for video files and process each."""
    directory = Path(directory)
    if not directory.is_dir():
        log.error(f"Not a directory: {directory}")
        return

    videos = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not videos:
        log.warning(f"No video files found in {directory}")
        return

    log.info(f"Found {len(videos)} video file(s) in {directory}")
    success = 0
    skipped = 0
    failed = 0

    for video in videos:
        try:
            result = process_file(video, fps, suffix, overwrite, max_duration)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as exc:
            log.error(f"FAILED: {video.name} — {exc}")
            failed += 1

    log.info(f"Done: {success} ok, {skipped} skipped, {failed} failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract LTC audio timecode from video files and embed as metadata.",
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Video file(s) or directory containing video files",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override frame rate (auto-detected from video if omitted)",
    )
    parser.add_argument(
        "--suffix",
        default=OUTPUT_SUFFIX,
        help=f"Output filename suffix (default: '{OUTPUT_SUFFIX}')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original file instead of creating _tc copy",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="SEC",
        help="Seconds of audio to analyze for LTC (default: 10)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not ffmpeg_available():
        log.error("ffmpeg not found. Install:  brew install ffmpeg")
        sys.exit(1)

    if not libltc_available() and not HAS_NUMPY:
        log.error(
            "No LTC decoder available. Install one:\n"
            "  brew install libltc    (recommended)\n"
            "  pip install numpy      (pure Python fallback)"
        )
        sys.exit(1)

    for path in args.input:
        p = Path(path)
        if p.is_dir():
            process_directory(p, args.fps, args.suffix, args.overwrite, args.duration)
        elif p.is_file():
            try:
                process_file(p, args.fps, args.suffix, args.overwrite, args.duration)
            except Exception as exc:
                log.error(f"FAILED: {p.name} — {exc}")
        else:
            log.error(f"Not found: {path}")


if __name__ == "__main__":
    main()
