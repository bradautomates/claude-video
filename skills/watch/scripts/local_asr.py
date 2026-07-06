#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#   "faster-whisper>=1.0",
# ]
# ///
"""Fully-local ASR for /watch — no API key, no Groq/OpenAI, no network at inference.

Runs faster-whisper (CTranslate2 backend, CPU-friendly) on an audio/video file
and prints a JSON transcript in /watch's exact segment schema:

    {"language": "en", "model": "base", "segments": [
       {"start": 0.0, "end": 3.2, "text": "Hello there."}, ...
    ]}

watch.py consumes `segments` the same way it consumes captions or the old
Whisper-API output, so filter_range / format_transcript work unchanged.

The engine is chosen to sidestep the network entirely: faster-whisper downloads
its model from HuggingFace ONCE (cached under ~/.cache/huggingface), then runs
100% offline on every subsequent call. Pick the model with --model or the
WATCH_WHISPER_MODEL env var (tiny|base|small|medium|large-v3; default base).

Usage:
    uv run local_asr.py <video-or-audio-path>
    uv run local_asr.py clip.mp4 --model small --language en
    uv run local_asr.py clip.mp4 --language ''        # auto-detect
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_audio(media_path: Path, dest: Path) -> Path:
    """Extract mono 16kHz PCM wav — the format Whisper models expect."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(media_path.resolve()),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — media may have no audio track")
    return dest


def transcribe(audio_path: Path, model_name: str, language: str | None) -> dict:
    # Imported here so --help / arg errors don't pay the import cost.
    from faster_whisper import WhisperModel

    print(f"[local_asr] loading faster-whisper '{model_name}' (cpu, int8)…", file=sys.stderr)
    # int8 keeps memory/latency low and needs no GPU. compute_type auto-falls
    # back to a supported type if int8 is unavailable on the platform.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")

    print("[local_asr] transcribing…", file=sys.stderr)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
        beam_size=5,
        temperature=0,
    )

    segments: list[dict] = []
    for seg in segments_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments.append({
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": text,
        })

    return {
        "language": getattr(info, "language", None) or (language or "unknown"),
        "model": model_name,
        "segments": segments,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="local_asr",
        description="Fully-local transcription (faster-whisper) in /watch's segment schema.",
    )
    ap.add_argument("media", help="Video or audio file path")
    ap.add_argument(
        "--model",
        default=os.environ.get("WATCH_WHISPER_MODEL", "base"),
        help="Whisper model: tiny|base|small|medium|large-v3 (default: base, or $WATCH_WHISPER_MODEL)",
    )
    ap.add_argument(
        "--language",
        default=os.environ.get("WATCH_WHISPER_LANGUAGE", ""),
        help="ISO language code (default: '' = auto-detect, right for arbitrary videos).",
    )
    args = ap.parse_args()

    media = Path(args.media).expanduser().resolve()
    if not media.exists():
        raise SystemExit(f"media not found: {media}")

    with tempfile.TemporaryDirectory() as tmp:
        audio = extract_audio(media, Path(tmp) / "audio.wav")
        result = transcribe(audio, args.model, args.language or None)

    print(
        f"[local_asr] {len(result['segments'])} segments "
        f"(lang={result['language']}, model={result['model']})",
        file=sys.stderr,
    )
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
