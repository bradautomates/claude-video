#!/usr/bin/env python3
"""Audio extraction for /watch's local Voicebox transcription fallback.

Pure stdlib — ffmpeg does the real work. (Previously also held Groq/OpenAI
Whisper API clients; deleted along with the paid-API fallback — see
docs/superpowers/specs/2026-08-13-claude-video-fork-design.md.)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def extract_audio(video_path: str, out_path: Path) -> Path:
    """Extract mono 16kHz 64kbps mp3 — ~480 kB/min, fits Voicebox easily."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not installed. Install with: brew install ffmpeg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(Path(video_path).resolve()),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "64k",
        str(out_path.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg audio extraction failed: {result.stderr.strip()}")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit("ffmpeg produced no audio — video may have no audio track")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: whisper.py <video-path> [<audio-out.mp3>]", file=sys.stderr)
        raise SystemExit(2)
    video = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("audio.mp3")
    extract_audio(video, out)
    print(f"[watch] audio extracted: {out}", file=sys.stderr)
