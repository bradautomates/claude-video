"""Audio extraction for the Voicebox transcription fallback."""
from __future__ import annotations

import subprocess
from pathlib import Path

import whisper


def _build_clip_with_audio(path: Path, duration: float = 2.0) -> None:
    """One video stream + one sine-tone audio stream, muxed together."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(duration), "-i", "color=c=blue:s=320x240:r=10",
            "-f", "lavfi", "-t", str(duration), "-i", "sine=frequency=440:sample_rate=44100",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(path),
        ],
        check=True,
    )


def _build_silent_clip(path: Path, duration: float = 1.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(duration), "-i", "color=c=red:s=320x240:r=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_extract_audio_produces_mp3(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _build_clip_with_audio(video)
    out = whisper.extract_audio(str(video), tmp_path / "audio.mp3")
    assert out.exists()
    assert out.stat().st_size > 0


def test_extract_audio_raises_on_no_audio_track(tmp_path: Path):
    video = tmp_path / "silent.mp4"
    _build_silent_clip(video)
    raised = False
    try:
        whisper.extract_audio(str(video), tmp_path / "audio.mp3")
    except SystemExit:
        raised = True
    assert raised, "expected SystemExit when video has no audio track"
