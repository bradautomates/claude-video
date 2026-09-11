"""extract-audio.sh CLI wrapper around whisper.extract_audio()."""
from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "extract-audio.sh"


def _build_clip_with_audio(path: Path, duration: float = 2.0) -> None:
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


def test_extract_audio_sh_produces_mp3(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _build_clip_with_audio(video)
    out = tmp_path / "audio.mp3"
    proc = subprocess.run(
        ["bash", str(SCRIPT), str(video), str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
