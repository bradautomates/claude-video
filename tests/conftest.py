"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def build_cut_clip(
    path: Path,
    n: int = 14,
    seg: float = 0.4,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """Concatenate ``n`` solid-color segments into one clip with ``n`` cuts.

    Each color change is a hard scene cut, so the scene selector finds ~n-1
    changes. x264's own scenecut detection is unreliable on flat fills, so we
    force a keyframe at every ``seg`` boundary — giving ~n real keyframes for
    the keyframe engine to find.
    """
    inputs: list[str] = []
    for i in range(n):
        color = COLORS[i % len(COLORS)]
        inputs += ["-f", "lavfi", "-t", str(seg), "-i", f"color=c={color}:s={size}:r={fps}"]
    streams = "".join(f"[{i}:v]" for i in range(n))
    filt = f"{streams}concat=n={n}:v=1:a=0[out]"
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs,
        "-filter_complex", filt, "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg})",
        str(path),
    ])


def build_static_clip(
    path: Path,
    duration: float = 3.0,
    size: str = "320x240",
    fps: int = 10,
) -> None:
    """One solid color: 1 keyframe, no scene changes → triggers both fallbacks."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-t", str(duration), "-i", f"color=c=blue:s={size}:r={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "600",
        str(path),
    ])


@pytest.fixture(scope="session")
def cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "cuts.mp4"
    build_cut_clip(path)
    return path


@pytest.fixture(scope="session")
def static_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("clips") / "static.mp4"
    build_static_clip(path)
    return path


def make_stub_yt_dlp(
    bin_dir: Path,
    *,
    vtt_text: str | None,
    duration: float = 8.0,
    title: str = "Stub Video",
    version: str = "2026.07.04",
) -> Path:
    """Write a fake ``yt-dlp`` executable that never touches the network.

    Mimics the two yt-dlp invocations watch.py makes for a URL source:

    - ``--skip-download`` (download.fetch_captions): "succeeds" -- writes
      video.info.json and, when ``vtt_text`` is given, video.en.vtt -- then
      exits 0. Pass ``vtt_text=None`` to simulate a video with no captions
      available at all.
    - the real download attempt (download.download_url): always fails with a
      403 on the media stream and never writes a video file. This reproduces
      the real 2026-09-17 incident shape: captions download fine, only the
      media stream 403s.

    Synthesized, not vendored: no real caption track or media file is ever
    written into the repo, only generated fresh per test into a tmp dir.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "yt-dlp"
    script = f'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

argv = sys.argv[1:]

if "--version" in argv:
    print({version!r})
    sys.exit(0)


def _opt(name):
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


out_tmpl = _opt("-o")
out_dir = Path(out_tmpl).parent if out_tmpl else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

if "--skip-download" in argv:
    info = {{
        "title": {title!r},
        "uploader": "Stub Channel",
        "duration": {duration!r},
        "webpage_url": argv[-1] if argv else "",
    }}
    (out_dir / "video.info.json").write_text(json.dumps(info), encoding="utf-8")
    vtt_text = {vtt_text!r}
    if vtt_text is not None:
        (out_dir / "video.en.vtt").write_text(vtt_text, encoding="utf-8")
    sys.exit(0)
else:
    sys.stderr.write(
        "ERROR: unable to download video data: HTTP Error 403: Forbidden\\n"
    )
    sys.exit(1)
'''
    stub.write_text(script, encoding="utf-8")
    stub.chmod(0o755)
    return stub


@pytest.fixture
def stub_yt_dlp(tmp_path: Path):
    """Factory fixture: build a PATH-prepend-able dir holding a fake yt-dlp.

    Usage: ``bin_dir = stub_yt_dlp(vtt_text="WEBVTT\\n...")`` then run the
    subprocess under test with ``PATH=f"{bin_dir}{os.pathsep}{PATH}"``.
    """
    def _make(vtt_text: str | None, **kwargs) -> Path:
        bin_dir = tmp_path / "stub-bin"
        make_stub_yt_dlp(bin_dir, vtt_text=vtt_text, **kwargs)
        return bin_dir

    return _make
