"""Shared pytest fixtures: ffmpeg-synthesized clips and scripts/ on sys.path."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Isolation from the developer's real config (#96, PR #106).
#
# Every script module computes CONFIG_DIR/CONFIG_FILE from Path.home() at
# import time, and test modules import them at collection — before any
# fixture runs. So HOME is redirected here, at conftest import, to a scratch
# dir; the autouse fixture below re-asserts it per test and strips the shell
# variables that would otherwise change what watch.py / setup.py resolve.
# ---------------------------------------------------------------------------
import tempfile as _tempfile

_ISOLATED_HOME = Path(_tempfile.mkdtemp(prefix="watch-test-home-"))
os.environ["HOME"] = str(_ISOLATED_HOME)
os.environ["USERPROFILE"] = str(_ISOLATED_HOME)  # Windows
for _var in ("WATCH_DETAIL", "WATCH_WHISPER_BACKEND", "WATCH_YTDLP", "WATCH_MAX_FPS",
             "GROQ_API_KEY", "OPENAI_API_KEY", "SETUP_COMPLETE",
             "WATCH_WHISPER_BASE_URL", "WATCH_COOKIES_FILE", "WATCH_COOKIES_FROM_BROWSER"):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real ~/.config/watch/.env and shell env out of tests.

    Subprocess-based tests inherit os.environ, so a customised WATCH_DETAIL or
    a real API key on the developer's machine would otherwise change what
    watch.py / setup.py resolve (#96).
    """
    monkeypatch.setenv("HOME", str(_ISOLATED_HOME))
    monkeypatch.setenv("USERPROFILE", str(_ISOLATED_HOME))
    for var in ("WATCH_DETAIL", "WATCH_WHISPER_BACKEND", "GROQ_API_KEY", "OPENAI_API_KEY", "SETUP_COMPLETE"):
        monkeypatch.delenv(var, raising=False)

# Make the bundled scripts importable (mirrors watch.py's sys.path insert).
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# 14 visually distinct fills → 14 abrupt cuts → x264 emits a keyframe per cut.
COLORS = [
    "red", "green", "blue", "white", "black", "yellow", "cyan",
    "magenta", "gray", "orange", "purple", "brown", "navy", "olive",
]


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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


def build_clustered_clip(
    path: Path,
    n_cards: float = 12,
    card: float = 0.7,
    tail: float = 25.0,
    size: str = "320x180",
    fps: int = 15,
    with_audio: bool = True,
) -> None:
    """The shape a scene detector gets wrong: hard cuts bunched at the head,
    then a long body whose only boundary is a *fade* through black.

    This is the SBC-2025 booth loop in miniature — card, card, card, then
    minutes of gameplay that fades between segments. The cards clear
    SCENE_MIN_FRAMES but fail the coverage check, so it is the exact input that
    used to make the uniform fallback delete every cut; and the fade in the
    middle is invisible to `select=gt(scene,…)` but plain to `blackdetect`.
    """
    n = int(n_cards)
    # Cards must be BRIGHT, not merely non-black: blackdetect's pix_th=0.10
    # counts any pixel under 10% luma, and navy (0,0,128) is ~5.7% — so a navy
    # card reads as a fade and the fixture would assert its own confusion.
    card_colors = ["red", "green", "white", "yellow", "cyan", "magenta",
                   "orange", "gray", "lime", "aqua", "pink", "silver"]
    inputs: list[str] = []
    for i in range(n):
        inputs += ["-f", "lavfi", "-t", str(card), "-i", f"color=c={card_colors[i % len(card_colors)]}:s={size}:r={fps}"]
    half = tail / 2
    inputs += ["-f", "lavfi", "-t", str(half), "-i", f"testsrc=s={size}:r={fps}"]
    inputs += ["-f", "lavfi", "-t", "1.0", "-i", f"color=c=black:s={size}:r={fps}"]
    inputs += ["-f", "lavfi", "-t", str(half), "-i", f"smptebars=s={size}:r={fps}"]
    total = n * card + tail + 1.0
    if with_audio:
        inputs += ["-f", "lavfi", "-t", str(total), "-i", f"sine=frequency=440:duration={total}"]

    a, blk, b = n, n + 1, n + 2
    cards = "".join(f"[{i}:v]" for i in range(n))
    filt = (
        f"[{a}:v]fade=t=out:st={half - 0.7:.2f}:d=0.7[fa];"
        f"[{b}:v]fade=t=in:st=0:d=0.7[fb];"
        f"{cards}[fa][{blk}:v][fb]concat=n={n + 3}:v=1:a=0[out]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *inputs, "-filter_complex", filt, "-map", "[out]",
    ]
    if with_audio:
        cmd += ["-map", f"{n + 3}:a", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "60", str(path)]
    _run(cmd)


@pytest.fixture(scope="session")
def clustered_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Cuts bunched at the head + a mid-clip fade. Triggers the hybrid engine."""
    path = tmp_path_factory.mktemp("clips") / "clustered.mp4"
    build_clustered_clip(path)
    return path


@pytest.fixture(scope="session")
def clustered_clip_silent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Same shape with no audio stream — silencedetect must be skipped, not fail."""
    path = tmp_path_factory.mktemp("clips") / "clustered_silent.mp4"
    build_clustered_clip(path, with_audio=False)
    return path


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


@pytest.fixture(scope="session")
def long_cut_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A longer (~42s) clip with visibly changing content, for uniform
    extract() spread tests. A short clip like ``cut_clip`` can't distinguish
    "sampled from the head" from "spread across the range" — the two only
    diverge once there's real duration between them."""
    path = tmp_path_factory.mktemp("clips") / "long_cuts.mp4"
    build_cut_clip(path, n=14, seg=3.0, size="320x240", fps=5)
    return path


# ---------------------------------------------------------------------------
# Host-binary isolation
#
# Tests that exercise argv construction or the setup state machine care about
# *logic*, not about whether the developer's machine happens to have ffmpeg and
# yt-dlp installed. Both code paths guard with `shutil.which(...)` and bail
# early, so without isolation those tests fail on a clean checkout for reasons
# unrelated to what they assert. `fake_bin_dir` supplies no-op executables to
# put on PATH so the guard passes deterministically everywhere.
# ---------------------------------------------------------------------------

STUB_BINARIES = ("ffmpeg", "ffprobe", "yt-dlp", "deno")  # deno: keep the JS-runtime preflight off host state


@pytest.fixture(scope="session")
def fake_bin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A PATH entry holding no-op stand-ins for every required binary."""
    bin_dir = tmp_path_factory.mktemp("fakebin")
    for name in STUB_BINARIES:
        if os.name == "nt":
            # shutil.which on Windows only matches PATHEXT extensions, so a
            # bare shebang file is invisible there and the real binary wins.
            (bin_dir / f"{name}.bat").write_text("@echo off\r\nexit /b 0\r\n", encoding="utf-8")
            continue
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        stub.chmod(0o755)
    return bin_dir


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
    if os.name == "nt":
        # Windows: no shebang launch and no .bat shim (cmd.exe would read the
        # `<` in the -f selector as a redirection). Tests point WATCH_YTDLP at
        # `python yt-dlp.py` instead — see ytdlp_env().
        stub = bin_dir / "yt-dlp.py"
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


def ytdlp_env(bin_dir: Path) -> dict[str, str]:
    """Environment additions that make watch.py run the stub in *bin_dir*."""
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    if os.name == "nt":
        env["WATCH_YTDLP"] = f'"{sys.executable}" "{bin_dir / "yt-dlp.py"}"'
    return env
