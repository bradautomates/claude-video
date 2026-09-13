"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English, so the request must stay
bounded to the English-only pattern.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_english_only(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert all(t.startswith("en") for t in tokens), f"sub-langs must be English-only, got {langs!r}"


def test_fetch_captions_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))


# --- gallery-dl fallback (image slideshows) ---------------------------------

TIKTOK = "https://www.tiktok.com/@iloveeestrayyykidsss/photo/7463596003225013536"


def _fake_tools(monkeypatch, *, ytdlp_files=(), gallery_files=(), gallery_dl=True, info=None):
    """Stub yt-dlp/gallery-dl: each 'writes' the given files into its output dir."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "yt-dlp":
            out = Path(cmd[cmd.index("-o") + 1]).parent
            for name in ytdlp_files:
                (out / name).write_bytes(b"x")
        elif cmd[0] == "gallery-dl":
            out = Path(cmd[cmd.index("-D") + 1])
            out.mkdir(parents=True, exist_ok=True)
            for name in gallery_files:
                (out / name).write_bytes(b"x")
            if info is not None:
                (out / "info.json").write_text(json.dumps(info), encoding="utf-8")
        return _Result()

    real_which = download.shutil.which
    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(
        download.shutil, "which",
        lambda name: (None if name == "gallery-dl" and not gallery_dl else real_which(name) or f"/usr/bin/{name}"),
    )
    return calls


def test_audio_only_ytdlp_result_falls_back_to_slides(monkeypatch, tmp_path):
    # yt-dlp on a slideshow /video/ URL returns just the soundtrack.
    calls = _fake_tools(
        monkeypatch,
        ytdlp_files=["video.mp3"],
        gallery_files=["000.mp3", "001.jpg", "002.jpg", "003.jpg"],
        info={"desc": "slide caption", "author": {"uniqueId": "kev"}},
    )
    result = download.download_url(TIKTOK, tmp_path / "download")
    assert [Path(p).name for p in result["image_paths"]] == ["001.jpg", "002.jpg", "003.jpg"]
    assert Path(result["video_path"]).name == "000.mp3"
    assert result["info"]["title"] == "slide caption"
    assert result["info"]["uploader"] == "kev"
    gallery_argv = next(c for c in calls if c[0] == "gallery-dl")
    assert gallery_argv[-2:] == ["--", TIKTOK]


def test_unsupported_url_falls_back_to_gallery_video(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, ytdlp_files=[], gallery_files=["000.mp4"])
    result = download.download_url(TIKTOK, tmp_path / "download")
    assert Path(result["video_path"]).name == "000.mp4"
    assert result["image_paths"] is None


def test_audio_only_mode_keeps_ytdlp_audio_without_gallery(monkeypatch, tmp_path):
    calls = _fake_tools(monkeypatch, ytdlp_files=["video.m4a"], gallery_files=["001.jpg"])
    result = download.download_url(URL, tmp_path / "download", audio_only=True)
    assert Path(result["video_path"]).name == "video.m4a"
    assert not any(c[0] == "gallery-dl" for c in calls)


def test_missing_gallery_dl_keeps_original_failure(monkeypatch, tmp_path):
    _fake_tools(monkeypatch, ytdlp_files=[], gallery_dl=False)
    with pytest.raises(SystemExit, match="yt-dlp did not produce"):
        download.download_url(TIKTOK, tmp_path / "download")


def test_gallery_with_nothing_usable_keeps_ytdlp_audio(monkeypatch, tmp_path):
    # Not a slideshow (e.g. a podcast page): gallery-dl finds nothing, audio stays.
    _fake_tools(monkeypatch, ytdlp_files=["video.mp3"], gallery_files=[])
    result = download.download_url(URL, tmp_path / "download")
    assert Path(result["video_path"]).name == "video.mp3"
    assert "image_paths" not in result
