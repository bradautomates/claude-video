"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English, so the request must stay
bounded to the English-only pattern.
"""
from __future__ import annotations

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
    assert "--ignore-config" in calls[0]


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))
    assert "--ignore-config" in calls[0]


def test_fetch_captions_clears_stale_artifacts(monkeypatch, tmp_path):
    out_dir = tmp_path / "download"
    out_dir.mkdir()
    stale_sub = out_dir / "video.en.vtt"
    stale_info = out_dir / "video.info.json"
    stale_sub.write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nold\n", encoding="utf-8")
    stale_info.write_text('{"title":"old title"}', encoding="utf-8")

    _capture_argv(monkeypatch)
    result = download.fetch_captions(URL, out_dir)

    assert result["subtitle_path"] is None
    assert result["info"] == {"url": URL}
    assert not stale_sub.exists()
    assert not stale_info.exists()


def test_download_url_clears_stale_video_before_success_check(monkeypatch, tmp_path):
    out_dir = tmp_path / "download"
    out_dir.mkdir()
    stale_video = out_dir / "video.mp4"
    stale_video.write_bytes(b"old")

    _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, out_dir)

    assert not stale_video.exists()
