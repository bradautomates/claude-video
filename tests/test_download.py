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


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_english_only(_sub_langs(calls[0]))


def _make_video(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    return video


def test_resolve_local_without_sidecar_has_no_subtitle(tmp_path):
    video = _make_video(tmp_path)
    assert download.resolve_local(str(video))["subtitle_path"] is None


def test_resolve_local_picks_up_exact_sidecar(tmp_path):
    video = _make_video(tmp_path)
    sidecar = tmp_path / "clip.vtt"
    sidecar.write_text("WEBVTT\n", encoding="utf-8")
    assert download.resolve_local(str(video))["subtitle_path"] == str(sidecar.resolve())


def test_resolve_local_picks_up_language_tagged_sidecar(tmp_path):
    video = _make_video(tmp_path)
    sidecar = tmp_path / "clip.en.vtt"
    sidecar.write_text("WEBVTT\n", encoding="utf-8")
    assert download.resolve_local(str(video))["subtitle_path"] == str(sidecar.resolve())


def test_resolve_local_ignores_unrelated_vtt(tmp_path):
    video = _make_video(tmp_path)
    (tmp_path / "other.vtt").write_text("WEBVTT\n", encoding="utf-8")
    assert download.resolve_local(str(video))["subtitle_path"] is None
