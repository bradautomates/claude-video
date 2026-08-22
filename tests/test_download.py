"""yt-dlp argv construction and subtitle selection for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The request must stay bounded to a handful of
tracks — the video's own language plus English.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import config  # noqa: E402
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


def _assert_bounded(langs: str) -> None:
    """The selector must stay small — never YouTube's full translation set."""
    tokens = [t for t in langs.split(",") if t]
    assert tokens, f"sub-langs must not be empty, got {langs!r}"
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert len(tokens) <= 4, f"sub-langs must stay bounded, got {langs!r}"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Keep a developer's own WATCH_SUBLANGS out of these assertions."""
    monkeypatch.delenv("WATCH_SUBLANGS", raising=False)
    monkeypatch.setattr(config, "read_env_file", lambda *a, **k: {})


def test_fetch_captions_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls[0]))


def test_download_url_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls[0]))


def test_default_requests_original_language_track(monkeypatch, tmp_path):
    """Asking for English only makes YouTube 429 on non-English videos."""
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    langs = _sub_langs(calls[0])
    assert "-orig" in langs, f"must request the original-language track, got {langs!r}"
    assert "en" in langs, f"must still request English, got {langs!r}"


def test_sub_langs_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_SUBLANGS", "es.*,en.*")
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    assert _sub_langs(calls[0]) == "es.*,en.*"


def test_blank_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WATCH_SUBLANGS", "   ")
    assert config.sub_langs() == config.DEFAULT_SUBLANGS


def _touch(dir_path: Path, *names: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dir_path / name).write_text("WEBVTT\n", encoding="utf-8")


def test_pick_subtitle_prefers_original_over_translation(tmp_path):
    """A same-run "en" track on a Portuguese video is a machine translation."""
    _touch(tmp_path, "video.en.vtt", "video.pt-orig.vtt")
    assert download._pick_subtitle(tmp_path).name == "video.pt-orig.vtt"


def test_pick_subtitle_english_video_unchanged(tmp_path):
    """English stays preferred over an unrelated track, as before."""
    _touch(tmp_path, "video.de.vtt", "video.en.vtt")
    assert download._pick_subtitle(tmp_path).name == "video.en.vtt"


def test_pick_subtitle_falls_back_to_any_track(tmp_path):
    _touch(tmp_path, "video.de.vtt")
    assert download._pick_subtitle(tmp_path).name == "video.de.vtt"


def test_pick_subtitle_none_when_empty(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    assert download._pick_subtitle(tmp_path) is None
