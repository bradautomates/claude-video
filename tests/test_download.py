"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The request must stay bounded to the configured
language list — which is a small explicit set, never ``all``.
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
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert tokens, f"sub-langs must not be empty, got {langs!r}"
    assert len(tokens) <= 4, f"sub-langs must stay a short list, got {langs!r}"


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


def test_sub_langs_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_SUB_LANGS", "de.*,fr.*")
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    assert _sub_langs(calls[0]) == "de.*,fr.*"


def test_sub_langs_defaults_to_english_and_russian(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_SUB_LANGS", raising=False)
    monkeypatch.setattr(config, "read_env_file", lambda *a, **k: {})
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    assert _sub_langs(calls[0]) == "en.*,ru.*"
