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


def _cookie_browser(argv: list[str]) -> str | None:
    if "--cookies-from-browser" not in argv:
        return None
    return argv[argv.index("--cookies-from-browser") + 1]


def test_download_url_first_attempt_has_no_cookies(monkeypatch, tmp_path):
    """The bare, fastest attempt must not pull browser cookies."""
    calls = _capture_argv(monkeypatch)
    monkeypatch.setattr(download.config, "cookies_from_browser", lambda: None)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    assert _cookie_browser(calls[0]) is None


def test_download_url_probes_browsers_on_failure(monkeypatch, tmp_path):
    """When the bare download yields no video, retry once per fallback browser."""
    calls = _capture_argv(monkeypatch)
    monkeypatch.setattr(download.config, "cookies_from_browser", lambda: None)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    retried = [_cookie_browser(c) for c in calls if _cookie_browser(c) is not None]
    assert retried == list(download.COOKIE_BROWSER_FALLBACKS)


def test_download_url_respects_forced_browser(monkeypatch, tmp_path):
    """An explicit WATCH_COOKIES_FROM_BROWSER short-circuits the probe."""
    calls = _capture_argv(monkeypatch)
    monkeypatch.setattr(download.config, "cookies_from_browser", lambda: "firefox")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    retried = [_cookie_browser(c) for c in calls if _cookie_browser(c) is not None]
    assert retried == ["firefox"]


def test_download_url_no_cookie_retry_on_success(monkeypatch, tmp_path):
    """A successful bare download must not trigger any cookie retry."""
    calls = _capture_argv(monkeypatch)
    monkeypatch.setattr(download, "_pick_video", lambda out_dir: tmp_path / "video.mp4")
    result = download.download_url(URL, tmp_path / "download")
    assert result["downloaded"] is True
    assert len(calls) == 1
    assert _cookie_browser(calls[0]) is None
