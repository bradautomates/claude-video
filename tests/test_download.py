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


def _stub_runs(monkeypatch, tmp_path, returncodes, write_video_on_attempt=None):
    """Stub subprocess.run with a scripted sequence of yt-dlp exit codes.

    Optionally materialise a video file on a given (1-based) attempt so
    _pick_video starts succeeding from that point on, as a real retry would.
    """
    out_dir = tmp_path / "download"
    attempts: list[int] = []
    slept: list[float] = []

    class _Result:
        def __init__(self, rc):
            self.returncode = rc
            self.stdout = ""
            self.stderr = ""

    def fake_run(cmd, *args, **kwargs):
        n = len(attempts) + 1
        attempts.append(n)
        if write_video_on_attempt is not None and n >= write_video_on_attempt:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "video.mp4").write_bytes(b"\x00")
        rc = returncodes[min(n - 1, len(returncodes) - 1)]
        return _Result(rc)

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.time, "sleep", lambda s: slept.append(s))
    return out_dir, attempts, slept


def test_download_url_retries_transient_403(monkeypatch, tmp_path):
    """A non-zero exit with no video file is retried, and a later success is used."""
    out_dir, attempts, slept = _stub_runs(
        monkeypatch, tmp_path, returncodes=[1, 0], write_video_on_attempt=2
    )
    result = download.download_url(URL, out_dir)
    assert len(attempts) == 2, "should have retried once after the failed attempt"
    assert slept == [download.DOWNLOAD_BACKOFF_SECONDS[0]], "should back off before retrying"
    assert result["video_path"].endswith("video.mp4")


def test_download_url_gives_up_after_all_attempts(monkeypatch, tmp_path):
    """Persistent failure still raises, and does not retry forever."""
    out_dir, attempts, slept = _stub_runs(monkeypatch, tmp_path, returncodes=[1])
    with pytest.raises(SystemExit):
        download.download_url(URL, out_dir)
    expected = len(download.DOWNLOAD_BACKOFF_SECONDS) + 1
    assert len(attempts) == expected, f"should try exactly {expected} times"
    assert slept == list(download.DOWNLOAD_BACKOFF_SECONDS)


def test_download_url_does_not_retry_clean_exit(monkeypatch, tmp_path):
    """A clean exit with no video is a real error (bad URL/format) — retrying stalls."""
    out_dir, attempts, slept = _stub_runs(monkeypatch, tmp_path, returncodes=[0])
    with pytest.raises(SystemExit):
        download.download_url(URL, out_dir)
    assert len(attempts) == 1, "a clean exit must not be retried"
    assert slept == []
