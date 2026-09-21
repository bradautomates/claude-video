"""Tests for the two 2026-09-17 download-failure fixes.

(a) A bare `HTTP 403: Forbidden` from yt-dlp told the user nothing
    actionable. Real incident: a 27-minute YouTube video failed with exactly
    that error, and the actual cause was a yt-dlp build 2.5 months stale that
    had lost YouTube's current client/signature rotation -- upgrading fixed
    it with no code change, same URL, same format selector. A 403/Forbidden
    in the captured yt-dlp output now gets an actionable explanation (stale
    yt-dlp, not a blocked/region-locked video) plus an upgrade command; every
    other failure keeps the original bare message unchanged.

(b) In that same incident, subtitles downloaded fine (video.en.vtt, 272 KB)
    while only the media stream 403'd, and the run discarded them along with
    the failure. download_url() now hands back a degraded-but-usable result
    when a subtitle exists, and watch.py completes in transcript-only mode
    instead of dying -- but only when a video was genuinely attempted and
    failed, never when the user explicitly chose `--detail transcript`.

Unit-level tests mock ``download.subprocess.run`` directly, matching
test_download.py's existing convention, so they exercise the real
403-detection and message-assembly logic without any network access or
dependency on what yt-dlp happens to print in this environment. Integration
tests drive watch.py end-to-end via subprocess against a synthesized stub
`yt-dlp` executable placed on PATH (see conftest.py's `stub_yt_dlp` fixture)
-- no real caption track or media file is ever vendored into the repo;
everything is generated fresh, in a tmp dir, per test.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

WATCH = SCRIPTS_DIR / "watch.py"

NON_403_OUTPUT = "ERROR: [youtube] abc123: Video unavailable\n"
FORBIDDEN_403_OUTPUT = "ERROR: unable to download video data: HTTP Error 403: Forbidden\n"


class _FakeResult:
    def __init__(self, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _fake_run_factory(returncode: int, stdout: str):
    def fake_run(cmd, *args, **kwargs):
        return _FakeResult(returncode, stdout)
    return fake_run


# ---------------------------------------------------------------------------
# (a) message content -- unit tests directly against _download_failure_message
# ---------------------------------------------------------------------------

def test_non_403_failure_message_is_byte_for_byte_unchanged(tmp_path):
    out_dir = tmp_path / "download"
    msg = download._download_failure_message(NON_403_OUTPUT, 1, out_dir, None)
    assert msg == f"yt-dlp did not produce a video file in {out_dir} (exit 1)"
    assert "Update and retry" not in msg
    assert "\n" not in msg


def test_403_failure_names_staleness_and_upgrade_command(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: "2026.07.04")
    out_dir = tmp_path / "download"
    msg = download._download_failure_message(FORBIDDEN_403_OUTPUT, 1, out_dir, None)
    assert msg.startswith(f"yt-dlp did not produce a video file in {out_dir} (exit 1)")
    assert "2026.07.04" in msg
    assert "Update and retry" in msg
    assert any(
        hint in msg for hint in ("yt-dlp -U", "pipx upgrade yt-dlp", "brew upgrade yt-dlp")
    )
    # Explicitly rules out the wrong diagnosis a bare error code invited.
    assert "not a video that's actually blocked" in msg


def test_bare_forbidden_word_also_triggers_the_upgrade_path(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: None)
    out_dir = tmp_path / "download"
    msg = download._download_failure_message("Forbidden by remote host\n", 1, out_dir, None)
    assert "Update and retry" in msg


def test_403_message_omits_version_parenthetical_when_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: None)
    out_dir = tmp_path / "download"
    msg = download._download_failure_message(FORBIDDEN_403_OUTPUT, 1, out_dir, None)
    assert "(yt-dlp" not in msg
    assert "Update and retry" in msg


def test_403_message_mentions_subtitle_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: None)
    out_dir = tmp_path / "download"
    subtitle = out_dir / "video.en.vtt"
    msg = download._download_failure_message(FORBIDDEN_403_OUTPUT, 1, out_dir, subtitle)
    assert "video.en.vtt" in msg


def test_403_message_omits_subtitle_line_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: None)
    out_dir = tmp_path / "download"
    msg = download._download_failure_message(FORBIDDEN_403_OUTPUT, 1, out_dir, None)
    assert "Captions downloaded fine" not in msg


# ---------------------------------------------------------------------------
# download_url() -- unit tests with subprocess.run mocked (no yt-dlp spawned)
# ---------------------------------------------------------------------------

def test_download_url_non_403_failure_raises_unchanged_message(tmp_path, monkeypatch):
    monkeypatch.setattr(download.subprocess, "run", _fake_run_factory(1, NON_403_OUTPUT))
    out_dir = tmp_path / "download"
    with pytest.raises(SystemExit) as exc:
        download.download_url("https://example.invalid/watch?v=x", out_dir)
    assert str(exc.value) == f"yt-dlp did not produce a video file in {out_dir} (exit 1)"


def test_download_url_403_without_subtitle_still_raises_hard(tmp_path, monkeypatch):
    """Neither video nor subtitle: must still fail hard (no quiet success),
    but with the actionable 403 explanation rather than a bare exit code."""
    monkeypatch.setattr(download.subprocess, "run", _fake_run_factory(1, FORBIDDEN_403_OUTPUT))
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: "2026.07.04")
    out_dir = tmp_path / "download"
    with pytest.raises(SystemExit) as exc:
        download.download_url("https://example.invalid/watch?v=x", out_dir)
    assert "403" in str(exc.value)
    assert "Update and retry" in str(exc.value)


def test_download_url_403_with_subtitle_returns_degraded_result(tmp_path, monkeypatch):
    monkeypatch.setattr(download.subprocess, "run", _fake_run_factory(1, FORBIDDEN_403_OUTPUT))
    monkeypatch.setattr(download, "_yt_dlp_version", lambda: "2026.07.04")
    out_dir = tmp_path / "download"
    out_dir.mkdir(parents=True)
    subtitle = out_dir / "video.en.vtt"
    subtitle.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nHello.\n", encoding="utf-8")

    result = download.download_url("https://example.invalid/watch?v=x", out_dir)

    assert result["video_path"] is None
    assert result["subtitle_path"] == str(subtitle)
    assert result["degraded"] is True
    assert result["downloaded"] is False
    assert "403" in result["failure_message"]
    assert "Update and retry" in result["failure_message"]


def test_download_url_success_path_is_unaffected(tmp_path, monkeypatch):
    """A real success (video file present) must not be marked degraded."""
    monkeypatch.setattr(download.subprocess, "run", _fake_run_factory(0, "done\n"))
    out_dir = tmp_path / "download"
    out_dir.mkdir(parents=True)
    video = out_dir / "video.mp4"
    video.write_bytes(b"\x00")

    result = download.download_url("https://example.invalid/watch?v=x", out_dir)

    assert result["video_path"] == str(video)
    assert result["downloaded"] is True
    assert "degraded" not in result


# ---------------------------------------------------------------------------
# (b) watch.py end to end: transcript-only degradation
# ---------------------------------------------------------------------------

SAMPLE_VTT = (
    "WEBVTT\n\n"
    "00:00:00.000 --> 00:00:02.000\n"
    "Hello and welcome to this talk about degradation paths.\n\n"
    "00:00:05.000 --> 00:00:08.000\n"
    "This line is outside the focused range.\n"
)


def _run_watch(bin_dir: Path, work: Path, *extra_args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    cmd = [
        sys.executable, str(WATCH),
        "https://example.invalid/watch?v=stub-fixture",
        "--no-whisper",
        "--out-dir", str(work),
        *extra_args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)


def test_watch_transcript_only_on_video_missing_subtitle_present(tmp_path, stub_yt_dlp):
    bin_dir = stub_yt_dlp(SAMPLE_VTT)
    work = tmp_path / "work"
    proc = _run_watch(bin_dir, work)

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    report = proc.stdout

    assert "## Transcript" in report
    assert "degradation paths" in report

    # Unmissable in the report BODY, not just stderr.
    assert "NO VIDEO OBTAINED" in report
    assert "## Frames" in report
    frames_section = report.split("## Frames", 1)[1].split("## Transcript", 1)[0]
    assert "no video was downloaded" in frames_section.lower()
    assert "frame_0000" not in report

    # The 403/upgrade explanation from fix (a) is surfaced through.
    assert "403" in report
    assert "Update and retry" in report

    # Also flagged on stderr.
    assert "DEGRADED" in proc.stderr


def test_watch_degraded_report_honours_start_end(tmp_path, stub_yt_dlp):
    bin_dir = stub_yt_dlp(SAMPLE_VTT)
    work = tmp_path / "work-focused"
    proc = _run_watch(bin_dir, work, "--start", "0:00", "--end", "0:03")

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    report = proc.stdout
    assert "degradation paths" in report
    assert "outside the focused range" not in report


def test_watch_both_missing_still_fails_hard(tmp_path, stub_yt_dlp):
    bin_dir = stub_yt_dlp(None)  # no captions available either
    work = tmp_path / "work-total-failure"
    proc = _run_watch(bin_dir, work)

    assert proc.returncode != 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "# watch: video report" not in proc.stdout
    assert "## Transcript" not in proc.stdout
