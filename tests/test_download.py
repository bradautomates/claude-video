"""yt-dlp argv construction and media-response validation for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The default must stay bounded to Chinese and
English, with caller-provided language priorities validated before subprocess
execution.
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


def _capture_argv(
    monkeypatch: pytest.MonkeyPatch,
    node_path: str | None = None,
    impersonate_targets: str = "",
) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv."""
    calls: list[list[str]] = []

    class _Result:
        def __init__(self, stdout: str = ""):
            self.returncode = 0
            self.stdout = stdout
            self.stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if "--list-impersonate-targets" in cmd:
            return _Result(impersonate_targets)
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(
        download.shutil,
        "which",
        lambda name: "/usr/bin/yt-dlp" if name == "yt-dlp" else node_path,
    )
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _download_argv(calls: list[list[str]]) -> list[str]:
    return next(argv for argv in calls if "--list-impersonate-targets" not in argv)


def _assert_bounded_default(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert tokens == ["zh.*", "en.*"]


def test_fetch_captions_requests_bounded_chinese_then_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded_default(_sub_langs(_download_argv(calls)))


def test_download_url_requests_bounded_chinese_then_english(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded_default(_sub_langs(_download_argv(calls)))


def test_custom_caption_priority_reaches_ytdlp(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", sub_langs="ja.*,en.*")
    assert _sub_langs(_download_argv(calls)) == "ja.*,en.*"


def test_download_uses_node_runtime_when_available(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, node_path="/usr/bin/node")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")

    argv = _download_argv(calls)
    idx = argv.index("--js-runtimes")
    assert argv[idx + 1] == "node:/usr/bin/node"


def test_download_uses_chrome_impersonation_only_when_available(monkeypatch, tmp_path):
    calls = _capture_argv(
        monkeypatch,
        impersonate_targets="Client          OS           Source\nChrome-133      Macos-15     curl_cffi\n",
    )
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")

    argv = _download_argv(calls)
    idx = argv.index("--impersonate")
    assert argv[idx + 1] == "chrome"


def test_download_rejects_html_saved_as_video(monkeypatch, tmp_path):
    _capture_argv(monkeypatch)
    out_dir = tmp_path / "download"
    out_dir.mkdir()
    (out_dir / "video.mp4").write_bytes(
        b"<html><head><title>Site Unavailable</title></head></html>"
    )

    with pytest.raises(SystemExit, match="HTML error page"):
        download.download_url(URL, out_dir)


def test_rejects_sub_langs_that_look_like_options():
    with pytest.raises(SystemExit, match="--sub-langs"):
        download.normalize_sub_langs("--config-location")


def test_pick_subtitle_uses_requested_priority(tmp_path):
    (tmp_path / "video.en.vtt").touch()
    (tmp_path / "video.zh-Hans.vtt").touch()
    assert download._pick_subtitle(tmp_path, "zh.*,en.*").name == "video.zh-Hans.vtt"
    assert download._pick_subtitle(tmp_path, "en.*,zh.*").name == "video.en.vtt"
