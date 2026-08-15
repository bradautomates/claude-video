"""yt-dlp argv construction for download.py.

Regression guard: caption requests select one source-language track when
metadata is available and otherwise use a bounded original/English fallback.
They must never request every auto-translated caption track.
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
    languages: list[str] | None = None,
) -> list[list[str]]:
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
    monkeypatch.setattr(
        download,
        "_probe_caption_languages",
        lambda url: ["ko"] if languages is None else list(languages),
    )
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_bounded(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert "en.*" not in tokens, f"sub-langs must not request translated English variants: {langs!r}"
    assert len(tokens) <= len(download.FALLBACK_SUB_LANGS)


def test_fetch_captions_requests_exact_source_language(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    result = download.fetch_captions(URL, tmp_path / "download")
    assert _sub_langs(calls[0]) == "^ko$"
    assert result["caption_languages"] == ["ko"]


def test_download_url_requests_exact_source_language(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    assert _sub_langs(calls[0]) == "^ko$"


def test_probe_failure_uses_bounded_fallback(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch, languages=[])
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls[0]))


def test_choose_prefers_declared_manual_caption():
    info = {
        "language": "ko",
        "subtitles": {"en": {}, "ko": {}},
        "automatic_captions": {"ko-orig": {}},
    }
    assert download._choose_caption_languages(info) == ["ko"]


def test_choose_prefers_original_auto_caption_over_translation():
    info = {
        "language": "ko",
        "subtitles": {},
        "automatic_captions": {"en": {}, "ko": {}, "ko-orig": {}},
    }
    assert download._choose_caption_languages(info) == ["ko-orig"]


def test_choose_uses_orig_marker_when_declared_language_is_missing():
    info = {
        "subtitles": {},
        "automatic_captions": {"en": {}, "pt-orig": {}},
    }
    assert download._choose_caption_languages(info) == ["pt-orig"]


def test_pick_subtitle_prefers_selected_language(tmp_path):
    for name in ("video.en.vtt", "video.ko-orig.vtt"):
        (tmp_path / name).write_text("WEBVTT\n", encoding="utf-8")
    assert download._pick_subtitle(tmp_path, ["ko-orig"]).name == "video.ko-orig.vtt"


def test_probe_parses_yt_dlp_metadata(monkeypatch):
    class Result:
        returncode = 0
        stdout = '{"language":"ko","automatic_captions":{"ko-orig":{},"en":{}}}'
        stderr = ""

    monkeypatch.setattr(download.subprocess, "run", lambda *args, **kwargs: Result())
    assert download._probe_caption_languages(URL) == ["ko-orig"]
