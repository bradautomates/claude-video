"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The request must stay bounded to the languages the
caller actually asked for.
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


def _assert_bounded(langs: str, expected: list[str]) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    expanded = [v for code in expected for v in (code, f"{code}-orig")]
    assert tokens == expanded, (
        f"sub-langs must request exactly the caller's languages, got {langs!r}"
    )
    # A `code.*` glob would also pull YouTube's auto-translated pairs, which is
    # the request explosion this guard exists to prevent.
    assert not any("*" in t for t in tokens), f"sub-langs must not use globs, got {langs!r}"


def test_fetch_captions_defaults_to_en_pt(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls[0]), ["en", "pt"])


def test_download_url_defaults_to_en_pt(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded(_sub_langs(calls[0]), ["en", "pt"])


def test_explicit_lang_is_honoured(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download", lang="es")
    _assert_bounded(_sub_langs(calls[0]), ["es"])


def test_lang_list_normalises_regions_and_dedupes():
    assert download._lang_list("en,pt") == ["en", "pt"]
    assert download._lang_list("pt-BR, EN") == ["pt", "en"]
    assert download._lang_list("en,en-US") == ["en"]
    assert download._lang_list("") == ["en"]


def test_pick_subtitle_follows_priority_order(tmp_path):
    for name in ("video.en.vtt", "video.pt.vtt"):
        (tmp_path / name).touch()
    assert download._pick_subtitle(tmp_path, "en,pt").name == "video.en.vtt"
    assert download._pick_subtitle(tmp_path, "pt,en").name == "video.pt.vtt"


def test_pick_subtitle_skips_languages_that_produced_no_file(tmp_path):
    # yt-dlp commonly fails one variant (429, no such track) while another
    # succeeds; the first *available* requested language must win.
    (tmp_path / "video.pt-orig.vtt").touch()
    assert download._pick_subtitle(tmp_path, "en,pt").name == "video.pt-orig.vtt"


def test_pick_subtitle_prefers_original_over_a_translation(tmp_path):
    # A Portuguese video that also publishes an English auto-translation. Under
    # the default `en,pt` the priority order alone would hand back the English
    # translation; the `-orig` track is the language actually spoken, so it wins.
    for name in ("video.en.vtt", "video.pt.vtt", "video.pt-orig.vtt"):
        (tmp_path / name).touch()
    assert download._pick_subtitle(tmp_path, "en,pt").name == "video.pt-orig.vtt"


def test_pick_subtitle_prefers_requested_original_when_several_exist(tmp_path):
    for name in ("video.en-orig.vtt", "video.pt-orig.vtt"):
        (tmp_path / name).touch()
    assert download._pick_subtitle(tmp_path, "en,pt").name == "video.en-orig.vtt"
    assert download._pick_subtitle(tmp_path, "pt,en").name == "video.pt-orig.vtt"


def test_pick_subtitle_falls_back_to_any_track(tmp_path):
    (tmp_path / "video.de.vtt").touch()
    assert download._pick_subtitle(tmp_path, "en,pt").name == "video.de.vtt"


def test_pick_subtitle_returns_none_when_empty(tmp_path):
    assert download._pick_subtitle(tmp_path, "en,pt") is None
