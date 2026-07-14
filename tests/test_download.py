"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. A cheap metadata-only probe (_probe_langs) picks
one exact target language per video — manual subtitle in the video's own
language first, then any manual subtitle, then the automatic "-orig" track —
and only that single language is ever requested. When the probe can't tell
us anything (non-YouTube site, network hiccup), we fall back to the bounded
"en.*,.*-orig" pattern. Never "all".
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


def _sub_langs_from_calls(calls: list[list[str]]) -> str:
    """Find the yt-dlp call that actually requests subtitles.

    A probe call (`-j`, no --sub-langs) always runs first now — see
    _probe_langs in download.py — so the real subtitle-fetching call isn't
    necessarily calls[0] anymore.
    """
    for argv in calls:
        if "--sub-langs" in argv:
            return _sub_langs(argv)
    raise AssertionError(f"no call requested --sub-langs: {calls!r}")


def _assert_bounded_langs(langs: str) -> None:
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    assert all(
        t.startswith("en") or t == ".*-orig" for t in tokens
    ), f"sub-langs must be English or the single original-language track, got {langs!r}"


def test_fetch_captions_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded_langs(_sub_langs_from_calls(calls))


def test_download_url_requests_bounded_langs(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    # _pick_video returns None with no real file, which raises SystemExit after
    # the yt-dlp argv is already built — that's all we need to inspect.
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded_langs(_sub_langs_from_calls(calls))


def test_pick_subtitle_prefers_original_language_over_english(tmp_path):
    # A non-English video: yt-dlp writes both the machine-translated English
    # track and the "-orig" track for the language actually spoken (Italian
    # here). The original must win — it's the real text, not a translation.
    (tmp_path / "video.en.vtt").write_text("WEBVTT")
    (tmp_path / "video.it-orig.vtt").write_text("WEBVTT")
    assert download._pick_subtitle(tmp_path).name == "video.it-orig.vtt"


def test_pick_subtitle_falls_back_to_english_without_orig(tmp_path):
    # No "-orig" track at all (e.g. a manual English subtitle only) — English
    # is still a reasonable fallback.
    (tmp_path / "video.en.vtt").write_text("WEBVTT")
    (tmp_path / "video.fr.vtt").write_text("WEBVTT")
    assert download._pick_subtitle(tmp_path).name == "video.en.vtt"


def test_choose_target_lang_prefers_manual_subtitle_in_declared_language():
    # An Italian video with a hand-written Italian subtitle: manual + matches
    # the declared language — the best possible case, must win outright.
    probe = {"language": "it", "manual_langs": {"it"}, "auto_langs": {"it", "it-orig"}}
    assert download._choose_target_lang(probe) == "it"


def test_choose_target_lang_falls_back_to_any_manual_subtitle():
    # Declared language has no matching manual subtitle, but some manual
    # subtitle exists (e.g. uploader only added Spanish by hand) — still
    # beats an auto-generated transcript.
    probe = {"language": "it", "manual_langs": {"es"}, "auto_langs": {"it", "it-orig"}}
    assert download._choose_target_lang(probe) == "es"


def test_choose_target_lang_matches_declared_language_by_prefix():
    # Seen live on a real video: an English video whose manual English
    # subtitle (from a third-party localization vendor) is keyed
    # "en-<vendor-id>", not a plain "en". An exact-match-only check misses
    # this and falls through to "any manual subtitle", which then picks
    # whichever language sorts first alphabetically (here "ar") — a random
    # wrong language instead of the real English track. Must match by
    # prefix, the same way automatic captions' "en-US"/"en-GB" do.
    probe = {
        "language": "en",
        "manual_langs": {"ar", "de", "en-ehkg1hFWq8A", "es", "fr"},
        "auto_langs": {"en", "en-orig"},
    }
    assert download._choose_target_lang(probe) == "en-ehkg1hFWq8A"


def test_choose_target_lang_uses_original_auto_caption_without_manual():
    # No manual subtitles at all: fall back to the automatic transcript in
    # the video's own language, not a translation.
    probe = {"language": "it", "manual_langs": set(), "auto_langs": {"it", "it-orig", "en"}}
    assert download._choose_target_lang(probe) == "it-orig"


def test_choose_target_lang_returns_none_when_nothing_available():
    probe = {"language": "it", "manual_langs": set(), "auto_langs": set()}
    assert download._choose_target_lang(probe) is None
    assert download._choose_target_lang(None) is None


def test_sub_langs_for_targets_exact_language_from_probe(monkeypatch):
    # End-to-end: a successful probe must produce an *exact*, anchored
    # pattern for the chosen language — not a broad regex that could also
    # match unrelated translated tracks.
    monkeypatch.setattr(
        download, "_probe_langs",
        lambda url: {"language": "it", "manual_langs": {"it"}, "auto_langs": {"it", "it-orig"}},
    )
    assert download._sub_langs_for(URL) == "^it$"


def test_sub_langs_for_falls_back_when_probe_fails(monkeypatch):
    monkeypatch.setattr(download, "_probe_langs", lambda url: None)
    _assert_bounded_langs(download._sub_langs_for(URL))
