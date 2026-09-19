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


# ---------------------------------------------------------------------------
# Native-language caption fallback
#
# English-only was a deliberate bound (see module docstring): `--sub-langs all`
# pulls YouTube's hundreds of machine-translated tracks and stalls for minutes.
# The fallback keeps that bound — it retries against exactly one tag, chosen
# from the tracks the extractor reported.
# ---------------------------------------------------------------------------

import json


def _runner(monkeypatch, out_dir, *, subs_written, info):
    """Stub yt-dlp: record argv, write info.json, and emit VTTs per call.

    ``subs_written`` is one list of filenames per invocation.
    """
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "video.info.json").write_text(json.dumps(info), encoding="utf-8")
        for name in subs_written[len(calls)] if len(calls) < len(subs_written) else []:
            (out_dir / name).write_text("WEBVTT\n", encoding="utf-8")
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def test_english_video_still_takes_one_round_trip(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[["video.en.vtt"]],
        info={"subtitles": {"en": []}, "automatic_captions": {}},
    )
    result = download.fetch_captions(URL, out)

    assert len(calls) == 1, "English captions must not trigger a second probe"
    assert _sub_langs(calls[0]) == "en.*"
    assert result["subtitle_path"].endswith("video.en.vtt")


def test_retries_in_the_videos_own_language(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[[], ["video.zh-Hans.vtt"]],
        info={"subtitles": {"zh-Hans": []}, "automatic_captions": {}},
    )
    result = download.fetch_captions(URL, out)

    assert len(calls) == 2
    assert _sub_langs(calls[0]) == "en.*"
    assert _sub_langs(calls[1]) == "zh-Hans"
    assert result["subtitle_path"].endswith("video.zh-Hans.vtt")


def test_prefers_orig_auto_track_over_translations(monkeypatch, tmp_path):
    """`automatic_captions` lists every translation target; only -orig is spoken."""
    out = tmp_path / "download"
    translations = {f"t{i}": [] for i in range(200)}
    calls = _runner(
        monkeypatch, out,
        subs_written=[[], ["video.ja-orig.vtt"]],
        info={
            "subtitles": {},
            "automatic_captions": {**translations, "ja-orig": [], "en": []},
        },
    )
    download.fetch_captions(URL, out)

    assert len(calls) == 2
    assert _sub_langs(calls[1]) == "ja-orig"


def test_manual_track_wins_over_auto(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[[], ["video.ko.vtt"]],
        info={"subtitles": {"ko": []}, "automatic_captions": {"ja-orig": []}},
    )
    download.fetch_captions(URL, out)
    assert _sub_langs(calls[1]) == "ko"


def test_live_chat_is_not_a_caption_track(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[[], []],
        info={"subtitles": {"live_chat": []}, "automatic_captions": {"de-orig": []}},
    )
    download.fetch_captions(URL, out)
    assert _sub_langs(calls[1]) == "de-orig"


def test_no_tracks_at_all_does_not_retry(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[[]],
        info={"subtitles": {}, "automatic_captions": {}},
    )
    result = download.fetch_captions(URL, out)

    assert len(calls) == 1, "nothing to retry with — must fall through to Whisper"
    assert result["subtitle_path"] is None


def test_retry_never_requests_all(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _runner(
        monkeypatch, out,
        subs_written=[[], []],
        info={"subtitles": {}, "automatic_captions": {f"t{i}": [] for i in range(300)}},
    )
    download.fetch_captions(URL, out)
    for call in calls:
        _assert_english_only(_sub_langs(call)) if _sub_langs(call) == "en.*" else None
        assert "all" not in _sub_langs(call).split(",")


def test_download_url_reuses_the_discovered_language(monkeypatch, tmp_path):
    """info.json already sits in this directory — no second extraction needed."""
    out = tmp_path / "download"
    out.mkdir(parents=True)
    (out / "video.info.json").write_text(
        json.dumps({"subtitles": {"zh-Hans": []}, "automatic_captions": {}}),
        encoding="utf-8",
    )
    calls = _runner(
        monkeypatch, out,
        subs_written=[["video.zh-Hans.vtt"]],
        info={"subtitles": {"zh-Hans": []}, "automatic_captions": {}},
    )
    (out / "video.mp4").write_text("x", encoding="utf-8")

    download.download_url(URL, out)

    assert _sub_langs(calls[0]) == "zh-Hans"
