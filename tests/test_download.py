"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. The first pass must stay English-only; a second
pass, when made, must be bounded to one language.
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
    # download.py bails via `shutil.which(...)` before it ever builds the argv,
    # so stubbing subprocess alone leaves the test dependent on yt-dlp actually
    # being installed on the host. These cases inspect argv only — no binary is
    # ever executed.
    monkeypatch.setattr(download.shutil, "which", lambda name: f"/usr/bin/{name}")
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
# Caption language selection (issues #144 / #153, PRs #212 / #221 / #234 / #187)
# ---------------------------------------------------------------------------

import json  # noqa: E402


def _capture_with_info(monkeypatch, out_dir: Path, info: dict, vtts_per_call: list[list[str]]):
    """Stub yt-dlp: the first call writes info.json, every call drops the VTTs
    listed for it (as language tags) so the picker has files to rank."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        n = len(calls)
        calls.append(list(cmd))
        out_dir.mkdir(parents=True, exist_ok=True)
        if n == 0:
            (out_dir / "video.info.json").write_text(json.dumps(info), encoding="utf-8")
        for tag in (vtts_per_call[n] if n < len(vtts_per_call) else []):
            (out_dir / f"video.{tag}.vtt").write_text("WEBVTT\n", encoding="utf-8")
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.shutil, "which", lambda name: f"/usr/bin/{name}")
    return calls


def test_english_video_is_one_round_trip(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _capture_with_info(monkeypatch, out, {"language": "en", "automatic_captions": {"en-orig": []}}, [["en"]])
    res = download.fetch_captions(URL, out)
    assert len(calls) == 1
    assert res["subtitle_path"].endswith("video.en.vtt")


def test_non_english_video_fetches_its_own_language(monkeypatch, tmp_path):
    """A German video: YouTube serves a machine-translated `en` track on the
    first pass; the picker must end up on `de-orig`, not on it."""
    out = tmp_path / "download"
    info = {"language": "de", "automatic_captions": {"de-orig": [], "en": []}}
    calls = _capture_with_info(monkeypatch, out, info, [["en"], ["de-orig"]])
    res = download.fetch_captions(URL, out)
    assert len(calls) == 2
    second = _sub_langs(calls[1]).split(",")
    assert "all" not in second and all(t.startswith("de") for t in second), second
    assert res["subtitle_path"].endswith("video.de-orig.vtt")


def test_lang_override_wins_over_detected_language(monkeypatch, tmp_path):
    out = tmp_path / "download"
    info = {"language": "de"}
    calls = _capture_with_info(monkeypatch, out, info, [["en"], ["fr"]])
    res = download.fetch_captions(URL, out, lang="fr")
    assert _sub_langs(calls[1]).startswith("fr,")
    assert res["subtitle_path"].endswith("video.fr.vtt")


def test_human_track_beats_auto_generated(monkeypatch, tmp_path):
    """Both `en` (auto) and `en-US` (uploader) exist; filename sort would pick
    `en`, info.json says `en-US` is the human one."""
    out = tmp_path / "download"
    info = {"language": "en", "subtitles": {"en-US": []}, "automatic_captions": {"en": [], "en-orig": []}}
    _capture_with_info(monkeypatch, out, info, [["en", "en-US", "en-orig"]])
    res = download.fetch_captions(URL, out)
    assert res["subtitle_path"].endswith("video.en-US.vtt")


def test_unknown_language_and_no_english_falls_back_to_orig(monkeypatch, tmp_path):
    out = tmp_path / "download"
    info = {"automatic_captions": {"ko-orig": [], "ko": []}}  # no `language` field
    calls = _capture_with_info(monkeypatch, out, info, [[], ["ko-orig"]])
    res = download.fetch_captions(URL, out)
    assert len(calls) == 2
    assert res["subtitle_path"].endswith("video.ko-orig.vtt")


def test_no_captions_anywhere_returns_none(monkeypatch, tmp_path):
    out = tmp_path / "download"
    calls = _capture_with_info(monkeypatch, out, {"language": "en"}, [[]])
    res = download.fetch_captions(URL, out)
    assert len(calls) == 1
    assert res["subtitle_path"] is None


def test_download_url_reuses_known_language(monkeypatch, tmp_path):
    """fetch_captions already wrote info.json with language=de into out_dir, so
    the video download must request German up front, not English."""
    out = tmp_path / "download"
    out.mkdir(parents=True)
    (out / "video.info.json").write_text(json.dumps({"language": "de"}), encoding="utf-8")
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, out)
    langs = _sub_langs(calls[0]).split(",")
    assert all(t.startswith("de") for t in langs), langs
