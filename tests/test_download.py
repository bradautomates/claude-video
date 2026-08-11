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


# --- Per-source cache isolation -------------------------------------------
#
# Regression guard: reusing one --out-dir across two different URLs used to hand
# the second run the first video's media. yt-dlp skips the download when a
# video.mp4 is already there ("has already been downloaded") but still rewrites
# info.json and the subtitles, so the report reads correct while every extracted
# frame belongs to the previous video. Each source therefore gets its own cache
# directory, and the media is cross-checked against the reported duration.

OTHER_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_same_video_different_spellings_share_one_key():
    variants = [
        "https://www.youtube.com/watch?v=rlOpbu3Enkw",
        "https://www.youtube.com/watch?v=rlOpbu3Enkw&t=90s",
        "https://youtu.be/rlOpbu3Enkw?si=aBcDeF",
        "https://www.youtube.com/shorts/rlOpbu3Enkw",
        "https://m.youtube.com/watch?v=rlOpbu3Enkw&feature=share",
    ]
    keys = {download.source_key(v) for v in variants}
    assert len(keys) == 1, f"same video mapped to {len(keys)} cache entries: {keys}"


def test_different_videos_get_different_keys():
    assert download.source_key(URL) != download.source_key(OTHER_URL)


def test_unknown_host_keeps_meaningful_query_params():
    """A false merge would be a correctness bug; a redundant download is not."""
    a = "https://vids.example.com/player?clip=111"
    b = "https://vids.example.com/player?clip=222"
    assert download.source_key(a) != download.source_key(b)
    # Volatile params must still not split one source into two entries.
    assert download.source_key(a) == download.source_key(a + "&t=30")


def test_two_urls_do_not_share_a_cache_directory(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    shared = tmp_path / "download"

    download.fetch_captions(URL, shared)
    download.fetch_captions(OTHER_URL, shared)

    def out_template(argv):
        return argv[argv.index("-o") + 1]

    first, second = out_template(calls[0]), out_template(calls[1])
    assert first != second, "two different URLs wrote to the same output template"
    assert download.source_key(URL) in first
    assert download.source_key(OTHER_URL) in second


def test_captions_and_download_share_one_directory_per_source(monkeypatch, tmp_path):
    """Otherwise the captions pass and the video pass would disagree on location."""
    calls = _capture_argv(monkeypatch)
    shared = tmp_path / "download"

    download.fetch_captions(URL, shared)
    with pytest.raises(SystemExit):
        download.download_url(URL, shared)

    templates = {argv[argv.index("-o") + 1] for argv in calls}
    assert len(templates) == 1, f"captions and download disagreed on cache dir: {templates}"


def test_fresh_discards_the_cached_directory(monkeypatch, tmp_path):
    _capture_argv(monkeypatch)
    shared = tmp_path / "download"
    cache = shared / download.source_key(URL)
    cache.mkdir(parents=True)
    stale = cache / "video.mp4"
    stale.write_bytes(b"not really a video")

    download.fetch_captions(URL, shared, fresh=True)
    assert not stale.exists(), "--fresh left the cached media in place"


def test_duration_mismatch_flags_a_foreign_media_file(tmp_path, static_clip):
    """The guard that catches a leftover video from a different URL."""
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    video = out_dir / "video.mp4"
    video.write_bytes(static_clip.read_bytes())

    # Source claims a 40-minute video; the file on disk is a few seconds.
    (out_dir / "video.info.json").write_text('{"duration": 2400}', encoding="utf-8")
    assert download._duration_mismatch(out_dir, video) is not None


def test_duration_mismatch_accepts_the_matching_file(tmp_path, static_clip):
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    video = out_dir / "video.mp4"
    video.write_bytes(static_clip.read_bytes())

    actual = download._probe_duration(video)
    assert actual is not None, "ffprobe could not read the synthesized clip"
    (out_dir / "video.info.json").write_text(
        f'{{"duration": {actual:.3f}}}', encoding="utf-8"
    )
    assert download._duration_mismatch(out_dir, video) is None


def test_duration_mismatch_is_silent_without_info_json(tmp_path, static_clip):
    out_dir = tmp_path / "cache"
    out_dir.mkdir()
    video = out_dir / "video.mp4"
    video.write_bytes(static_clip.read_bytes())
    assert download._duration_mismatch(out_dir, video) is None
