"""yt-dlp argv construction for download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English, so the request must stay
bounded to the English-only pattern.
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
# Download cache
# ---------------------------------------------------------------------------

import json


class TestCacheKey:
    def test_audio_and_video_get_separate_entries(self, tmp_path):
        """A transcript run fetches audio only; reusing it for frames would
        hand ffmpeg a file with no video stream."""
        video = download.cache_dir_for(URL, False, root=tmp_path)
        audio = download.cache_dir_for(URL, True, root=tmp_path)
        assert video != audio

    def test_same_request_is_stable(self, tmp_path):
        assert download.cache_dir_for(URL, False, root=tmp_path) == \
            download.cache_dir_for(URL, False, root=tmp_path)

    def test_different_urls_do_not_collide(self, tmp_path):
        other = "https://www.youtube.com/watch?v=something-else"
        assert download.cache_dir_for(URL, False, root=tmp_path) != \
            download.cache_dir_for(other, False, root=tmp_path)


class TestCachedDownload:
    def _populate(self, d: Path, size: int = 10):
        d.mkdir(parents=True, exist_ok=True)
        (d / "video.mp4").write_bytes(b"x" * size)
        (d / "video.info.json").write_text(json.dumps({"title": "t"}), encoding="utf-8")

    def test_hit_skips_yt_dlp_entirely(self, monkeypatch, tmp_path):
        out = tmp_path / "entry"
        self._populate(out)
        calls = _capture_argv(monkeypatch)

        result = download.download_url(URL, out, use_cache=True)

        assert calls == [], "a cache hit must not shell out to yt-dlp"
        assert result["cached"] is True
        assert result["video_path"].endswith("video.mp4")

    def test_miss_downloads_normally(self, monkeypatch, tmp_path):
        out = tmp_path / "entry"
        calls = _capture_argv(monkeypatch)
        monkeypatch.setattr(download, "_pick_video", lambda d: None)

        with pytest.raises(SystemExit):
            download.download_url(URL, out, use_cache=True)
        assert len(calls) == 1

    def test_zero_byte_file_is_a_miss(self, monkeypatch, tmp_path):
        """An interrupted run can leave a truncated file behind."""
        out = tmp_path / "entry"
        out.mkdir(parents=True)
        (out / "video.mp4").write_bytes(b"")
        assert download._cached_download(out) is None

    def test_cache_disabled_always_downloads(self, monkeypatch, tmp_path):
        out = tmp_path / "entry"
        self._populate(out)
        calls = _capture_argv(monkeypatch)

        download.download_url(URL, out, use_cache=False)
        assert len(calls) == 1, "use_cache=False must re-fetch"


class TestPruneCache:
    def _entry(self, root: Path, name: str, size: int, mtime: float):
        d = root / name
        d.mkdir(parents=True)
        (d / "video.mp4").write_bytes(b"x" * size)
        os.utime(d, (mtime, mtime))
        return d

    def test_under_limit_evicts_nothing(self, tmp_path):
        self._entry(tmp_path, "a", 100, 1000)
        assert download.prune_cache(limit=1000, root=tmp_path) == 0
        assert (tmp_path / "a").exists()

    def test_evicts_least_recently_used_first(self, tmp_path):
        self._entry(tmp_path, "old", 100, 1000)
        self._entry(tmp_path, "mid", 100, 2000)
        self._entry(tmp_path, "new", 100, 3000)

        download.prune_cache(limit=250, root=tmp_path)

        assert not (tmp_path / "old").exists()
        assert (tmp_path / "mid").exists()
        assert (tmp_path / "new").exists()

    def test_keep_is_never_evicted(self, tmp_path):
        keep = self._entry(tmp_path, "current", 500, 1000)  # oldest AND largest
        self._entry(tmp_path, "other", 100, 3000)

        download.prune_cache(limit=100, root=tmp_path, keep=keep)

        assert keep.exists(), "the entry this run is about to use must survive"

    def test_missing_root_is_not_an_error(self, tmp_path):
        assert download.prune_cache(root=tmp_path / "nope") == 0
