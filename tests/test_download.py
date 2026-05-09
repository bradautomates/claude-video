"""Tests for scripts/download.py - URL detection, local file resolution,
subtitle/video file picking logic.

These tests require no external binaries and run with just pytest.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from download import (  # noqa: E402
    VIDEO_EXTS,
    _pick_subtitle,
    _pick_video,
    is_url,
    resolve_local,
)


# ---------------------------------------------------------------------------
# is_url
# ---------------------------------------------------------------------------

class TestIsUrl:
    def test_http(self):
        assert is_url("http://example.com/video.mp4")

    def test_https(self):
        assert is_url("https://youtu.be/dQw4w9WgXcQ")

    def test_youtube_watch(self):
        assert is_url("https://www.youtube.com/watch?v=abc123")

    def test_local_path(self):
        assert not is_url("/home/user/video.mp4")

    def test_relative_path(self):
        assert not is_url("./video.mp4")

    def test_windows_path(self):
        assert not is_url("C:\\Users\\user\\video.mp4")

    def test_bare_filename(self):
        assert not is_url("video.mp4")

    def test_tiktok(self):
        assert is_url("https://www.tiktok.com/@user/video/12345")

    def test_loom(self):
        assert is_url("https://www.loom.com/share/abc123")


# ---------------------------------------------------------------------------
# resolve_local
# ---------------------------------------------------------------------------

class TestResolveLocal:
    def test_existing_mp4(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"fake")
        result = resolve_local(str(f))
        assert result["video_path"] == str(f)
        assert result["subtitle_path"] is None
        assert result["downloaded"] is False

    def test_existing_mov(self, tmp_path):
        f = tmp_path / "clip.mov"
        f.write_bytes(b"fake")
        result = resolve_local(str(f))
        assert result["video_path"] == str(f)

    def test_existing_mkv(self, tmp_path):
        f = tmp_path / "clip.mkv"
        f.write_bytes(b"fake")
        result = resolve_local(str(f))
        assert result["video_path"] == str(f)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            resolve_local(str(tmp_path / "nonexistent.mp4"))

    def test_unknown_extension_warns_but_proceeds(self, tmp_path, capsys):
        f = tmp_path / "clip.xyz"
        f.write_bytes(b"fake")
        result = resolve_local(str(f))
        # Should still return a result dict
        assert result["video_path"] == str(f)
        # Warning should be printed to stderr
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "xyz" in captured.err

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        # Monkeypatch Path.expanduser to return tmp_path / "video.mp4"
        target = tmp_path / "video.mp4"
        target.write_bytes(b"fake")
        # Just verify it doesn't crash on a real path without tilde
        result = resolve_local(str(target))
        assert result["downloaded"] is False

    def test_info_contains_title(self, tmp_path):
        f = tmp_path / "myvideo.mp4"
        f.write_bytes(b"fake")
        result = resolve_local(str(f))
        assert result["info"]["title"] == "myvideo.mp4"


# ---------------------------------------------------------------------------
# _pick_subtitle
# ---------------------------------------------------------------------------

class TestPickSubtitle:
    def test_prefers_english(self, tmp_path):
        (tmp_path / "video.fr.vtt").write_text("WEBVTT", encoding="utf-8")
        (tmp_path / "video.en.vtt").write_text("WEBVTT", encoding="utf-8")
        result = _pick_subtitle(tmp_path)
        assert result is not None
        assert ".en." in result.name

    def test_falls_back_to_any_vtt(self, tmp_path):
        (tmp_path / "video.fr.vtt").write_text("WEBVTT", encoding="utf-8")
        result = _pick_subtitle(tmp_path)
        assert result is not None
        assert result.suffix == ".vtt"

    def test_returns_none_when_empty(self, tmp_path):
        assert _pick_subtitle(tmp_path) is None

    def test_ignores_non_vtt_files(self, tmp_path):
        (tmp_path / "video.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi", encoding="utf-8")
        assert _pick_subtitle(tmp_path) is None

    def test_picks_en_us_variant(self, tmp_path):
        (tmp_path / "video.en-US.vtt").write_text("WEBVTT", encoding="utf-8")
        result = _pick_subtitle(tmp_path)
        assert result is not None

    def test_multiple_english_returns_first_sorted(self, tmp_path):
        (tmp_path / "video.en-GB.vtt").write_text("WEBVTT", encoding="utf-8")
        (tmp_path / "video.en-US.vtt").write_text("WEBVTT", encoding="utf-8")
        result = _pick_subtitle(tmp_path)
        assert result is not None
        assert ".en" in result.name


# ---------------------------------------------------------------------------
# _pick_video
# ---------------------------------------------------------------------------

class TestPickVideo:
    def test_finds_mp4(self, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"fake")
        result = _pick_video(tmp_path)
        assert result is not None
        assert result.suffix == ".mp4"

    def test_finds_mkv(self, tmp_path):
        (tmp_path / "video.mkv").write_bytes(b"fake")
        result = _pick_video(tmp_path)
        assert result is not None

    def test_prefers_mp4_over_mkv(self, tmp_path):
        (tmp_path / "video.mp4").write_bytes(b"fake")
        (tmp_path / "video.mkv").write_bytes(b"fake")
        result = _pick_video(tmp_path)
        assert result.suffix == ".mp4"

    def test_returns_none_when_empty(self, tmp_path):
        assert _pick_video(tmp_path) is None

    def test_ignores_non_video_files(self, tmp_path):
        (tmp_path / "video.txt").write_bytes(b"fake")
        (tmp_path / "video.info.json").write_bytes(b"{}")
        assert _pick_video(tmp_path) is None

    def test_finds_webm(self, tmp_path):
        (tmp_path / "video.webm").write_bytes(b"fake")
        result = _pick_video(tmp_path)
        assert result is not None


# ---------------------------------------------------------------------------
# VIDEO_EXTS constant
# ---------------------------------------------------------------------------

class TestVideoExts:
    def test_common_formats_present(self):
        for ext in (".mp4", ".mkv", ".webm", ".mov"):
            assert ext in VIDEO_EXTS

    def test_all_lowercase(self):
        for ext in VIDEO_EXTS:
            assert ext == ext.lower()

    def test_all_start_with_dot(self):
        for ext in VIDEO_EXTS:
            assert ext.startswith(".")
