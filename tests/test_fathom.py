"""URL classification, data-page parsing, and cookie filtering for fathom.py.

No network: page HTML is synthesized to mirror Fathom's Inertia layout, and
the cookie-jar test writes a Netscape file by hand (including the expiry-0
session cookie and #HttpOnly_ forms yt-dlp emits).
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import fathom  # noqa: E402


SHARE_URL = "https://fathom.video/share/G9mkjkspnohVVZ_L5nrsoPycyWcB8y7s"
CALLS_URL = "https://fathom.video/calls/737523223"


def _page_html(props: dict) -> str:
    payload = html.escape(json.dumps({"component": "CallShow", "props": props}), quote=True)
    return f'<!DOCTYPE html><html><body><div id="app" data-page="{payload}"></div></body></html>'


class TestClassify:
    def test_share_url(self):
        assert fathom.classify(SHARE_URL) == "share"

    def test_share_url_www(self):
        assert fathom.classify("https://www.fathom.video/share/abc123") == "share"

    def test_calls_url(self):
        assert fathom.classify(CALLS_URL) == "calls"

    def test_other_url(self):
        assert fathom.classify("https://youtu.be/dQw4w9WgXcQ") == "other"

    def test_sign_in_is_not_calls(self):
        assert fathom.classify("https://fathom.video/users/sign_in") == "other"


class TestExtractProps:
    def test_roundtrip(self):
        props = {"call": {"id": 737523223, "video_url": "https://fathom.video/calls/737523223/video.m3u8"}}
        assert fathom.extract_props(_page_html(props)) == props

    def test_no_data_page(self):
        assert fathom.extract_props("<html><body>plain</body></html>") is None

    def test_invalid_json(self):
        page = '<div id="app" data-page="not-json"></div>'
        assert fathom.extract_props(page) is None


class TestFindKey:
    def test_nested_dict(self):
        obj = {"a": {"b": {"universalShareable": {"shareUrl": SHARE_URL}}}}
        assert fathom.find_key(obj, "universalShareable") == {"shareUrl": SHARE_URL}

    def test_inside_list(self):
        obj = {"items": [{"x": 1}, {"shareUrl": SHARE_URL}]}
        assert fathom.find_key(obj, "shareUrl") == SHARE_URL

    def test_missing(self):
        assert fathom.find_key({"a": [1, 2]}, "nope") is None


class TestCookieFiltering:
    def test_filters_to_fathom_and_survives_session_and_httponly(self, tmp_path, monkeypatch):
        """Only fathom.video cookies are kept; expiry-0 session cookies and
        #HttpOnly_ lines (both yt-dlp export forms) survive the round trip."""
        full_jar = tmp_path / "browser-cookies-full.txt"
        jar_text = (
            "# Netscape HTTP Cookie File\n"
            "fathom.video\tFALSE\t/\tTRUE\t0\t_fathom_session\tsecret-session\n"
            "#HttpOnly_.fathom.video\tTRUE\t/\tTRUE\t0\thttponly_session\thidden\n"
            "fathom.video\tFALSE\t/\tTRUE\t9999999999\tXSRF-TOKEN\ttok\n"
            ".youtube.com\tTRUE\t/\tTRUE\t9999999999\tPREF\tleak-me-not\n"
        )

        def fake_run(cmd, *args, **kwargs):
            full_jar.write_text(jar_text, encoding="utf-8")

            class _Result:
                returncode = 0

            return _Result()

        monkeypatch.setattr(fathom.subprocess, "run", fake_run)
        monkeypatch.setattr(fathom.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

        filtered = fathom.export_fathom_cookies("chrome", tmp_path)

        assert not full_jar.exists(), "full-browser jar must be deleted"
        content = filtered.read_text(encoding="utf-8")
        assert "_fathom_session" in content
        assert "httponly_session" in content
        assert "PREF" not in content and "youtube" not in content
        assert (filtered.stat().st_mode & 0o777) == 0o600

    def test_no_fathom_cookies_errors(self, tmp_path, monkeypatch):
        def fake_run(cmd, *args, **kwargs):
            (tmp_path / "browser-cookies-full.txt").write_text(
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tTRUE\t9999999999\tPREF\tv\n",
                encoding="utf-8",
            )

            class _Result:
                returncode = 0

            return _Result()

        monkeypatch.setattr(fathom.subprocess, "run", fake_run)
        monkeypatch.setattr(fathom.shutil, "which", lambda _: "/usr/local/bin/yt-dlp")

        with pytest.raises(SystemExit, match="no fathom.video cookies"):
            fathom.export_fathom_cookies("chrome", tmp_path)


class TestResolve:
    def test_share_url_passthrough_no_cookie_export(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("share URLs must not trigger cookie export")

        monkeypatch.setattr(fathom, "export_fathom_cookies", boom)
        result = fathom.resolve(SHARE_URL, "chrome", tmp_path)
        assert result["method"] == "passthrough"
        assert result["resolved_url"] == SHARE_URL
        assert result["requires_cookies"] is False

    def _patch_common(self, tmp_path, monkeypatch, final_url, page_html):
        jar = tmp_path / "fathom-cookies.txt"
        jar.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
        monkeypatch.setattr(fathom, "export_fathom_cookies", lambda browser, work: jar)
        monkeypatch.setattr(fathom, "fetch_call_page", lambda url, jar_path: (final_url, page_html))

    def test_calls_redirected_to_share(self, tmp_path, monkeypatch):
        self._patch_common(tmp_path, monkeypatch, SHARE_URL, "<html></html>")
        result = fathom.resolve(CALLS_URL, "chrome", tmp_path)
        assert result["method"] == "share"
        assert result["resolved_url"] == SHARE_URL
        assert result["requires_cookies"] is False

    def test_calls_page_with_share_url_in_props(self, tmp_path, monkeypatch):
        props = {
            "head": {"title": "Leadership Sync"},
            "call": {
                "id": 737523223,
                "video_url": "https://fathom.video/calls/737523223/video.m3u8",
                "universalShareable": {"shareUrl": SHARE_URL},
            },
        }
        self._patch_common(tmp_path, monkeypatch, CALLS_URL, _page_html(props))
        result = fathom.resolve(CALLS_URL, "chrome", tmp_path)
        assert result["method"] == "share"
        assert result["resolved_url"] == SHARE_URL
        assert result["title"] == "Leadership Sync"

    def test_calls_page_without_share_falls_back_to_hls(self, tmp_path, monkeypatch):
        props = {
            "call": {
                "id": 737523223,
                "video_url": "https://fathom.video/calls/737523223/video.m3u8",
                "universalShareable": {"shareUrl": None},
            },
        }
        self._patch_common(tmp_path, monkeypatch, CALLS_URL, _page_html(props))
        result = fathom.resolve(CALLS_URL, "chrome", tmp_path)
        assert result["method"] == "hls+cookies"
        assert result["resolved_url"].endswith("/video.m3u8")
        assert result["requires_cookies"] is True
        assert result["cookies_file"] is not None

    def test_sign_in_redirect_errors(self, tmp_path, monkeypatch):
        self._patch_common(
            tmp_path, monkeypatch, "https://fathom.video/users/sign_in", "<html></html>"
        )
        with pytest.raises(SystemExit, match="sign-in"):
            fathom.resolve(CALLS_URL, "chrome", tmp_path)
