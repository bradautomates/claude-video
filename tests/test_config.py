"""WATCH_DETAIL resolution, frame_cap mapping, and output encoding."""
from __future__ import annotations

import sys

import pytest

import config


def test_default_detail_is_balanced(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_env_overrides_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "efficient")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "efficient"


def test_invalid_detail_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "bogus")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_get_config_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert set(cfg) == {"detail", "config_file", "whisper_backend"}


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


def test_force_utf8_output_survives_cp1252_console(monkeypatch):
    """The report's em dash and arrow must survive a cp1252 console.

    Regression test for a Windows crash: printing the focus-range line
    ("0:10 -> 0:20") raised UnicodeEncodeError under the default cp1252
    console encoding, killing the run after the video had already been
    downloaded and transcribed.
    """
    import io

    raw_out, raw_err = io.BytesIO(), io.BytesIO()
    # Simulate a Windows console: cp1252-backed, strict.
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw_out, encoding="cp1252", errors="strict"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(raw_err, encoding="cp1252", errors="strict"))

    line = "- **Focus range:** 0:10 → 0:20 — done"

    # Without the fix this raises UnicodeEncodeError.
    with pytest.raises(UnicodeEncodeError):
        print(line)
        sys.stdout.flush()

    config.force_utf8_output()

    print(line)
    sys.stdout.flush()
    assert "→" in raw_out.getvalue().decode("utf-8")
    assert sys.stderr.encoding.lower().replace("-", "") == "utf8"


def test_force_utf8_output_reconfigures_real_streams(capsys):
    """Calling the helper leaves both streams on UTF-8."""
    config.force_utf8_output()
    # capsys replaces the streams with objects that may lack reconfigure;
    # the helper must not raise either way.
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if hasattr(stream, "reconfigure") and encoding is not None:
            assert encoding.lower().replace("-", "") == "utf8"
class TestReadEnvValue:
    """One parser for every consumer.

    whisper.py and setup.py each carried their own .env reader that predated
    read_env_file()'s inline-comment handling, so the same line resolved
    differently depending on who read it: a commented key reached the API with
    the comment attached (401), and `SETUP_COMPLETE=true  # done` never
    registered, re-triggering first-run setup forever.
    """

    def _env(self, tmp_path, body):
        path = tmp_path / ".env"
        path.write_text(body, encoding="utf-8")
        return [path]

    def test_inline_comment_is_stripped(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        paths = self._env(tmp_path, "GROQ_API_KEY=sk-secret-abc   # my groq key\n")
        assert config.read_env_value("GROQ_API_KEY", paths=paths) == "sk-secret-abc"

    def test_inline_comment_does_not_break_setup_complete(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SETUP_COMPLETE", raising=False)
        paths = self._env(tmp_path, "SETUP_COMPLETE=true   # finished\n")
        assert config.read_env_value("SETUP_COMPLETE", paths=paths) == "true"

    def test_hash_inside_quoted_value_is_kept(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        paths = self._env(tmp_path, 'GROQ_API_KEY="sk-has#hash-inside"\n')
        assert config.read_env_value("GROQ_API_KEY", paths=paths) == "sk-has#hash-inside"

    def test_environment_wins_over_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "sk-from-env")
        paths = self._env(tmp_path, "GROQ_API_KEY=sk-from-file\n")
        assert config.read_env_value("GROQ_API_KEY", paths=paths) == "sk-from-env"

    def test_blank_environment_falls_through_to_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GROQ_API_KEY", "   ")
        paths = self._env(tmp_path, "GROQ_API_KEY=sk-from-file\n")
        assert config.read_env_value("GROQ_API_KEY", paths=paths) == "sk-from-file"

    def test_earlier_path_wins(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        first, second = tmp_path / "a.env", tmp_path / "b.env"
        first.write_text("GROQ_API_KEY=sk-first\n", encoding="utf-8")
        second.write_text("GROQ_API_KEY=sk-second\n", encoding="utf-8")
        assert config.read_env_value("GROQ_API_KEY", paths=[first, second]) == "sk-first"

    def test_missing_everywhere_is_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        assert config.read_env_value("GROQ_API_KEY", paths=[tmp_path / "nope.env"]) is None

    def test_on_file_hook_sees_each_existing_path(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        paths = self._env(tmp_path, "GROQ_API_KEY=sk-x\n")
        seen = []
        config.read_env_value("GROQ_API_KEY", paths=paths, on_file=seen.append)
        assert seen == paths


def test_read_env_file_strips_inline_comment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("WATCH_DETAIL=balanced  # note\n", encoding="utf-8")
    assert config.read_env_file(path) == {"WATCH_DETAIL": "balanced"}


def test_read_env_file_strips_comment_after_quoted_value(tmp_path):
    path = tmp_path / ".env"
    path.write_text('WATCH_DETAIL="balanced"  # note\n', encoding="utf-8")
    assert config.read_env_file(path) == {"WATCH_DETAIL": "balanced"}


def test_read_env_file_keeps_hash_inside_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text('K="a # b"\n', encoding="utf-8")
    assert config.read_env_file(path) == {"K": "a # b"}


def test_ytdlp_cmd_defaults_to_path_lookup(monkeypatch):
    monkeypatch.delenv("WATCH_YTDLP", raising=False)
    monkeypatch.setattr(config, "read_env_value", lambda name: None)
    monkeypatch.setattr(config.shutil, "which", lambda n: "/opt/bin/yt-dlp")
    assert config.ytdlp_cmd() == ["/opt/bin/yt-dlp"]


def test_ytdlp_cmd_override_may_be_a_command(monkeypatch):
    monkeypatch.setattr(config, "read_env_value", lambda name: "python -m yt_dlp" if name == "WATCH_YTDLP" else None)
    assert config.ytdlp_cmd() == ["python", "-m", "yt_dlp"]


class TestEnvEncodings:
    """Windows writes .env files in several encodings (#119): PowerShell 5's
    `> .env` is UTF-16LE with BOM, Notepad offers UTF-8 with BOM, Set-Content
    uses the ANSI code page. All must parse; none may raise."""

    def test_utf16le_with_bom(self, tmp_path):
        p = tmp_path / ".env"; p.write_bytes("GROQ_API_KEY=abc\n".encode("utf-16"))
        assert config.read_env_file(p) == {"GROQ_API_KEY": "abc"}

    def test_utf16le_without_bom(self, tmp_path):
        p = tmp_path / ".env"; p.write_bytes("WATCH_DETAIL=efficient\n".encode("utf-16-le"))
        assert config.read_env_file(p) == {"WATCH_DETAIL": "efficient"}

    def test_utf8_bom_does_not_poison_first_key(self, tmp_path):
        p = tmp_path / ".env"; p.write_bytes(b"\xef\xbb\xbfWATCH_DETAIL=balanced\n")
        assert config.read_env_file(p) == {"WATCH_DETAIL": "balanced"}

    def test_ansi_codepage_does_not_raise(self, tmp_path, capsys):
        p = tmp_path / ".env"; p.write_bytes("GROQ_API_KEY=abc # ó\n".encode("cp1252"))
        assert config.read_env_file(p)["GROQ_API_KEY"] == "abc"
        assert "not valid UTF-8" in capsys.readouterr().err
