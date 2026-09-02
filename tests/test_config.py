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
    assert set(cfg) == {"detail", "config_file"}


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
