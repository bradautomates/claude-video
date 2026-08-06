"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

import codecs
from pathlib import Path

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


# --- .env encodings Windows actually produces -------------------------------
# Each of these was a real failure before decode_env_bytes: the first two raised
# UnicodeDecodeError (a ValueError, so `except OSError` never caught it) and
# crashed the run; the last two parsed "fine" but dropped the key silently.

@pytest.mark.parametrize(
    "name,data",
    [
        ("utf-8", b"WATCH_DETAIL=efficient\n"),
        ("utf-8-bom", "WATCH_DETAIL=efficient\n".encode("utf-8-sig")),
        ("utf-16-bom", "WATCH_DETAIL=efficient\n".encode("utf-16")),
        ("utf-16-be-bom", "WATCH_DETAIL=efficient\n".encode("utf-16-be")
                          .join([codecs.BOM_UTF16_BE, b""])),
        ("utf-16-le-no-bom", "WATCH_DETAIL=efficient\n".encode("utf-16-le")),
        ("cp1252", "WATCH_DETAIL=efficient  # configuración\n".encode("cp1252")),
    ],
)
def test_env_readable_in_every_windows_encoding(name, data, tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(data)
    assert config.read_env_file(path) == {"WATCH_DETAIL": "efficient"}, name


def test_undecodable_env_never_raises(tmp_path):
    """Worst case is a degraded read, never a crash that kills the whole run."""
    path = tmp_path / ".env"
    path.write_bytes(b"\xff\xfe\x00WATCH_DETAIL\x81\x8d=efficient")
    config.read_env_file(path)  # must not raise


def test_unreadable_env_falls_back_to_defaults(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"WATCH_DETAIL=efficient\n")

    def boom(self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", boom)
    assert config.read_env_file(path) == {}


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100
