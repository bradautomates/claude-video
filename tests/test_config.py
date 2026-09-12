"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

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


def test_read_env_file_strips_inline_comment(tmp_path):
    path = tmp_path / ".env"
    path.write_text("WATCH_DETAIL=balanced  # note\n", encoding="utf-8")
    assert config.read_env_file(path) == {"WATCH_DETAIL": "balanced"}


def test_read_env_file_strips_comment_after_quoted_value(tmp_path):
    path = tmp_path / ".env"
    path.write_text('WATCH_DETAIL="balanced"  # note\n', encoding="utf-8")
    assert config.read_env_file(path) == {"WATCH_DETAIL": "balanced"}


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100
