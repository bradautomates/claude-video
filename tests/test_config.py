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
    assert set(cfg) == {
        "detail",
        "whisper",
        "whisper_model",
        "whisper_device",
        "whisper_compute",
        "whisper_language",
        "config_file",
    }


def test_whisper_defaults_to_auto(monkeypatch, tmp_path):
    for name in ("WATCH_WHISPER", "WATCH_WHISPER_MODEL", "WATCH_WHISPER_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert cfg["whisper"] == "auto"
    assert cfg["whisper_model"] == ""


def test_whisper_backend_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_WHISPER", "LOCAL")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["whisper"] == "local"


def test_whisper_backend_invalid_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_WHISPER", "mlx")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["whisper"] == "auto"


def test_whisper_settings_from_config_file(monkeypatch, tmp_path):
    for name in ("WATCH_WHISPER", "WATCH_WHISPER_MODEL", "WATCH_WHISPER_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "WATCH_WHISPER=local\n"
        "WATCH_WHISPER_MODEL=small  # smaller download\n"
        "WATCH_WHISPER_DEVICE=cpu\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    cfg = config.get_config()
    assert cfg["whisper"] == "local"
    assert cfg["whisper_model"] == "small"
    assert cfg["whisper_device"] == "cpu"


def test_blank_env_var_falls_through_to_file(monkeypatch, tmp_path):
    """A scaffolded-but-empty env var must not shadow a real config value."""
    monkeypatch.setenv("WATCH_WHISPER", "")
    env = tmp_path / ".env"
    env.write_text("WATCH_WHISPER=local\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_FILE", env)
    assert config.get_config()["whisper"] == "local"


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100
