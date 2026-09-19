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


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


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
