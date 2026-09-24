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
    assert {"detail", "config_file", "whisper_backend", "whisperx_model", "sub_lang"} <= set(cfg)


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


import pytest
from pathlib import Path


@pytest.mark.parametrize('value,expected', [
    ('"a#=b" # comment', 'a#=b'), ("'C:\\Users\\a' # note", 'C:\\Users\\a'),
    ('C:\\Users\\a # note', 'C:\\Users\\a'), ('abc#def=ghi', 'abc#def=ghi'),
    ('"$HOME\\n"', '$HOME\\n'),
])
def test_literal_dotenv_values(tmp_path, value, expected):
    path = tmp_path / '.env'
    path.write_text('KEY=' + value)
    assert config.read_env_file(path)['KEY'] == expected


@pytest.mark.parametrize('value', ['"private-secret', '"private-secret" junk'])
def test_malformed_quotes_never_echo_value(tmp_path, value):
    path = tmp_path / '.env'
    path.write_text('KEY=' + value)
    with pytest.raises(config.ConfigError) as exc:
        config.read_env_file(path)
    assert 'line 1' in str(exc.value) and 'private-secret' not in str(exc.value)


def test_provider_preference_precedes_location(monkeypatch):
    Path('.env').write_text('GROQ_API_KEY=project-test\n')
    monkeypatch.setenv('OPENAI_API_KEY', 'environment-test')
    assert config.load_api_key() == ('groq', 'project-test')
    assert config.load_api_key('openai') == ('openai', 'environment-test')


def test_user_file_beats_project_file(monkeypatch):
    config.write_settings({'GROQ_API_KEY': 'user-test'})
    Path('.env').write_text('GROQ_API_KEY=project-test\n')
    assert config.load_api_key() == ('groq', 'user-test')
    monkeypatch.setenv('GROQ_API_KEY', 'environment-test')
    assert config.load_api_key() == ('groq', 'environment-test')


def test_engine_defaults_to_auto_and_local_without_key():
    cfg = config.get_config()
    assert cfg['engine'] == 'auto'
    assert cfg['gemini_model'] == 'gemini-3.7-flash'
    assert cfg['gemini_timeout'] == 600.0
    assert config.load_gemini_key() is None
    assert config.resolve_engine('auto', has_key=False) == 'local'


def test_auto_selects_gemini_when_key_present(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    assert config.load_gemini_key() == 'test-key'
    assert config.resolve_engine('auto', has_key=True) == 'gemini'
    assert config.resolve_engine('local', has_key=True) == 'local'


def test_gemini_key_lookup_order(monkeypatch, tmp_path):
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_text('GEMINI_API_KEY=from-user-file\n')
    (tmp_path / 'cwd' / '.env').write_text('GEMINI_API_KEY=from-cwd\n')
    assert config.load_gemini_key() == 'from-user-file'
    monkeypatch.setenv('GEMINI_API_KEY', 'from-env')
    assert config.load_gemini_key() == 'from-env'
    monkeypatch.delenv('GEMINI_API_KEY')
    config.CONFIG_FILE.write_text('GEMINI_API_KEY=\n')
    assert config.load_gemini_key() == 'from-cwd'


def test_explicit_gemini_without_key_is_a_safe_error():
    with pytest.raises(config.ConfigError, match='GEMINI_API_KEY'):
        config.resolve_engine('gemini', has_key=False)


def test_invalid_engine_and_timeout_rejected(monkeypatch):
    monkeypatch.setenv('WATCH_ENGINE', 'vertex')
    with pytest.raises(config.ConfigError, match='WATCH_ENGINE'):
        config.get_config()
    monkeypatch.setenv('WATCH_ENGINE', 'auto')
    monkeypatch.setenv('WATCH_GEMINI_TIMEOUT', '-5')
    with pytest.raises(config.ConfigError, match='WATCH_GEMINI_TIMEOUT'):
        config.get_config()
