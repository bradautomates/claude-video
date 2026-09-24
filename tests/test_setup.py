"""Preflight and config writes use controlled dependency availability."""
import json
import os
from pathlib import Path

import pytest
import config
import setup


@pytest.fixture(autouse=True)
def binaries_present(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: [])


def test_json_reports_watch_detail(monkeypatch, capsys):
    monkeypatch.setattr(setup, '_probe', lambda cmd: {'ok': True, 'output': 'test'})
    setup.cmd_json()
    assert json.loads(capsys.readouterr().out)['watch_detail'] == 'balanced'


def test_keyless_completed_setup_proceeds_silently(capsys):
    config.write_settings({'SETUP_COMPLETE': 'true', 'WATCH_WHISPER_BACKEND': 'none'})
    assert setup.cmd_check() == 0
    assert capsys.readouterr() == ('', '')
    assert setup._status()['status'] == 'ready'
    assert not setup._status()['first_run']


def test_keyless_first_run_can_proceed(capsys):
    assert setup.cmd_check() == 0
    assert capsys.readouterr() == ('', '')
    assert setup._status()['first_run']


def test_key_present_is_ready(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    assert setup._status()['whisper_backend'] == 'groq'
    assert setup._status()['can_proceed']


def test_missing_binaries_are_independent_of_keys(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: ['ffprobe'])
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    assert setup.cmd_check() == 2
    assert setup._status()['status'] == 'needs_install'


def test_project_marker_cannot_complete_setup(monkeypatch):
    Path('.env').write_text('SETUP_COMPLETE=true\nGROQ_API_KEY=test-project\n')
    monkeypatch.setenv('SETUP_COMPLETE', 'true')
    assert setup.is_first_run()
    assert setup._have_api_key() == (True, 'groq')


def test_marker_replacement_preserves_comments_and_key():
    setup._scaffold_env()
    config.CONFIG_FILE.write_text('# preserved\nGROQ_API_KEY="test#=key"\nSETUP_COMPLETE=false\nSETUP_COMPLETE=false\n')
    setup._write_setup_complete()
    text = config.CONFIG_FILE.read_text()
    assert '# preserved' in text and 'test#=key' in text
    assert text.count('SETUP_COMPLETE=') == 1
    assert not setup.is_first_run()


def test_failed_atomic_write_preserves_original(monkeypatch):
    setup._scaffold_env()
    before = config.CONFIG_FILE.read_bytes()
    def denied(*args):
        raise PermissionError()
    monkeypatch.setattr(config.os, 'replace', denied)
    with pytest.raises(PermissionError):
        setup._write_setup_complete()
    assert config.CONFIG_FILE.read_bytes() == before


def test_config_write_refuses_bad_encoding():
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_bytes(b'KEY=\xff')
    with pytest.raises(config.ConfigError, match='UTF-8'):
        setup._write_setup_complete()
    assert config.CONFIG_FILE.read_bytes() == b'KEY=\xff'


@pytest.mark.parametrize('system,mode,warns', [('Windows', 0o644, False), ('Linux', 0o600, False), ('Linux', 0o644, True)])
def test_permission_diagnostics(monkeypatch, tmp_path, capsys, system, mode, warns):
    if os.name == 'nt' and system != 'Windows':
        pytest.skip('Real POSIX mode semantics require POSIX')
    path = tmp_path / 'secrets'
    path.write_text('')
    path.chmod(mode)
    monkeypatch.setattr(setup.platform, 'system', lambda: system)
    setup._check_file_permissions(path)
    assert bool(capsys.readouterr().err) == warns


def test_check_never_launches_whisperx(monkeypatch):
    config.write_settings({'WATCH_WHISPER_BACKEND': 'whisperx'})
    monkeypatch.setattr(setup, '_probe', lambda *a: pytest.fail('check must be lightweight'))
    assert setup.cmd_check() == 0


def test_none_backend_completion():
    assert setup.cmd_install('none', 'efficient') == 0
    assert not setup.is_first_run()
    assert config.get_config()['whisper_backend'] == 'none'
    assert config.get_config()['detail'] == 'efficient'


def test_gemini_engine_is_ready_without_media_binaries(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    status = setup._status()
    assert status['engine'] == 'gemini' and status['configured_engine'] == 'auto'
    assert status['gemini_key_present'] is True and status['gemini_model'] == 'gemini-3.7-flash'
    assert status['can_proceed'] is True and status['status'] == 'ready'
    assert status['binaries_required'] is False
    assert status['missing_binaries'] == ['ffmpeg', 'ffprobe', 'yt-dlp']
    assert 'unit-key' not in json.dumps(status)
    assert setup.cmd_check() == 0


def test_local_engine_still_requires_binaries(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    status = setup._status()
    assert status['engine'] == 'local' and status['gemini_key_present'] is False
    assert status['can_proceed'] is False and status['binaries_required'] is True
    assert setup.cmd_check() == 2


def test_engine_gemini_without_key_writes_choice_and_asks_for_key(monkeypatch, capsys):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    assert setup.cmd_install(engine='gemini') == 3
    assert 'GEMINI_API_KEY' in capsys.readouterr().err
    values = config.read_env_file(config.CONFIG_FILE)
    assert values['WATCH_ENGINE'] == 'gemini' and values.get('SETUP_COMPLETE') != 'true'


def test_engine_gemini_with_key_completes_setup_and_preserves_lines(monkeypatch):
    monkeypatch.setattr(setup, '_check_binaries', lambda: list(setup.REQUIRED_BINARIES))
    config.CONFIG_DIR.mkdir(parents=True)
    config.CONFIG_FILE.write_text('# my notes\nGEMINI_API_KEY=unit-key\nGROQ_API_KEY=keep-me\n')
    assert setup.cmd_install(engine='gemini') == 0
    text = config.CONFIG_FILE.read_text()
    assert '# my notes' in text and 'GROQ_API_KEY=keep-me' in text
    values = config.read_env_file(config.CONFIG_FILE)
    assert values['WATCH_ENGINE'] == 'gemini' and values['SETUP_COMPLETE'] == 'true'
