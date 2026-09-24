"""Exercise real child processes without downloading models or importing Torch."""
import json
import os
import sys
from pathlib import Path

import pytest
import config
import local_whisperx
import setup
import whisper


@pytest.fixture
def fake_whisperx(tmp_path):
    root = tmp_path / 'venv with spaces'
    bin_dir = root / ('Scripts' if os.name == 'nt' else 'bin')
    bin_dir.mkdir(parents=True)
    script = bin_dir / 'fake.py'
    script.write_text('''import json, os, pathlib, sys, time
if '--help' in sys.argv:
    print('--no_align --model'); sys.exit(0)
mode = os.environ.get('WATCH_TEST_MODE', 'good')
if mode == 'slow': time.sleep(10)
if mode == 'crash':
    sys.stderr.buffer.write(b'CUDA out of memory\\xff'); sys.exit(1)
if mode == 'offline':
    print('HuggingFace offline cache missing'); sys.exit(1)
out = pathlib.Path(sys.argv[sys.argv.index('--output_dir')+1])
path = out / (pathlib.Path(sys.argv[1]).stem + '.json')
if mode == 'missing': sys.exit(0)
if mode == 'malformed': path.write_text('{'); sys.exit(0)
assert os.environ['PYTHONIOENCODING'] == 'utf-8'
assert os.environ['PYTHONWARNINGS'] == 'ignore'
assert '--no_align' in sys.argv and '--task' in sys.argv and 'silero' in sys.argv
segments = [] if mode == 'empty' else [{'start': 1.0, 'end': 2.0, 'text': ' hola ', 'avg_logprob': -0.1}]
path.write_text(json.dumps({'segments': segments, 'language': 'en'}))
''', encoding='utf-8')
    exe = bin_dir / ('whisperx.cmd' if os.name == 'nt' else 'whisperx')
    if os.name == 'nt':
        exe.write_text(f'@"{sys.executable}" "%~dp0fake.py" %*\r\n')
    else:
        exe.write_text(f'#!{sys.executable}\n' + script.read_text())
        exe.chmod(0o755)
    (root / '.deps-ok').write_text('3.8.6')
    config.write_settings({'WATCH_WHISPERX_BIN': str(exe), 'WATCH_WHISPER_BACKEND': 'whisperx'})
    audio = tmp_path / 'audio with spaces.mp3'
    audio.write_bytes(b'audio')
    return exe, audio


def test_local_child_normalizes_segments_and_ignores_language(fake_whisperx):
    _, audio = fake_whisperx
    assert local_whisperx.transcribe_audio(audio) == [{'start': 1, 'end': 2, 'text': 'hola'}]
    assert not list(audio.parent.glob('watch-whisperx-*'))


def test_empty_is_no_speech(fake_whisperx, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'empty')
    segments = local_whisperx.transcribe_audio(fake_whisperx[1])
    assert not segments and segments.no_speech


@pytest.mark.parametrize('mode,message', [('malformed', 'malformed'), ('missing', 'missing'), ('crash', 'Memory or device'), ('offline', 'Model download')])
def test_local_failure_categories(fake_whisperx, monkeypatch, mode, message):
    monkeypatch.setenv('WATCH_TEST_MODE', mode)
    with pytest.raises(SystemExit, match=message):
        local_whisperx.transcribe_audio(fake_whisperx[1])


def test_timeout_kills_child_and_cleans_output(fake_whisperx, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'slow')
    monkeypatch.setenv('WATCH_WHISPERX_TIMEOUT', '0.2')
    with pytest.raises(SystemExit, match='timed out'):
        local_whisperx.transcribe_audio(fake_whisperx[1])
    assert not list(fake_whisperx[1].parent.glob('watch-whisperx-*'))


def test_incomplete_install_and_missing_executable(fake_whisperx):
    exe, audio = fake_whisperx
    local_whisperx.sentinel_for(exe).unlink()
    with pytest.raises(SystemExit, match='incomplete'):
        local_whisperx.transcribe_audio(audio)
    exe.unlink()
    with pytest.raises(SystemExit, match='missing'):
        local_whisperx.transcribe_audio(audio)


def test_local_route_never_resolves_a_cloud_key(fake_whisperx, monkeypatch):
    exe, audio = fake_whisperx
    monkeypatch.setattr(whisper, 'load_api_key', lambda *a, **k: pytest.fail('no cloud credentials for local inference'))
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: audio)
    assert whisper.transcribe_video('video.mp4', audio, backend='whisperx')[1] == 'whisperx'


@pytest.mark.parametrize('key,value', [('WATCH_WHISPERX_TIMEOUT', 'nan'), ('WATCH_WHISPERX_BATCH_SIZE', '1.5'),
                                     ('WATCH_WHISPERX_DEVICE', 'mps'), ('WATCH_WHISPERX_LANGUAGE', 'english')])
def test_invalid_local_settings(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(config.ConfigError):
        local_whisperx.settings()


def test_installer_rebuilds_partial_and_is_idempotent(monkeypatch):
    commands = []
    venv = Path.home() / '.cache/watch/whisperx-venv'
    venv.mkdir(parents=True)
    (venv / 'half-built').write_text('partial')
    monkeypatch.setattr(setup, '_ensure_uv', lambda: '/fake/uv')
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/' + name)
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    def step(cmd, label):
        commands.append(cmd)
        if 'venv' in cmd:
            (venv / 'bin').mkdir(parents=True)
            (venv / 'bin/python').touch()
            (venv / 'bin/whisperx').touch()
        if cmd[0] == 'ffmpeg':
            Path(cmd[-1]).write_bytes(b'audio')
    monkeypatch.setattr(setup, '_install_step', step)
    monkeypatch.setattr(local_whisperx, 'transcribe_audio', lambda *a, **kw: [])
    import subprocess
    monkeypatch.setattr(setup, 'run_text', lambda *a, **kw: subprocess.CompletedProcess([], 0, '[]', ''))
    assert setup.install_whisperx() == 0
    assert not (venv / 'half-built').exists()
    assert (venv / '.deps-ok').exists()
    assert config.get_config()['whisper_backend'] == 'whisperx'
    assert not setup.is_first_run()
    setup.install_whisperx()
    assert sum('venv' in cmd for cmd in commands) == 1


def test_failed_warmup_never_marks_ready(monkeypatch):
    venv = Path.home() / '.cache/watch/whisperx-venv'
    (venv / 'bin').mkdir(parents=True)
    for name in ['python', 'whisperx']:
        (venv / 'bin' / name).touch()
    (venv / '.deps-ok').write_text('old')
    monkeypatch.setattr(setup, '_ensure_uv', lambda: '/fake/uv')
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/' + name)
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup, '_install_step', lambda *a: None)
    def fail(*a, **kw):
        raise SystemExit('warmup failed')
    monkeypatch.setattr(local_whisperx, 'transcribe_audio', fail)
    with pytest.raises(SystemExit, match='warmup failed'):
        setup.install_whisperx()
    assert not (venv / '.deps-ok').exists() and setup.is_first_run()
    assert config.get_config()['whisper_backend'] == 'auto'


def test_cancellation_stops_the_process(fake_whisperx, monkeypatch):
    monkeypatch.setenv('WATCH_TEST_MODE', 'slow')
    original_get = local_whisperx.queue.Queue.get
    interrupted = False
    def interrupt_once(self, *args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original_get(self, *args, **kwargs)
    monkeypatch.setattr(local_whisperx.queue.Queue, 'get', interrupt_once)
    with pytest.raises(SystemExit, match='cancelled and stopped'):
        local_whisperx.transcribe_audio(fake_whisperx[1])
    assert not list(fake_whisperx[1].parent.glob('watch-whisperx-*'))


@pytest.mark.parametrize('system', ['Linux', 'Windows'])
def test_cpu_installer_recipe_keeps_torch_out_of_cuda_index(monkeypatch, system):
    import subprocess
    commands = []
    venv = Path.home() / '.cache/watch/whisperx-venv'
    monkeypatch.setattr(setup.platform, 'system', lambda: system)
    monkeypatch.setattr(setup, '_ensure_uv', lambda: '/fake/uv')
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/' + name)
    def step(cmd, label):
        commands.append([str(a) for a in cmd])
        if 'venv' in cmd:
            bin_dir = venv / ('Scripts' if system == 'Windows' else 'bin')
            bin_dir.mkdir(parents=True)
            for name in (('python.exe', 'whisperx.exe') if system == 'Windows' else ('python', 'whisperx')):
                (bin_dir / name).touch()
    monkeypatch.setattr(setup, '_install_step', step)
    monkeypatch.setattr(local_whisperx, 'transcribe_audio', lambda *a, **kw: [])
    monkeypatch.setattr(setup, 'run_text', lambda *a, **kw: subprocess.CompletedProcess([], 0, '[]', ''))
    assert setup.install_whisperx() == 0
    torch = next(cmd for cmd in commands if 'torch==2.8.0' in cmd)
    assert 'torchaudio==2.8.0' in torch and 'https://download.pytorch.org/whl/cpu' in torch
    assert any('whisperx==3.8.6' in cmd for cmd in commands)


@pytest.mark.parametrize('system', ['Darwin', 'Linux', 'Windows'])
def test_uv_bootstrap_discovers_new_executable_without_path_refresh(monkeypatch, system):
    commands = []
    monkeypatch.setattr(setup.platform, 'system', lambda: system)
    monkeypatch.setattr(setup, '_which', lambda name: '/fake/brew' if name == 'brew' else None)
    fallback = Path.home() / '.local/bin' / ('uv.exe' if system == 'Windows' else 'uv')
    def step(cmd, label):
        commands.append(cmd)
        if label == 'Install uv':
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.touch()
    monkeypatch.setattr(setup, '_install_step', step)
    assert setup._ensure_uv() == str(fallback)
    assert all('sudo' not in cmd for cmd in commands)
    assert commands[0][0] == {'Darwin': 'brew', 'Linux': 'curl', 'Windows': 'powershell'}[system]


def test_failed_uv_download_does_not_execute_shell(monkeypatch):
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(setup, '_which', lambda name: None)
    commands = []
    def fail(cmd, label):
        commands.append(cmd)
        raise SystemExit('download failed')
    monkeypatch.setattr(setup, '_install_step', fail)
    with pytest.raises(SystemExit, match='download failed'):
        setup._ensure_uv()
    assert len(commands) == 1 and commands[0][0] == 'curl'


def test_uv_bootstrap_reports_missing_homebrew(monkeypatch):
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup, '_which', lambda name: None)
    with pytest.raises(SystemExit, match='Homebrew'):
        setup._ensure_uv()
