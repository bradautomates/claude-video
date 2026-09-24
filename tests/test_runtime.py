"""Real byte/pipe behavior and actionable launch diagnostics."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import runtime
import frames
from conftest import SCRIPTS_DIR


def test_cp1252_streams_preserve_unicode():
    code = f"import sys; sys.path.insert(0, {str(SCRIPTS_DIR)!r}); from runtime import configure_stdio; configure_stdio(); print('안녕 → café'); print('日本語', file=sys.stderr)"
    result = subprocess.run([sys.executable, '-c', code], env={**os.environ, 'PYTHONIOENCODING': 'cp1252'}, capture_output=True)
    assert result.returncode == 0
    assert '안녕 → café' in result.stdout.decode('utf-8')
    assert '日本語' in result.stderr.decode('utf-8')


def test_replacement_stream_without_reconfigure(monkeypatch):
    monkeypatch.setattr(runtime.sys, 'stdout', object())
    monkeypatch.setattr(runtime.sys, 'stderr', object())
    runtime.configure_stdio()


def test_invalid_diagnostic_bytes():
    result = runtime.run_text([sys.executable, '-c', "import sys; sys.stderr.buffer.write(b'bad\\xff')"])
    assert result.stderr == 'bad\ufffd'


@pytest.mark.parametrize('error', [PermissionError(), FileNotFoundError()])
def test_launch_failure_names_binary(monkeypatch, error):
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    with pytest.raises(SystemExit, match='ffprobe'):
        frames.get_metadata('anything.mp4')


def test_byte_valued_timeout(monkeypatch):
    def fail(*a, **kw):
        raise subprocess.TimeoutExpired('ffprobe', 30, stderr=b'bad\xff')
    monkeypatch.setattr(runtime.subprocess, 'run', fail)
    with pytest.raises(SystemExit, match='timed out.*bad'):
        frames.get_metadata('anything.mp4')


@pytest.mark.parametrize('data', ['not json', '{}', '{"streams":[]}', '{"streams":[{"codec_type":"video"}],"format":{"duration":"NaN"}}'])
def test_invalid_probe_metadata(monkeypatch, data):
    monkeypatch.setattr(frames, 'run_text', lambda *a, **kw: subprocess.CompletedProcess([], 0, data, ''))
    with pytest.raises(SystemExit, match='metadata'):
        frames.get_metadata('test.mp4')


@pytest.mark.parametrize('help_text,expected', [('-vsync  set video sync', '-vsync'), ('-fps_mode  set fps\n-vsync  set sync', '-fps_mode'), ('-fps_mode  set fps', '-fps_mode'), ('-fps_mode[:<stream_spec>]  set framerate mode for matching video streams', '-fps_mode'), ('unknown option: fps_mode', None)])
def test_sync_capability_probe(monkeypatch, help_text, expected):
    calls = []
    def help_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, help_text, '')
    monkeypatch.setattr(frames, 'run_text', help_run)
    frames._sync_option.cache_clear()
    if expected:
        assert frames._sync_option('fake-ffmpeg')[0] == expected
        frames._sync_option('fake-ffmpeg')
        assert len(calls) == 1
    else:
        with pytest.raises(SystemExit, match='advertise'):
            frames._sync_option('fake-ffmpeg')
    frames._sync_option.cache_clear()


@pytest.mark.parametrize('stderr', ['no pts', 'pts_time:nan', 'pts_time:inf'])
def test_never_substitutes_missing_pts(stderr):
    with pytest.raises(SystemExit, match='timestamp'):
        frames._emitted_frames([Path('a.jpg')], stderr, 0, 'test')


def test_signed_exponent_pts():
    out = frames._emitted_frames([Path('a.jpg')], 'pts_time:+1.2e-3', 3.5, 'test')
    assert out[0]['timestamp_seconds'] == 3.5012
