"""Behavioral reproductions from the v0.3.0 review ledger."""
import json
import subprocess
from pathlib import Path

import pytest
import config
import frames
import transcribe
import whisper
from conftest import build_static_clip


@pytest.mark.parametrize('encoding', ['utf-8-sig', 'utf-16', 'utf-16-be'])
def test_config_bom_quotes_and_last_assignment(tmp_path, encoding):
    body = '# comment\r\nWATCH_DETAIL=balanced\r\nWATCH_DETAIL="efficient" # note\r\n'
    data = body.encode(encoding)
    if encoding == 'utf-16-be':
        data = b'\xfe\xff' + data
    path = tmp_path / '.env'
    path.write_bytes(data)
    assert config.read_env_file(path)['WATCH_DETAIL'] == 'efficient'


def test_uniform_capped_full_range(tmp_path):
    clip = tmp_path / 'long.mp4'
    build_static_clip(clip, duration=30)
    out = frames.extract(str(clip), tmp_path / 'frames', fps=2, max_frames=6)
    assert len(out) == 6
    assert out[-1]['timestamp_seconds'] >= 25


def test_keyframe_gap_falls_back(static_clip, tmp_path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / 'f', start_seconds=1, end_seconds=2)
    assert out and meta['fallback']


def test_optional_hours_entities_and_independent_repetition(tmp_path):
    path = tmp_path / 'subs.vtt'
    path.write_text('WEBVTT\n\n00:01.000 --> 00:02.000\nThank &lt;you&gt;\n\n01:00.000 --> 01:01.000\nThank &lt;you&gt;\n')
    assert transcribe.parse_vtt(str(path)) == [
        {'start': 1.0, 'end': 2.0, 'text': 'Thank <you>'},
        {'start': 60.0, 'end': 61.0, 'text': 'Thank <you>'},
    ]


@pytest.mark.parametrize('backend', ['groq', 'openai'])
def test_helper_resolves_only_matching_key(monkeypatch, tmp_path, backend):
    def key(preferred=None):
        assert preferred == backend
        return backend, 'dummy'
    monkeypatch.setattr(whisper, 'load_api_key', key)
    audio = tmp_path / 'audio.mp3'
    audio.write_bytes(b'audio')
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: audio)
    monkeypatch.setattr(whisper, '_transcribe_file', lambda b, k, p: [{'start': 0, 'end': 1, 'text': 'ok'}])
    assert whisper.transcribe_video('video', audio, backend=backend)[1] == backend
