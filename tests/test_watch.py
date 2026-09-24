"""End-to-end routing of --detail through watch.py on a local clip."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"


def _run(clip: Path, *args: str, env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_efficient_uses_keyframe_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient")
    assert "(keyframe" in out
    assert "**Detail:** efficient" in out


def test_balanced_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced")
    assert "(scene" in out
    assert "**Detail:** balanced" in out


def test_token_burner_uses_scene_engine(cut_clip: Path):
    out = _run(cut_clip, "--detail", "token-burner")
    assert "(scene" in out


def test_transcript_skips_frames(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript")
    assert "skipped" in out
    assert "frame_0000.jpg" not in out


def test_flag_overrides_env(cut_clip: Path):
    out = _run(cut_clip, "--detail", "efficient", env_extra={"WATCH_DETAIL": "balanced"})
    assert "(keyframe" in out


def test_default_is_balanced(cut_clip: Path):
    out = _run(cut_clip)  # no flag, WATCH_DETAIL cleared
    assert "**Detail:** balanced" in out
    assert "(scene" in out


def test_timestamps_add_cue_frames_to_detail(cut_clip: Path):
    out = _run(cut_clip, "--detail", "balanced", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" in out  # detail frames still present (additive)


def test_timestamps_with_transcript_detail_is_cue_only(cut_clip: Path):
    out = _run(cut_clip, "--detail", "transcript", "--timestamps", "1,3")
    assert "reason=transcript-cue" in out
    assert "reason=scene-change" not in out
    assert "reason=keyframe" not in out


def _frame_lines(out: str) -> int:
    return sum(1 for line in out.splitlines() if "/frames/frame_" in line.replace("\\", "/") and "(t=" in line)


def test_dedup_collapses_static_by_default(static_clip: Path):
    out = _run(static_clip)  # solid blue → identical frames collapse to one
    assert "near-duplicate" in out
    assert _frame_lines(out) == 1


def test_no_dedup_preserves_static_frames(static_clip: Path):
    out = _run(static_clip, "--no-dedup")
    assert "near-duplicate" not in out
    assert _frame_lines(out) > 1


import pytest
import watch
from transcribe import Segments


def _caption_context(monkeypatch, tmp_path, body):
    subtitle = tmp_path / 'captions.vtt'
    subtitle.write_text('WEBVTT\n\n' + body)
    monkeypatch.setattr(watch, 'fetch_captions', lambda *a, **kw: {
        'subtitle_path': str(subtitle), 'info': {'title': '안녕', 'duration': 30},
        'caption_track': {'language': 'ko', 'kind': 'manual', 'provenance': 'original'},
    })
    return subtitle


def test_silent_focus_never_calls_asr(monkeypatch, tmp_path, capsys):
    _caption_context(monkeypatch, tmp_path, '00:00.000 --> 00:10.000\nhello\n')
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: pytest.fail('caption-only focus must not download'))
    monkeypatch.setattr(watch, 'transcribe_video', lambda *a, **kw: pytest.fail('a silent focus is not a missing track'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://example.com/video', '--detail', 'transcript', '--start', '20', '--end', '30'])
    assert watch.main() == 0
    assert 'no speech in selected range' in capsys.readouterr().out


def test_good_captions_survive_media_failure(monkeypatch, tmp_path, capsys):
    _caption_context(monkeypatch, tmp_path, '00:00.000 --> 00:10.000\nhello\n')
    def fail(*a, **kw):
        raise SystemExit('blocked download')
    monkeypatch.setattr(watch, 'download', fail)
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://example.com/video'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'hello' in out and 'Media unavailable' in out and 'ko, manual, original' in out


def test_good_captions_survive_probe_denial(monkeypatch, tmp_path, capsys):
    _caption_context(monkeypatch, tmp_path, '00:00.000 --> 00:10.000\nhello\n')
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: {'video_path': 'media.mp4'})
    def fail(*a):
        raise SystemExit('Cannot run ffprobe (PermissionError)')
    monkeypatch.setattr(watch, 'get_metadata', fail)
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://example.com/video'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'hello' in out and 'PermissionError' in out


def _mock_audio(monkeypatch):
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: {'video_path': 'video.mp4'})
    monkeypatch.setattr(watch, 'get_metadata', lambda *a: {'duration_seconds': 30, 'has_audio': True, 'has_video': False})


def test_partial_report_includes_missing_intervals(monkeypatch, capsys):
    _mock_audio(monkeypatch)
    monkeypatch.setattr(watch, 'load_api_key', lambda *a: ('groq', 'dummy'))
    monkeypatch.setattr(watch, 'transcribe_video', lambda *a, **kw: (Segments([{'start': 1, 'end': 2, 'text': 'good'}], gaps=[{'start': 10, 'end': 20}]), 'groq'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert 'partial; missing intervals' in out and '00:10 → 00:20' in out


def test_no_whisper_disables_configured_local_backend(monkeypatch):
    _mock_audio(monkeypatch)
    monkeypatch.setenv('WATCH_WHISPER_BACKEND', 'whisperx')
    monkeypatch.setattr(watch, 'transcribe_video', lambda *a, **kw: pytest.fail('fallback disabled'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript', '--no-whisper'])
    assert watch.main() == 0


def test_local_failure_does_not_resolve_cloud(monkeypatch, capsys):
    _mock_audio(monkeypatch)
    monkeypatch.setattr(watch, 'load_api_key', lambda *a: pytest.fail('no cloud fallback'))
    def fail(*a, **kw):
        assert kw['backend'] == 'whisperx'
        raise SystemExit('local model failed')
    monkeypatch.setattr(watch, 'transcribe_video', fail)
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript', '--whisper', 'whisperx'])
    assert watch.main() == 1
    assert 'local model failed' in capsys.readouterr().out


def test_contradictory_flags_rejected(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--no-whisper', '--whisper', 'whisperx'])
    with pytest.raises(SystemExit):
        watch.main()


def test_user_out_dir_and_local_source_preserved(static_clip, tmp_path):
    source = tmp_path / 'source.mp4'
    source.write_bytes(static_clip.read_bytes())
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('keep')
    _run(source, '--out-dir', str(tmp_path))
    assert source.read_bytes() == static_clip.read_bytes() and sentinel.read_text() == 'keep'
    assert list(tmp_path.glob('watch-*/frames/frame_*.jpg'))


def test_cli_no_whisper_overrides_invalid_saved_backend(monkeypatch):
    _mock_audio(monkeypatch)
    monkeypatch.setenv('WATCH_WHISPER_BACKEND', 'invalid-old-preference')
    monkeypatch.setattr(sys, 'argv', ['watch', 'video.mp4', '--detail', 'transcript', '--no-whisper'])
    assert watch.main() == 0


def _gemini_answer(monkeypatch, seen):
    import gemini
    def fake_ask(video, question, **kw):
        seen.update(video=video, question=question, **kw)
        return {'text': 'At 00:10 it turns blue.', 'model': kw['model'], 'processing': 'agentic', 'total_tokens': 4348}
    monkeypatch.setattr(gemini, 'ask', fake_ask)


def test_youtube_with_key_uses_gemini_and_touches_nothing_local(monkeypatch, capsys):
    seen = {}
    _gemini_answer(monkeypatch, seen)
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    for name in ('fetch_captions', 'download', 'get_metadata', 'transcribe_video'):
        monkeypatch.setattr(watch, name, lambda *a, _n=name, **kw: pytest.fail(f'{_n} must not run on a Gemini YouTube run'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://youtu.be/abc', '--question', 'When does it change?', '--detail', 'efficient'])
    assert watch.main() == 0
    out = capsys.readouterr().out
    assert seen['video'] == {'uri': 'https://youtu.be/abc'} and seen['question'] == 'When does it change?'
    assert seen['clip'] is None and seen['model'] == 'gemini-3.7-flash'
    assert '**Engine:** gemini-3.7-flash (agentic)' in out
    assert '**Ignored local options:** --detail' in out
    assert '## Answer (from Gemini)' in out and 'At 00:10 it turns blue.' in out
    assert 'unit-key' not in out


def test_local_file_is_uploaded_asked_then_deleted(monkeypatch, capsys, static_clip):
    import gemini
    seen, deleted = {}, []
    _gemini_answer(monkeypatch, seen)
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    monkeypatch.setattr(gemini, 'upload_file', lambda path, key, **kw: {'name': 'files/abc', 'uri': 'https://g/files/abc', 'mime_type': 'video/mp4'})
    monkeypatch.setattr(gemini, 'delete_file', lambda name, key: deleted.append(name))
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--start', '0:20', '--end', '0:30'])
    assert watch.main() == 0
    assert seen['video'] == {'uri': 'https://g/files/abc', 'mime_type': 'video/mp4'}
    assert seen['clip'] == (20.0, 30.0)
    assert deleted == ['files/abc']
    assert 'uploaded to Google' in capsys.readouterr().out


def test_upload_is_deleted_even_when_the_question_fails(monkeypatch, capsys, static_clip):
    import gemini
    deleted = []
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    monkeypatch.setattr(gemini, 'upload_file', lambda path, key, **kw: {'name': 'files/abc', 'uri': 'u', 'mime_type': 'video/mp4'})
    monkeypatch.setattr(gemini, 'delete_file', lambda name, key: deleted.append(name))
    def boom(*a, **kw):
        raise SystemExit('Gemini quota: slow down. No local fallback was attempted; rerun with --engine local')
    monkeypatch.setattr(gemini, 'ask', boom)
    monkeypatch.setattr(watch, 'get_metadata', lambda *a, **kw: pytest.fail('no silent fallback to the local pipeline'))
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip)])
    assert watch.main() == 1
    assert deleted == ['files/abc']
    out = capsys.readouterr().out
    assert 'Gemini quota' in out and '--engine local' in out and '## Frames' not in out


def test_engine_local_never_contacts_gemini(monkeypatch, static_clip, capsys):
    import gemini
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    for name in ('ask', 'upload_file'):
        monkeypatch.setattr(gemini, name, lambda *a, **kw: pytest.fail('local engine must not contact Google'))
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--engine', 'local', '--no-whisper'])
    assert watch.main() == 0
    assert '## Frames' in capsys.readouterr().out


def test_no_key_keeps_the_local_pipeline(monkeypatch, static_clip, capsys):
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--no-whisper'])
    assert watch.main() == 0
    assert '## Frames' in capsys.readouterr().out


def test_explicit_gemini_without_key_fails_before_any_work(monkeypatch):
    from config import ConfigError
    monkeypatch.setattr(watch, 'download', lambda *a, **kw: pytest.fail('must fail before downloading'))
    monkeypatch.setattr(sys, 'argv', ['watch', 'https://vimeo.com/1', '--engine', 'gemini'])
    with pytest.raises(ConfigError, match='GEMINI_API_KEY'):
        watch.main()


def test_failed_upload_is_reported_honestly_and_work_dir_removed(monkeypatch, capsys, static_clip, tmp_path):
    import gemini
    monkeypatch.setenv('GEMINI_API_KEY', 'unit-key')
    def boom(path, key, **kw):
        raise SystemExit('Gemini upload: nope. No local fallback was attempted; rerun with --engine local')
    monkeypatch.setattr(gemini, 'upload_file', boom)
    out_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['watch', str(static_clip), '--out-dir', str(out_dir)])
    assert watch.main() == 1
    out = capsys.readouterr().out
    assert 'URL sent to Google' not in out and 'upload to Google failed' in out
    assert out_dir.is_dir() and list(out_dir.iterdir()) == [], 'the run-owned work dir must be removed'
    assert static_clip.is_file(), 'the user source file is never deleted'
