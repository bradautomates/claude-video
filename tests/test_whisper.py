"""Whisper auto-chunking: plan, split, and timestamp stitching."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import pytest

import whisper


MB = 1024 * 1024


class TestPlanChunks:
    def test_under_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=5 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_at_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=24 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_over_limit_splits_into_enough_chunks(self):
        # 71 MB against a 24 MB cap → ceil(71/24) = 3 chunks.
        plan = whisper.plan_chunks(total_seconds=3600.0, total_bytes=71 * MB, max_bytes=24 * MB)
        assert len(plan) == 3

    def test_chunks_are_contiguous_and_cover_full_duration(self):
        total = 3600.0
        plan = whisper.plan_chunks(total_seconds=total, total_bytes=71 * MB, max_bytes=24 * MB)
        # Offsets start at 0 and each picks up where the previous ended.
        assert plan[0][0] == 0.0
        for (off, dur), (next_off, _) in zip(plan, plan[1:]):
            assert math.isclose(off + dur, next_off)
        last_off, last_dur = plan[-1]
        assert math.isclose(last_off + last_dur, total)

    def test_each_chunk_estimated_under_limit(self):
        total_seconds, total_bytes, cap = 3600.0, 71 * MB, 24 * MB
        plan = whisper.plan_chunks(total_seconds, total_bytes, cap)
        bytes_per_second = total_bytes / total_seconds
        for _off, dur in plan:
            assert dur * bytes_per_second <= cap

    def test_zero_duration_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=0.0, total_bytes=0, max_bytes=24 * MB)
        assert plan == [(0.0, 0.0)]


class TestShiftSegments:
    def test_adds_offset_to_start_and_end(self):
        segs = [{"start": 0.0, "end": 2.5, "text": "hi"}, {"start": 2.5, "end": 4.0, "text": "there"}]
        shifted = whisper.shift_segments(segs, 1800.0)
        assert shifted == [
            {"start": 1800.0, "end": 1802.5, "text": "hi"},
            {"start": 1802.5, "end": 1804.0, "text": "there"},
        ]

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert whisper.shift_segments(segs, 0.0) == segs

    def test_does_not_mutate_input(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "x"}]
        whisper.shift_segments(segs, 10.0)
        assert segs[0]["start"] == 0.0


def _make_mp3(path: Path, seconds: float) -> None:
    """Synthesize a mono 16k 64k mp3 of a sine tone — mirrors extract_audio's format."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(path),
        ],
        check=True,
    )


class TestSplitAudio:
    def test_creates_one_file_per_plan_entry(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert len(chunks) == 2
        for chunk_path, _offset in chunks:
            assert chunk_path.exists() and chunk_path.stat().st_size > 0

    def test_returns_plan_offsets(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert [offset for _path, offset in chunks] == [0.0, 3.0]

    def test_chunks_are_smaller_than_full(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        full_size = full.stat().st_size
        for chunk_path, _offset in chunks:
            assert chunk_path.stat().st_size < full_size


class TestAudioDuration:
    def test_reads_duration_of_synthesized_clip(self, tmp_path: Path):
        audio = tmp_path / "audio.mp3"
        _make_mp3(audio, 5.0)
        assert whisper.audio_duration(audio) == pytest.approx(5.0, abs=0.5)


class TestTranscribeChunks:
    def test_shifts_and_concatenates_each_chunk(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def fake_transcribe(path: Path) -> list[dict]:
            return [{"start": 0.0, "end": 2.0, "text": path.stem}]

        out = whisper.transcribe_chunks(chunks, fake_transcribe)

        assert out == [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 100.0, "end": 102.0, "text": "b"},
        ]

    def test_keeps_successful_chunks_when_one_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def flaky(path: Path) -> list[dict]:
            if path.stem == "b":
                raise SystemExit("chunk b failed")
            return [{"start": 1.0, "end": 2.0, "text": "a"}]

        out = whisper.transcribe_chunks(chunks, flaky)

        assert out == [{"start": 1.0, "end": 2.0, "text": "a"}]

    def test_raises_when_every_chunk_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def always_fail(path: Path) -> list[dict]:
            raise SystemExit("boom")

        with pytest.raises(SystemExit):
            whisper.transcribe_chunks(chunks, always_fail)


def test_key_without_backend_rejected_before_extraction(monkeypatch, tmp_path):
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: pytest.fail('must validate before extraction'))
    with pytest.raises(SystemExit, match='explicit'):
        whisper.transcribe_video('video.mp4', tmp_path / 'audio', api_key='dummy')


@pytest.mark.parametrize('size,allowed', [(23_999_999, True), (24_000_000, True), (24_000_001, False)])
def test_upload_budget_checks_actual_file(monkeypatch, tmp_path, size, allowed):
    audio = tmp_path / 'audio.mp3'
    with audio.open('wb') as f:
        f.truncate(size)
    calls = []
    def multipart(*a):
        calls.append(True)
        raise RuntimeError('passed budget')
    monkeypatch.setattr(whisper, '_build_multipart', multipart)
    with pytest.raises(RuntimeError if allowed else SystemExit):
        whisper._post_whisper('https://example.test', 'dummy', 'model', audio)
    assert bool(calls) == allowed


def test_multipart_size_checked_before_request(monkeypatch, tmp_path):
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'a')
    monkeypatch.setattr(whisper, 'MAX_MULTIPART_BYTES', 3)
    monkeypatch.setattr(whisper, '_build_multipart', lambda *a: (b'1234', 'boundary'))
    with pytest.raises(SystemExit, match='Multipart'):
        whisper._post_whisper('https://example.test', 'dummy', 'model', audio)


def test_unexpected_oversized_split_never_uploads(monkeypatch, tmp_path):
    audio = tmp_path / 'a.mp3'
    audio.write_bytes(b'123456')
    monkeypatch.setattr(whisper, 'MAX_UPLOAD_BYTES', 5)
    monkeypatch.setattr(whisper, 'extract_audio', lambda *a: audio)
    monkeypatch.setattr(whisper, 'audio_duration', lambda *a: 20)
    monkeypatch.setattr(whisper, 'plan_chunks', lambda *a: [(0, 10), (10, 10)])
    monkeypatch.setattr(whisper, 'split_audio', lambda *a: [(audio, 0), (audio, 10)])
    monkeypatch.setattr(whisper, '_transcribe_file', lambda *a: pytest.fail('no oversized uploads'))
    with pytest.raises(SystemExit, match='exceeds'):
        whisper.transcribe_video('video', audio, backend='groq', api_key='dummy')


def test_partial_chunks_keep_gap_metadata():
    def transcribe(path):
        if path.name == 'b':
            raise SystemExit('failed')
        return [{'start': 0, 'end': 1, 'text': 'ok'}]
    out = whisper.transcribe_chunks([(Path('a'), 0), (Path('b'), 10), (Path('c'), 20)], transcribe)
    assert out.gaps == [{'start': 10, 'end': 20}]
    assert out[-1]['start'] == 20


def test_no_speech_and_malformed_response_differ():
    assert whisper._segments_from_response({'segments': []}).no_speech
    with pytest.raises(SystemExit, match='malformed'):
        whisper._segments_from_response({'text': 'timestamps missing'})
