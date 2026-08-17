"""Whisper auto-chunking: plan, split, and timestamp stitching."""
from __future__ import annotations

import json
import math
import subprocess
import urllib.error
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


class _FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class TestAtlasCloud:
    def test_default_backend_order_stays_groq_first(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-test-key")
        assert whisper.load_api_key() == ("groq", "groq-test-key")

    def test_explicit_key_selection(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "atlas-test-key")
        assert whisper.load_api_key("atlas") == ("atlas", "atlas-test-key")

    def test_prefers_utterance_segments(self):
        data = {
            "stt_result": {
                "text": "Hello world.",
                "duration": 2.0,
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word"},
                    {"text": "Hello world.", "start": 0.0, "end": 2.0, "type": "utterance"},
                ],
            }
        }
        assert whisper._segments_from_atlas_response(data) == [
            {"start": 0.0, "end": 2.0, "text": "Hello world."}
        ]

    def test_falls_back_to_full_transcript(self):
        data = {"stt_result": {"text": "Fallback text", "duration": 3.25}}
        assert whisper._segments_from_atlas_response(data) == [
            {"start": 0.0, "end": 3.25, "text": "Fallback text"}
        ]

    def test_merges_word_level_segments(self):
        data = {
            "stt_result": {
                "words": [
                    {"text": "Atlas", "start": 0.16, "end": 0.44, "type": "word"},
                    {"text": "Cloud", "start": 0.6, "end": 0.68, "type": "word"},
                    {
                        "text": "transcription",
                        "start": 0.68,
                        "end": 1.36,
                        "type": "word",
                    },
                    {"text": "works.", "start": 1.4, "end": 1.9, "type": "word"},
                ]
            }
        }
        assert whisper._segments_from_atlas_response(data) == [
            {"start": 0.16, "end": 1.9, "text": "Atlas Cloud transcription works."}
        ]

    def test_word_merging_honors_pauses_and_cjk_spacing(self):
        entries = [
            {"text": "你好", "start": 0.0, "end": 0.4},
            {"text": "世界。", "start": 0.4, "end": 0.8},
            {"text": "Next", "start": 2.2, "end": 2.5},
            {"text": "sentence", "start": 2.5, "end": 3.0},
        ]
        assert whisper._merge_atlas_words(entries) == [
            {"start": 0.0, "end": 0.8, "text": "你好世界。"},
            {"start": 2.2, "end": 3.0, "text": "Next sentence"},
        ]

    def test_upload_submit_and_bounded_poll(self, monkeypatch, tmp_path: Path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake-mp3")
        responses = [
            {"code": 200, "data": {"download_url": "https://media.example/audio.mp3"}},
            {"code": 200, "data": {"id": "prediction-123"}},
            {"code": 200, "data": {"id": "prediction-123", "status": "processing"}},
            {
                "code": 200,
                "data": {
                    "id": "prediction-123",
                    "status": "completed",
                    "stt_result": {
                        "words": [
                            {
                                "text": "Atlas works.",
                                "start": 0.1,
                                "end": 1.2,
                                "type": "utterance",
                            }
                        ]
                    },
                },
            },
        ]
        requests = []

        def fake_urlopen(request, **_kwargs):
            requests.append(request)
            return _FakeResponse(responses.pop(0))

        monkeypatch.setattr(whisper, "urlopen", fake_urlopen)
        monkeypatch.setattr(whisper.time, "sleep", lambda _seconds: None)

        segments = whisper._transcribe_file("atlas", "secret", audio)

        assert segments == [{"start": 0.1, "end": 1.2, "text": "Atlas works."}]
        assert [request.get_method() for request in requests] == ["POST", "POST", "GET", "GET"]
        assert requests[0].full_url == whisper.ATLAS_UPLOAD_ENDPOINT
        assert requests[1].full_url == whisper.ATLAS_GENERATE_ENDPOINT
        submitted = json.loads(requests[1].data.decode("utf-8"))
        assert submitted == {
            "model": whisper.ATLAS_MODEL,
            "audio_url": "https://media.example/audio.mp3",
            "format": "mp3",
            "enable_itn": True,
            "enable_punc": True,
            "show_utterances": True,
        }

    def test_submission_network_error_is_not_retried(self, monkeypatch):
        calls = 0

        def fail_once(_request, **_kwargs):
            nonlocal calls
            calls += 1
            raise urllib.error.URLError("offline")

        monkeypatch.setattr(whisper, "urlopen", fail_once)
        with pytest.raises(SystemExit, match="Atlas Cloud ASR submission failed"):
            whisper._atlas_submit_transcription("secret", "https://media.example/audio.mp3")
        assert calls == 1
