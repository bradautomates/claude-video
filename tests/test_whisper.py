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


class TestCustomEndpoint:
    """Self-hosted OpenAI-compatible transcription servers."""

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch, tmp_path):
        # Keep the developer's real keys and dotenv files out of these tests.
        for var in (
            "GROQ_API_KEY",
            "OPENAI_API_KEY",
            whisper.CUSTOM_ENDPOINT_VAR,
            whisper.CUSTOM_MODEL_VAR,
            whisper.CUSTOM_KEY_VAR,
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [tmp_path / "absent.env"])

    def test_unset_endpoint_reports_nothing(self):
        assert whisper.custom_endpoint() == (None, None)

    def test_endpoint_defaults_the_model(self, monkeypatch):
        monkeypatch.setenv(whisper.CUSTOM_ENDPOINT_VAR, "http://localhost:8000/v1/audio/transcriptions")

        endpoint, model = whisper.custom_endpoint()

        assert endpoint == "http://localhost:8000/v1/audio/transcriptions"
        assert model == whisper.CUSTOM_MODEL_DEFAULT

    def test_model_is_overridable(self, monkeypatch):
        monkeypatch.setenv(whisper.CUSTOM_ENDPOINT_VAR, "http://localhost:8000/v1/audio/transcriptions")
        monkeypatch.setenv(whisper.CUSTOM_MODEL_VAR, "Systran/faster-whisper-large-v3")

        assert whisper.custom_endpoint()[1] == "Systran/faster-whisper-large-v3"

    def test_custom_wins_over_hosted_keys(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_hosted")
        monkeypatch.setenv(whisper.CUSTOM_ENDPOINT_VAR, "http://localhost:8000/v1/audio/transcriptions")

        assert whisper.load_api_key() == ("custom", "")

    def test_custom_key_is_optional_but_honoured(self, monkeypatch):
        monkeypatch.setenv(whisper.CUSTOM_ENDPOINT_VAR, "http://localhost:8000/v1/audio/transcriptions")
        monkeypatch.setenv(whisper.CUSTOM_KEY_VAR, "local-secret")

        assert whisper.load_api_key() == ("custom", "local-secret")

    def test_forcing_custom_without_endpoint_yields_nothing(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_hosted")

        # Must not silently fall back to Groq — the user asked for local only.
        assert whisper.load_api_key("custom") == (None, None)

    def test_hosted_still_works_when_no_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_hosted")

        assert whisper.load_api_key() == ("groq", "gsk_hosted")

    def test_endpoint_readable_from_dotenv(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            f'{whisper.CUSTOM_ENDPOINT_VAR}="http://box:8000/v1/audio/transcriptions"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(whisper, "_dotenv_paths", lambda: [env_file])

        assert whisper.custom_endpoint()[0] == "http://box:8000/v1/audio/transcriptions"
