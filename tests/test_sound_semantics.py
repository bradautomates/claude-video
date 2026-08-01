import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import sound_semantics  # noqa: E402
from sound_semantics import (  # noqa: E402
    analyze_sound_semantics,
    build_prompt,
    call_gemini,
    call_openai,
    extract_semantic_audio,
    extract_json_object,
    normalize_result,
    resolve_provider,
)


class ProviderResolutionTests(unittest.TestCase):
    def test_auto_does_not_reuse_whisper_key_without_opt_in(self) -> None:
        settings = {"OPENAI_API_KEY": "existing-whisper-key"}
        with patch.object(sound_semantics, "read_config", side_effect=settings.get):
            provider, key, reason = resolve_provider("auto", "auto")

        self.assertIsNone(provider)
        self.assertIsNone(key)
        self.assertIn("not enabled", reason)

    def test_provider_flag_alone_does_not_bypass_auto_privacy_gate(self) -> None:
        settings = {"OPENAI_API_KEY": "existing-whisper-key"}
        with patch.object(sound_semantics, "read_config", side_effect=settings.get):
            provider, key, reason = resolve_provider("auto", "openai")

        self.assertIsNone(provider)
        self.assertIsNone(key)
        self.assertIn("not enabled", reason)

    def test_api_mode_selects_available_key(self) -> None:
        settings = {"OPENAI_API_KEY": "openai-key"}
        with patch.object(sound_semantics, "read_config", side_effect=settings.get):
            provider, key, reason = resolve_provider("api", "auto")

        self.assertEqual("openai", provider)
        self.assertEqual("openai-key", key)
        self.assertEqual("enabled", reason)

    def test_auto_uses_dedicated_provider_setting(self) -> None:
        settings = {
            "SOUND_SEMANTICS_PROVIDER": "gemini",
            "GEMINI_API_KEY": "gemini-key",
        }
        with patch.object(sound_semantics, "read_config", side_effect=settings.get):
            provider, key, _ = resolve_provider("auto", "auto")

        self.assertEqual("gemini", provider)
        self.assertEqual("gemini-key", key)


class SemanticParsingTests(unittest.TestCase):
    def test_extract_json_tolerates_fence_and_preamble(self) -> None:
        value = extract_json_object('Result follows:\n```json\n{"events": []}\n```')
        self.assertEqual({"events": []}, value)

    def test_normalization_offsets_and_grounds_event_times(self) -> None:
        raw = {
            "summary": "A sharp hit punctuates the edit.",
            "music": {"present": False},
            "events": [
                {
                    "start": 1.2,
                    "end": 1.45,
                    "label": "low-frequency impact",
                    "category": "SFX",
                    "confidence": 1.7,
                    "description": "Fast attack and bass-heavy decay.",
                    "story_function": "Punctuates a reveal.",
                    "diegetic": "maybe",
                    "evidence": ["fast attack"],
                }
            ],
        }
        result = normalize_result(
            raw,
            10.0,
            15.0,
            "openai",
            "audio-model",
            audio_analysis={"transient_peaks": [{"time": 11.22, "label": "spectral change"}]},
            video_analysis={"change_peaks": [{"time": 11.19, "score": 9.2}]},
        )

        event = result["events"][0]
        self.assertEqual(11.2, event["start"])
        self.assertEqual(11.45, event["end"])
        self.assertEqual("sfx", event["category"])
        self.assertEqual(1.0, event["confidence"])
        self.assertEqual("unclear", event["diegetic"])
        self.assertEqual(11.22, event["grounding"]["audio_change"]["time"])
        self.assertEqual(11.19, event["grounding"]["visual_change"]["time"])

    def test_prompt_uses_clip_relative_measurement_times(self) -> None:
        prompt = build_prompt(
            5.0,
            audio_analysis={"transient_peaks": [{"time": 12.25, "label": "change"}]},
            video_analysis={"change_peaks": [{"time": 13.0, "score": 5.0}]},
            source_start=10.0,
        )
        self.assertIn('"time":2.25', prompt)
        self.assertIn('"time":3.0', prompt)


class ProviderPayloadTests(unittest.TestCase):
    def test_openai_sends_mp3_as_audio_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "clip.mp3"
            audio_path.write_bytes(b"audio")
            response = {"choices": [{"message": {"content": '{"events":[]}'}}]}
            with patch.object(sound_semantics, "_post_json", return_value=response) as post:
                text = call_openai(audio_path, "prompt", "secret", "gpt-audio-test")

        self.assertEqual('{"events":[]}', text)
        url, payload, headers = post.call_args.args
        self.assertEqual(sound_semantics.OPENAI_ENDPOINT, url)
        self.assertEqual("gpt-audio-test", payload["model"])
        audio_part = payload["messages"][0]["content"][1]
        self.assertEqual("input_audio", audio_part["type"])
        self.assertEqual("mp3", audio_part["input_audio"]["format"])
        self.assertEqual("Bearer secret", headers["Authorization"])

    def test_gemini_requests_schema_constrained_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "clip.mp3"
            audio_path.write_bytes(b"audio")
            response = {
                "candidates": [{"content": {"parts": [{"text": '{"events":[]}' }]}}]
            }
            with patch.object(sound_semantics, "_post_json", return_value=response) as post:
                text = call_gemini(audio_path, "prompt", "secret", "gemini-audio-test")

        self.assertEqual('{"events":[]}', text)
        url, payload, headers = post.call_args.args
        self.assertIn("gemini-audio-test:generateContent", url)
        config = payload["generation_config"]
        self.assertEqual("application/json", config["response_format"]["text"]["mime_type"])
        self.assertIn("response_schema", config)
        inline = payload["contents"][0]["parts"][1]["inline_data"]
        self.assertEqual("audio/mp3", inline["mime_type"])
        self.assertEqual("secret", headers["x-goog-api-key"])


class SemanticCacheTests(unittest.TestCase):
    def test_successful_result_is_cached_without_persisting_audio(self) -> None:
        raw = json.dumps(
            {
                "summary": "One impact.",
                "events": [
                    {
                        "start": 0.5,
                        "end": 0.7,
                        "label": "impact",
                        "category": "sfx",
                        "confidence": 0.8,
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"fixture")

            def fake_extract(_source, out_path, _start, _end):
                audio = out_path.with_suffix(".mp3")
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_bytes(b"audio")
                return audio

            settings = {
                "SOUND_SEMANTICS_PROVIDER": "openai",
                "OPENAI_API_KEY": "key",
            }
            with (
                patch.object(sound_semantics, "read_config", side_effect=settings.get),
                patch.object(sound_semantics, "extract_semantic_audio", side_effect=fake_extract),
                patch.object(sound_semantics, "call_openai", return_value=raw) as provider_call,
            ):
                first = analyze_sound_semantics(
                    str(source), root / "work", root / "cache", 0.0, 2.0
                )
                second = analyze_sound_semantics(
                    str(source), root / "work", root / "cache", 0.0, 2.0
                )

            self.assertEqual("ok", first["status"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(1, provider_call.call_count)
            self.assertFalse((root / "work" / "semantic-audio.mp3").exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
class SemanticAudioExtractionTests(unittest.TestCase):
    def test_selected_range_becomes_small_mono_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture.wav"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=3",
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            audio = extract_semantic_audio(str(source), root / "selected", 1.0, 2.0)

            self.assertTrue(audio.exists())
            self.assertEqual(".mp3", audio.suffix)
            self.assertLess(audio.stat().st_size, 32_000)


if __name__ == "__main__":
    unittest.main()
