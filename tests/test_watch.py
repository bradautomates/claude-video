"""End-to-end routing of --detail through watch.py on a local clip."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WATCH = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"
TIKTOK = "https://www.tiktok.com/@iloveeestrayyykidsss/photo/7463596003225013536"


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
    return sum(1 for line in out.splitlines() if "/frames/frame_" in line and "(t=" in line)


def test_dedup_collapses_static_by_default(static_clip: Path):
    out = _run(static_clip)  # solid blue → identical frames collapse to one
    assert "near-duplicate" in out
    assert _frame_lines(out) == 1


def test_no_dedup_preserves_static_frames(static_clip: Path):
    out = _run(static_clip, "--no-dedup")
    assert "near-duplicate" not in out
    assert _frame_lines(out) > 1


def _media(path: Path, *ffmpeg_args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *ffmpeg_args, str(path)],
        check=True,
    )
    return path


def _run_in_process(monkeypatch, capsys, dl: dict, *args: str) -> str:
    import watch

    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(watch, "fetch_captions", lambda url, out: {"subtitle_path": None, "info": {}})
    monkeypatch.setattr(watch, "download", lambda *a, **k: dl)
    monkeypatch.setattr(sys, "argv", ["watch", TIKTOK, "--no-whisper", *args])
    assert watch.main() == 0
    return capsys.readouterr().out


def test_slideshow_report_lists_every_slide(monkeypatch, capsys, tmp_path: Path):
    slides = [
        str(_media(tmp_path / f"{i:03d}.jpg", "-f", "lavfi", "-i", f"color=c={c}:s=540x720", "-frames:v", "1"))
        for i, c in enumerate(["red", "green", "blue"], 1)
    ]
    audio = _media(tmp_path / "000.m4a", "-f", "lavfi", "-t", "2", "-i", "sine=f=440", "-c:a", "aac")
    dl = {
        "video_path": str(audio),
        "image_paths": slides,
        "subtitle_path": None,
        "info": {"title": "caption text", "uploader": "kev"},
        "downloaded": True,
    }
    out = _run_in_process(monkeypatch, capsys, dl, "--detail", "balanced", "--start", "5")
    assert "**Frames:** 3 of 3 slides" in out
    assert [line for line in out.splitlines() if "(slide " in line][-1].endswith("(slide 3)")
    assert sum("(slide " in line for line in out.splitlines()) == 3
    assert "Focus range" not in out  # --start is meaningless without a timeline
    assert "**Title:** caption text" in out


def test_audio_only_download_does_not_crash_frame_engine(monkeypatch, capsys, tmp_path: Path):
    audio = _media(tmp_path / "video.mp3", "-f", "lavfi", "-t", "2", "-i", "sine=f=440", "-c:a", "aac", "-f", "adts")
    dl = {"video_path": str(audio), "subtitle_path": None, "info": {}, "downloaded": True}
    out = _run_in_process(monkeypatch, capsys, dl, "--detail", "balanced")
    assert "_No frames extracted._" in out
