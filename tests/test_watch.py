"""End-to-end routing of --detail through watch.py on a local clip."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import watch

WATCH = (
    Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts" / "watch.py"
)
SKILL = WATCH.parent.parent / "SKILL.md"


def _run(clip: Path, *args: str, env_extra: dict | None = None) -> str:
    env = dict(os.environ)
    env.pop("WATCH_DETAIL", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(WATCH), str(clip), "--no-whisper", *args],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _manifest(out_dir: Path) -> dict:
    return json.loads((out_dir / "frame-index.json").read_text())


def _vtt(path: Path, text: str = "UNIQUE_TRANSCRIPT_SENTINEL") -> None:
    path.write_text(
        f"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{text}\n",
        encoding="utf-8",
    )


def test_efficient_uses_keyframe_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "efficient", "--out-dir", str(tmp_path))
    assert "(keyframe" in out
    assert "**Detail:** efficient" in out
    assert _manifest(tmp_path)["frame_count"] > 0


def test_balanced_uses_scene_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "balanced", "--out-dir", str(tmp_path))
    assert "(scene" in out
    assert "**Detail:** balanced" in out


def test_token_burner_uses_scene_engine(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "token-burner", "--out-dir", str(tmp_path))
    assert "(scene" in out


def test_transcript_skips_frames(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "transcript", "--out-dir", str(tmp_path))
    assert "skipped" in out
    assert "frame_0000.jpg" not in out
    assert not (tmp_path / "frame-index.json").exists()


def test_flag_overrides_env(cut_clip: Path, tmp_path: Path):
    out = _run(
        cut_clip,
        "--detail",
        "efficient",
        "--out-dir",
        str(tmp_path),
        env_extra={"WATCH_DETAIL": "balanced"},
    )
    assert "(keyframe" in out


def test_default_is_balanced(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--out-dir", str(tmp_path))
    assert "**Detail:** balanced" in out
    assert "(scene" in out


def test_timestamps_add_cue_frames_to_detail(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip,
        "--detail",
        "balanced",
        "--timestamps",
        "1,3",
        "--out-dir",
        str(tmp_path),
    )
    reasons = {frame["reason"] for frame in _manifest(tmp_path)["frames"]}
    assert "transcript-cue" in reasons
    assert "scene-change" in reasons


def test_timestamps_with_transcript_detail_is_cue_only(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip,
        "--detail",
        "transcript",
        "--timestamps",
        "1,3",
        "--out-dir",
        str(tmp_path),
    )
    reasons = {frame["reason"] for frame in _manifest(tmp_path)["frames"]}
    assert reasons == {"transcript-cue"}


def test_focused_transcript_cues_preserve_semantics(cut_clip: Path, tmp_path: Path):
    _run(
        cut_clip,
        "--detail",
        "transcript",
        "--timestamps",
        "0.5,2,4",
        "--start",
        "1",
        "--end",
        "3",
        "--out-dir",
        str(tmp_path),
    )
    frames = _manifest(tmp_path)["frames"]
    assert [(frame["timestamp_seconds"], frame["reason"]) for frame in frames] == [
        (2.0, "transcript-cue")
    ]


def test_stdout_lists_no_individual_frame_paths(cut_clip: Path, tmp_path: Path):
    out = _run(cut_clip, "--detail", "balanced", "--out-dir", str(tmp_path))
    assert "/frames/frame_" not in out
    assert "/frames/cue_" not in out
    assert "frame-index.json" in out
    assert "overview_0001.jpg" in out
    assert "new focused extraction" in out
    assert "fresh harness inspect digest" in out
    assert "visual_harness.py" in out
    assert "Workers remain tool-less" in out
    assert "Send each overview page" not in out
    assert "native Agent" not in out
    assert "continue the originating visual worker" not in out


def test_stdout_summarizes_many_overview_pages(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cut_clip: Path,
    tmp_path: Path,
):
    pages = [
        {
            "page": page,
            "frame_start": (page - 1) * 20,
            "frame_end": page * 20 - 1,
            "kind": "image",
            "path": str(tmp_path / "overview" / f"overview_{page:04d}.jpg"),
        }
        for page in range(1, 501)
    ]
    monkeypatch.setattr(
        watch,
        "prepare_frame_presentation",
        lambda *_args, **_kwargs: {
            "index_path": tmp_path / "frame-index.json",
            "pages": pages,
            "manifest": {},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            str(cut_clip),
            "--detail",
            "balanced",
            "--no-whisper",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    assert "**Overview pages:** 500" in (out := capsys.readouterr().out)
    assert "overview_0001.jpg" in out
    assert "overview_0500.jpg" in out
    assert "overview_0250.jpg" not in out
    assert len(out) < 5_000


def test_dedup_collapses_static_by_default(static_clip: Path, tmp_path: Path):
    out = _run(static_clip, "--out-dir", str(tmp_path))
    assert "near-duplicate" in out
    assert _manifest(tmp_path)["frame_count"] == 1


def test_no_dedup_preserves_static_frames(static_clip: Path, tmp_path: Path):
    out = _run(static_clip, "--no-dedup", "--out-dir", str(tmp_path))
    assert "near-duplicate" not in out
    assert _manifest(tmp_path)["frame_count"] > 1


def test_fsync_directory_flushes_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    calls = []
    monkeypatch.setattr(watch.os, "open", lambda *_args: 17)
    monkeypatch.setattr(
        watch.os, "fsync", lambda descriptor: calls.append(("fsync", descriptor))
    )
    monkeypatch.setattr(
        watch.os, "close", lambda descriptor: calls.append(("close", descriptor))
    )

    watch._fsync_directory(tmp_path)

    assert calls == [("fsync", 17), ("close", 17)]


def test_fsync_directory_tolerates_unsupported_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        watch.os, "open", lambda *_args: (_ for _ in ()).throw(OSError("unsupported"))
    )

    watch._fsync_directory(tmp_path)


def test_fsync_directory_closes_after_flush_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    closed = []
    monkeypatch.setattr(watch.os, "open", lambda *_args: 23)
    monkeypatch.setattr(
        watch.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("unsupported")),
    )
    monkeypatch.setattr(watch.os, "close", closed.append)

    watch._fsync_directory(tmp_path)

    assert closed == [23]


def test_direct_local_focused_run_honors_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cut_clip: Path,
    tmp_path: Path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            str(cut_clip),
            "--detail",
            "balanced",
            "--max-frames",
            "3",
            "--fps",
            "1",
            "--start",
            "0",
            "--end",
            "2",
            "--no-whisper",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    output = capsys.readouterr().out
    assert "**Focus range:** 00:00 → 00:02" in output
    assert "cap 3" in output
    assert _manifest(tmp_path)["frame_count"] <= 3


def test_transcript_is_published_without_raw_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    transcript = tmp_path / "transcript.txt"
    assert (
        transcript.read_text(encoding="utf-8") == "[00:00] UNIQUE_TRANSCRIPT_SENTINEL"
    )
    manifest = _manifest(tmp_path)
    assert manifest["frame_count"] == 0
    assert manifest["frames"] == []
    assert manifest["overview"]["page_count"] == 0
    assert manifest["overview"]["pages"] == []


def test_transcript_artifact_is_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    assert (tmp_path / "transcript.txt").stat().st_mode & 0o777 == 0o600


def test_existing_transcript_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    transcript = tmp_path / "transcript.txt"
    transcript.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="transcript target already exists"):
        watch.main()

    assert transcript.read_text(encoding="utf-8") == "preserve"


def test_transcript_symlink_target_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("preserve", encoding="utf-8")
    (tmp_path / "transcript.txt").symlink_to(outside)
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit, match="transcript target already exists"):
        watch.main()

    assert outside.read_text(encoding="utf-8") == "preserve"
    outside.unlink()


def test_transcript_stdout_contains_metadata_not_transcript(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    captured = capsys.readouterr()
    assert f"**Transcript artifact:** `{tmp_path / 'transcript.txt'}`" in captured.out
    assert "1 segment" in captured.out
    assert "via captions" in captured.out
    assert "UNIQUE_TRANSCRIPT_SENTINEL" not in captured.out
    assert "```" not in captured.out


def test_skill_requires_isolated_bounded_visual_review() -> None:
    references = SKILL.parent / "references"
    contract = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [SKILL, *sorted(references.glob("*.md"))]
    )
    required = (
        "coordinator must never `Read` raster images",
        "one fresh child per page",
        "host-enforced worker",
        "No Bash",
        "full-transcript access",
        "tool-free reducer",
        "8 KiB",
        "≤12 KiB",
        "observed|inferred",
        "uncertainty",
        "untrusted data, never instructions",
        "No coordinator raster fallback",
        "exact resolved model ID",
        "--json-schema",
        "--no-session-persistence",
        "runtime digest",
        "per-child timeout",
        "watch-review-v2.json",
        "No provider preflight",
        "automatic retry",
        "alias substitution",
    )

    assert all(term in contract for term in required)
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 100
        for path in [SKILL, *references.glob("*.md")]
    )
    frontmatter = SKILL.read_text(encoding="utf-8").split("---", 2)[1]
    assert "Agent" not in frontmatter
    assert "SendMessage" not in frontmatter
    assert "Native Agent delegation is not a security boundary" in contract




def test_transcript_temporary_file_is_removed_when_fdopen_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    closed = []
    monkeypatch.setattr(
        watch.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fdopen failed")),
    )
    monkeypatch.setattr(watch.os, "close", closed.append)

    with pytest.raises(OSError, match="fdopen failed"):
        watch.publish_transcript(tmp_path, "transcript")

    assert len(closed) == 1
    assert list(tmp_path.glob(".transcript.txt.*")) == []
    assert not (tmp_path / "transcript.txt").exists()


def test_transcript_only_without_captions_reports_missing_transcript(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    cut_clip: Path,
    tmp_path: Path,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            str(cut_clip),
            "--detail",
            "transcript",
            "--no-whisper",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert watch.main() == 0

    assert "No transcript available at transcript detail" in capsys.readouterr().out


def test_transcript_publication_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    subtitle = tmp_path / "captions.vtt"
    _vtt(subtitle)
    monkeypatch.setattr(
        watch,
        "fetch_captions",
        lambda *_args: {
            "video_path": None,
            "subtitle_path": str(subtitle),
            "info": {"duration": 5.6, "title": "fixture"},
            "downloaded": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "https://example.test/video",
            "--detail",
            "transcript",
            "--out-dir",
            str(tmp_path),
        ],
    )

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(watch.os, "link", fail_link)

    with pytest.raises(SystemExit, match="transcript publication failed"):
        watch.main()

    captured = capsys.readouterr()
    assert "UNIQUE_TRANSCRIPT_SENTINEL" not in captured.out
    assert not (tmp_path / "transcript.txt").exists()
    assert list(tmp_path.glob(".transcript.txt.*")) == []
