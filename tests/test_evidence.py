"""Durable evidence bundle export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidence import build_evidence_bundle, transcript_excerpt


def test_transcript_excerpt_collects_nearby_segments_without_duplicates():
    segments = [
        {"start": 8.0, "end": 10.0, "text": "First point."},
        {"start": 10.0, "end": 12.0, "text": "First point."},
        {"start": 12.0, "end": 14.0, "text": "Second point."},
    ]

    assert transcript_excerpt(segments, 11.0) == "First point. Second point."


def test_transcript_excerpt_uses_nearest_segment_within_threshold():
    segments = [{"start": 20.0, "end": 22.0, "text": "Nearby line."}]

    assert transcript_excerpt(segments, 13.0) == "Nearby line."
    assert transcript_excerpt(segments, 11.0) is None


def test_build_evidence_bundle_writes_portable_artifacts(tmp_path: Path):
    source_frames = tmp_path / "source-frames"
    source_frames.mkdir()
    first = source_frames / "frame_0000.jpg"
    second = source_frames / "frame_0001.jpg"
    first.write_bytes(b"first-frame")
    second.write_bytes(b"second-frame")

    output = tmp_path / "evidence"
    (output / "frames").mkdir(parents=True)
    (output / "frames" / "frame-0099.jpg").write_bytes(b"stale")
    (output / "frames" / "notes.txt").write_text("keep me", encoding="utf-8")
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Opening line."},
        {"start": 9.0, "end": 11.0, "text": "The important moment."},
    ]

    result = build_evidence_bundle(
        output,
        source="tutorial.mp4",
        title="Tutorial",
        duration_seconds=20.0,
        duration="00:20",
        focus=None,
        detail="balanced",
        frames=[
            {"path": str(first), "timestamp_seconds": 1.0, "reason": "scene-change"},
            {"path": str(second), "timestamp_seconds": 10.0, "reason": "keyframe"},
        ],
        transcript_segments=segments,
        transcript_text="[00:00] Opening line.\n[00:09] The important moment.",
        transcript_source="captions",
    )

    assert result["schema_version"] == 1
    assert result["frame_count"] == 2
    assert result["frames"][1]["transcript_excerpt"] == "The important moment."
    assert (output / "frames" / "frame-0001.jpg").read_bytes() == b"first-frame"
    assert (output / "frames" / "frame-0002.jpg").read_bytes() == b"second-frame"
    assert not (output / "frames" / "frame-0099.jpg").exists()
    assert (output / "frames" / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (output / "transcript.txt").read_text(encoding="utf-8").endswith("\n")

    index = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert index == result
    timeline = (output / "timeline.md").read_text(encoding="utf-8")
    assert "![Frame at 00:10](frames/frame-0002.jpg)" in timeline
    assert "> The important moment." in timeline


def test_build_evidence_bundle_handles_no_frames_or_transcript(tmp_path: Path):
    output = tmp_path / "evidence"

    result = build_evidence_bundle(
        output,
        source="silent.mp4",
        title=None,
        duration_seconds=3.0,
        duration="00:03",
        focus=None,
        detail="transcript",
        frames=[],
        transcript_segments=[],
        transcript_text=None,
        transcript_source=None,
    )

    assert result["frames"] == []
    assert (output / "transcript.txt").read_text(encoding="utf-8") == ""
    timeline = (output / "timeline.md").read_text(encoding="utf-8")
    assert "_No frames were selected._" in timeline
    assert "_No transcript was available for this run._" in timeline


def test_build_evidence_bundle_rejects_file_output(tmp_path: Path):
    output = tmp_path / "evidence"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        build_evidence_bundle(
            output,
            source="video.mp4",
            title=None,
            duration_seconds=0.0,
            duration="00:00",
            focus=None,
            detail="balanced",
            frames=[],
            transcript_segments=[],
            transcript_text=None,
            transcript_source=None,
        )
