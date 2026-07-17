#!/usr/bin/env python3
"""Write a durable, timestamp-aligned evidence bundle for a watch run."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


GENERATED_FRAME_RE = re.compile(r"frame-\d+\.[A-Za-z0-9]+$")


def _format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def transcript_excerpt(
    segments: list[dict],
    timestamp_seconds: float,
    *,
    padding_seconds: float = 2.0,
    nearest_seconds: float = 8.0,
    max_chars: int = 500,
) -> str | None:
    """Return transcript text spoken near a frame timestamp."""
    nearby = [
        segment
        for segment in segments
        if float(segment["start"]) - padding_seconds
        <= timestamp_seconds
        <= float(segment["end"]) + padding_seconds
    ]

    if not nearby and segments:
        def distance(segment: dict) -> float:
            start = float(segment["start"])
            end = float(segment["end"])
            if timestamp_seconds < start:
                return start - timestamp_seconds
            if timestamp_seconds > end:
                return timestamp_seconds - end
            return 0.0

        nearest = min(segments, key=distance)
        if distance(nearest) <= nearest_seconds:
            nearby = [nearest]

    texts: list[str] = []
    for segment in nearby:
        text = str(segment.get("text") or "").strip()
        if text and (not texts or text != texts[-1]):
            texts.append(text)

    if not texts:
        return None

    excerpt = " ".join(texts)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt


def _copy_frames(frames: list[dict], frames_dir: Path) -> list[dict]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for child in frames_dir.iterdir():
        if child.is_file() and GENERATED_FRAME_RE.fullmatch(child.name):
            child.unlink()

    copied: list[dict] = []
    for number, frame in enumerate(frames, start=1):
        source = Path(frame["path"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"selected frame does not exist: {source}")
        suffix = source.suffix.lower() or ".jpg"
        destination = frames_dir / f"frame-{number:04d}{suffix}"
        if source != destination.resolve():
            shutil.copy2(source, destination)
        copied.append({**frame, "evidence_path": destination})
    return copied


def _write_timeline(
    path: Path,
    *,
    source: str,
    title: str | None,
    duration: str,
    focus: dict | None,
    frames: list[dict],
    transcript_available: bool,
) -> None:
    lines = ["# Video Evidence Timeline", ""]
    if title:
        lines.extend([f"**Title:** {title}", ""])
    lines.extend([f"**Source:** {source}", "", f"**Duration:** {duration}", ""])
    if focus:
        lines.extend(
            [
                f"**Focus range:** {focus['start']} to {focus['end']}",
                "",
            ]
        )

    if not frames:
        lines.extend(["_No frames were selected._", ""])

    for frame in frames:
        stamp = frame["timestamp"]
        relative_path = frame["file"]
        lines.extend(
            [
                f"## {stamp}",
                "",
                f"![Frame at {stamp}]({relative_path})",
                "",
                f"**Selection:** {frame['reason']}",
                "",
            ]
        )
        if frame.get("transcript_excerpt"):
            lines.extend([f"> {frame['transcript_excerpt']}", ""])

    lines.extend(["## Transcript", ""])
    if transcript_available:
        lines.extend(["See [`transcript.txt`](transcript.txt) for the timestamped transcript.", ""])
    else:
        lines.extend(["_No transcript was available for this run._", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_evidence_bundle(
    output_dir: str | Path,
    *,
    source: str,
    title: str | None,
    duration_seconds: float,
    duration: str,
    focus: dict | None,
    detail: str,
    frames: list[dict],
    transcript_segments: list[dict],
    transcript_text: str | None,
    transcript_source: str | None,
) -> dict:
    """Copy selected evidence and write a portable index and timeline."""
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"evidence path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)

    copied_frames = _copy_frames(frames, root / "frames")
    indexed_frames: list[dict] = []
    for number, frame in enumerate(copied_frames, start=1):
        timestamp_seconds = float(frame["timestamp_seconds"])
        indexed_frames.append(
            {
                "index": number,
                "file": frame["evidence_path"].relative_to(root).as_posix(),
                "timestamp_seconds": timestamp_seconds,
                "timestamp": _format_time(timestamp_seconds),
                "reason": frame.get("reason", "selected"),
                "transcript_excerpt": transcript_excerpt(
                    transcript_segments,
                    timestamp_seconds,
                ),
            }
        )

    transcript_path = root / "transcript.txt"
    transcript_path.write_text(
        f"{transcript_text.rstrip()}\n" if transcript_text else "",
        encoding="utf-8",
    )

    index = {
        "schema_version": 1,
        "source": source,
        "title": title,
        "duration_seconds": float(duration_seconds),
        "focus": focus,
        "detail": detail,
        "frame_count": len(indexed_frames),
        "transcript_source": transcript_source,
        "frames": indexed_frames,
    }
    (root / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_timeline(
        root / "timeline.md",
        source=source,
        title=title,
        duration=duration,
        focus=focus,
        frames=indexed_frames,
        transcript_available=bool(transcript_text),
    )
    return index
