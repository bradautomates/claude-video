"""Transcript formatting: timestamp stamps stay aligned with frame markers."""
from __future__ import annotations

import transcribe


def test_format_transcript_under_an_hour_uses_mm_ss():
    segs = [{"start": 5.0, "end": 7.0, "text": "hi"}, {"start": 65.0, "end": 68.0, "text": "later"}]
    assert transcribe.format_transcript(segs) == "[00:05] hi\n[01:05] later"


def test_format_transcript_rolls_over_to_hours():
    """Past an hour the stamp must carry an hours field, not overflow minutes."""
    segs = [{"start": 3661.0, "end": 3665.0, "text": "past one hour"}]
    assert transcribe.format_transcript(segs) == "[1:01:01] past one hour"


def test_format_stamp_matches_frame_marker_shape():
    """A transcript stamp lines up with the frames.format_time `t=` marker so the
    two evidence streams reference the same clock on long videos."""
    from frames import format_time

    for seconds in (0.0, 59.0, 60.0, 3599.0, 3600.0, 7325.0):
        assert transcribe._format_stamp(seconds) == f"[{format_time(int(seconds))}]"
