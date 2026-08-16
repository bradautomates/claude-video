"""VTT parsing: YouTube rolling-caption dedup."""
from __future__ import annotations

from pathlib import Path

import transcribe

# Verbatim shape of a YouTube auto-sub VTT: each timed cue is
# [previous line] + [new line with word timings], interleaved with 10ms
# "hold" cues that repeat the previous line alone.
YT_ROLLOVER = """WEBVTT
Kind: captions
Language: en

00:00:00.240 --> 00:00:01.910 align:start position:0%

There<00:00:00.480><c> are</c><00:00:00.640><c> people</c>

00:00:01.910 --> 00:00:01.920 align:start position:0%
There are people


00:00:01.920 --> 00:00:04.309 align:start position:0%
There are people
agent<00:00:02.399><c> workforces</c>

00:00:04.309 --> 00:00:04.319 align:start position:0%
agent workforces


00:00:04.319 --> 00:00:06.550 align:start position:0%
agent workforces
and<00:00:04.560><c> sub</c><00:00:04.799><c> agents</c>

00:00:06.550 --> 00:00:06.560 align:start position:0%
and sub agents

"""


def test_youtube_rollover_lands_each_line_once(tmp_path: Path):
    p = tmp_path / "video.en.vtt"
    p.write_text(YT_ROLLOVER, encoding="utf-8")
    segs = transcribe.parse_vtt(str(p))
    texts = [s["text"] for s in segs]
    assert texts == ["There are people", "agent workforces", "and sub agents"]
    # every word appears exactly once across the transcript
    joined = " ".join(texts).split()
    assert len(joined) == len(set(joined))


def test_strip_rollover_partial_and_none():
    assert transcribe._strip_rollover(["a", "b"], ["b", "c"]) == ["c"]
    assert transcribe._strip_rollover(["a", "b"], ["a", "b", "c"]) == ["c"]
    assert transcribe._strip_rollover(["a", "b"], ["x", "y"]) == ["x", "y"]
    assert transcribe._strip_rollover([], ["x"]) == ["x"]
