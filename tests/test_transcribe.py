"""WebVTT parsing: cue tags, HTML entities, whitespace normalization, dedup."""
from __future__ import annotations

from pathlib import Path

import transcribe


def _write(tmp_path: Path, body: str) -> str:
    path = tmp_path / "subs.vtt"
    path.write_text("WEBVTT\nKind: captions\nLanguage: en\n\n" + body, encoding="utf-8")
    return str(path)


# --- _clean ------------------------------------------------------------------

def test_clean_strips_cue_tags():
    assert transcribe._clean("<c.colorE5E5E5>hello</c> world") == "hello world"


def test_clean_decodes_html_entities():
    assert transcribe._clean("dogmas &amp; sacred texts") == "dogmas & sacred texts"
    assert transcribe._clean("5 &lt; 6") == "5 < 6"


def test_clean_folds_nbsp_into_a_single_space():
    # YouTube pads wrapped caption lines with &nbsp;, which must not survive
    # into the transcript as a literal entity or a U+00A0.
    cleaned = transcribe._clean("is not&nbsp; a religion in the conventional&nbsp;&nbsp;")
    assert cleaned == "is not a religion in the conventional"
    assert "&nbsp;" not in cleaned
    assert " " not in cleaned


def test_clean_drops_zero_width_spaces():
    assert transcribe._clean("a​b") == "ab"


def test_clean_collapses_runs_of_whitespace():
    assert transcribe._clean("a   \t b\n") == "a b"


# --- parse_vtt ---------------------------------------------------------------

def test_parse_vtt_yields_clean_timestamped_segments(tmp_path):
    path = _write(
        tmp_path,
        "00:00:09.000 --> 00:00:12.000\n"
        "The religion of the Buddha is not&nbsp; a religion&nbsp;&nbsp;\n"
        "\n"
        "00:00:12.000 --> 00:00:18.000\n"
        "<c>because it lacks dogmas &amp; sacred texts.</c>&nbsp;\n",
    )
    segments = transcribe.parse_vtt(path)

    assert [s["text"] for s in segments] == [
        "The religion of the Buddha is not a religion",
        "because it lacks dogmas & sacred texts.",
    ]
    assert segments[0]["start"] == 9.0
    assert segments[0]["end"] == 12.0


def test_parse_vtt_joins_wrapped_lines_without_double_spaces(tmp_path):
    path = _write(
        tmp_path,
        "00:00:00.000 --> 00:00:04.000\n"
        "first line&nbsp;\n"
        "&nbsp;second line\n",
    )
    assert transcribe.parse_vtt(path)[0]["text"] == "first line second line"


def test_parse_vtt_skips_cue_lines_that_are_only_entities(tmp_path):
    path = _write(
        tmp_path,
        "00:00:00.000 --> 00:00:04.000\n"
        "&nbsp;\n"
        "real text\n",
    )
    assert transcribe.parse_vtt(path)[0]["text"] == "real text"


def test_parse_vtt_dedupes_rolling_duplicates(tmp_path):
    path = _write(
        tmp_path,
        "00:00:00.000 --> 00:00:02.000\nhello\n"
        "\n"
        "00:00:02.000 --> 00:00:04.000\nhello\n",
    )
    segments = transcribe.parse_vtt(path)
    assert len(segments) == 1
    assert segments[0]["end"] == 4.0


def test_parse_vtt_merges_rolling_prefix_growth(tmp_path):
    path = _write(
        tmp_path,
        "00:00:00.000 --> 00:00:02.000\nhello\n"
        "\n"
        "00:00:02.000 --> 00:00:04.000\nhello world\n",
    )
    segments = transcribe.parse_vtt(path)
    assert len(segments) == 1
    assert segments[0]["text"] == "hello world"


def test_parse_vtt_dedupe_matches_across_nbsp_padding(tmp_path):
    # Same line, one padded with &nbsp;: normalization must let dedup catch it.
    path = _write(
        tmp_path,
        "00:00:00.000 --> 00:00:02.000\nhello world\n"
        "\n"
        "00:00:02.000 --> 00:00:04.000\nhello&nbsp; world\n",
    )
    assert len(transcribe.parse_vtt(path)) == 1
