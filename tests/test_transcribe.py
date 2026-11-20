"""VTT syntax, independent repeated speech, and source segment validation."""
import pytest
from transcribe import parse_vtt, normalize_segments


def parse(tmp_path, body):
    path = tmp_path / 'captions.vtt'
    path.write_bytes(body.encode('utf-8'))  # Exact bytes: text mode on Windows would turn \r\n into \r\r\n.
    return parse_vtt(str(path))


def test_blocks_ids_settings_padding_and_missing_separator(tmp_path):
    out = parse(tmp_path, '\ufeffWEBVTT\r\n\r\nNOTE comment\r\n00:00.000 --> 00:01.000\r\nskip\r\n\r\nSTYLE\r\n::cue {}\r\n\r\nREGION\r\nid:one\r\n\r\ncue-id\r\n00:01.000 --> 00:02.000 align:start\r\nfirst\r\n   \r\n<c>last</c>\r\n00:03.000 --> 00:04.000\r\nnext\r\n')
    assert [s['text'] for s in out] == ['first  last', 'next']


def test_valid_long_hours_and_invalid_minutes(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n100:02:03.004 --> 100:02:04.000\nx\n\n00:60.000 --> 01:01.000\ninvalid\n')
    assert len(out) == 1 and out[0]['start'] == 360123.004


def test_overlapping_prefix_only_and_mixed_karaoke(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:01.000 --> 00:03.000\nThank you\n\n00:02.000 --> 00:04.000\nThank you all\n\n00:04.000 --> 00:05.000\nThank you all\n\n00:05.000 --> 00:06.000\nplain <00:05.100><c>spoken</c> ending &lt;literal&gt;\n')
    assert [s['text'] for s in out] == ['Thank you all', 'Thank you all', 'plain spoken ending <literal>']


@pytest.mark.parametrize('segment', [
    {'start': float('nan'), 'end': 1, 'text': 'hi'}, {'start': 2, 'end': 1, 'text': 'hi'},
    {'start': True, 'end': 1, 'text': 'hi'}, {'start': 0, 'end': 1, 'text': ' '},
    {'start': 0, 'end': float('inf'), 'text': 'hi'},
])
def test_rejects_invalid_segments(segment):
    with pytest.raises((ValueError, TypeError)):
        normalize_segments({'segments': [segment]})


def test_overlapping_separate_display_regions_are_not_merged(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:00.000 --> 00:02.000 line:10%\nHello\n\n00:01.000 --> 00:03.000 line:90%\nHello\n')
    assert len(out) == 2
    assert all(set(s) == {'start', 'end', 'text'} for s in out)


# YouTube auto-captions: each cue shows the line before and adds a timed line; a
# short cue in between repeats the finished line on its own.
ROLLING = (
    'WEBVTT\nKind: captions\nLanguage: en\n\n'
    '00:00:04.280 --> 00:00:06.830 align:start position:0%\n \n'
    'This<00:00:04.760><c> is</c><00:00:04.960><c> a</c><00:00:05.000><c> three</c>\n\n'
    '00:00:06.830 --> 00:00:06.840 align:start position:0%\nThis is a three\n \n\n'
    '00:00:06.840 --> 00:00:08.350 align:start position:0%\nThis is a three\n'
    'and<00:00:07.040><c> rendered</c><00:00:07.360><c> at</c><00:00:07.480><c> low</c>\n\n'
    '00:00:08.350 --> 00:00:08.360 align:start position:0%\nand rendered at low\n \n\n'
    '00:00:08.360 --> 00:00:11.230 align:start position:0%\nand rendered at low\n'
    'and<00:00:09.000><c> rendered</c><00:00:09.400><c> at</c><00:00:09.900><c> low</c>\n\n'
    '00:00:11.230 --> 00:00:11.240 align:start position:0%\nand rendered at low\n \n\n'
    '00:00:11.240 --> 00:00:13.000 align:start position:0%\nand rendered at low\n[Music]\n'
)


def test_rolling_auto_captions_emit_each_line_once(tmp_path):
    out = parse(tmp_path, ROLLING)
    # The third line repeats the second on purpose: repeated speech stays.
    assert [s['text'] for s in out] == ['This is a three', 'and rendered at low', 'and rendered at low', '[Music]']
    assert [s['start'] for s in out] == [4.28, 6.84, 8.36, 11.24]
    assert out[0]['end'] == 6.84 and out[2]['end'] == 11.24


def test_manual_captions_keep_a_repeated_first_line(tmp_path):
    out = parse(tmp_path, 'WEBVTT\n\n00:01.000 --> 00:02.000\nLa la la\n\n00:02.000 --> 00:04.000\nLa la la\nla la la la\n')
    assert [s['text'] for s in out] == ['La la la', 'La la la la la la la']

