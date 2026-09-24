"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path

import frames
import pytest


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


def test_uniform_pixels_match_source_time(tmp_path):
    """A changing fill exposes the old fps half-interval label error."""
    import subprocess
    from conftest import build_cut_clip
    clip = tmp_path / 'clock.mp4'
    build_cut_clip(clip, n=12, seg=1, fps=10)
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=4)
    expected = frames.extract_at_timestamps(str(clip), tmp_path / 'truth', [f['timestamp_seconds'] for f in out])[0]
    def pixels(path):
        return subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', 'scale=1:1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    assert [f['timestamp_seconds'] for f in out] == [0, 3, 6, 9]
    assert all(frames._frame_delta(pixels(a['path']), pixels(b['path'])) < 2 for a, b in zip(out, expected))


def test_fractional_seek_and_cap_use_actual_span(tmp_path):
    from conftest import build_static_clip
    clip = tmp_path / 'fps.mp4'
    build_static_clip(clip, duration=6, fps='30000/1001')
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=3, start_seconds=0.151, end_seconds=100)
    assert len(out) == 3 and out[-1]['timestamp_seconds'] > 4
    assert out[0]['timestamp_seconds'] >= 0.151
    for frame in out:
        assert abs(frame['timestamp_seconds'] * 30000 / 1001 - round(frame['timestamp_seconds'] * 30000 / 1001)) < 0.01


def test_nonzero_input_pts_are_source_relative(tmp_path):
    import subprocess
    clip = tmp_path / 'offset.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x120:r=10:d=3', '-vf', 'setpts=PTS+7/TB', '-c:v', 'libx264', str(clip)], check=True)
    out = frames.extract(str(clip), tmp_path / 'f', fps=1, max_frames=3)
    assert [f['timestamp_seconds'] for f in out] == [0, 1, 2]
    # FFmpeg 7.1 muxes this shifted clip one frame (0.1s) short; either is source-relative, 10 would not be.
    assert 2.9 <= frames.get_metadata(str(clip))['duration_seconds'] <= 3


@pytest.mark.parametrize('fps', [float('nan'), float('inf'), -1, 0])
def test_invalid_fps_rejected_before_extraction(static_clip, tmp_path, fps):
    with pytest.raises(SystemExit, match='fps'):
        frames.extract(str(static_clip), tmp_path / 'f', fps=fps)


def test_gap_with_corrupt_input_still_fails(tmp_path):
    path = tmp_path / 'bad.mp4'
    path.write_bytes(b'not media')
    with pytest.raises(SystemExit):
        frames.extract_keyframes(str(path), tmp_path / 'f', start_seconds=1, end_seconds=2)


def test_variable_rate_timestamps_are_actual_source_frames(tmp_path):
    import json
    import subprocess
    clip = tmp_path / 'vfr.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=160x120:r=30:d=6',
                    '-vf', "select='if(lt(t,3),not(mod(n,3)),not(mod(n,7)))'", *frames.sync_args(), '-c:v', 'libx264', str(clip)], check=True)
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(clip)], capture_output=True, text=True, check=True)
    source_times = [float(f['best_effort_timestamp_time']) for f in json.loads(result.stdout)['frames']]
    out = frames.extract(str(clip), tmp_path / 'f', fps=2, max_frames=4, start_seconds=0.17)
    assert len(out) == 4 and out[-1]['timestamp_seconds'] > 4
    assert all(any(abs(f['timestamp_seconds'] - t) < 0.002 for t in source_times) for f in out)


def test_single_frame_cap_keeps_first_real_frame(static_clip, tmp_path):
    out = frames.extract(str(static_clip), tmp_path / 'f', fps=2, max_frames=1)
    assert len(out) == 1 and out[0]['timestamp_seconds'] == 0


def test_uniform_candidate_count_includes_deduplicated_frames(static_clip, tmp_path):
    out, meta = frames.extract_scene_or_uniform(str(static_clip), tmp_path / 'f', fps=2, target_frames=6)
    assert meta['candidate_count'] == len(out) + meta['deduped_count']
