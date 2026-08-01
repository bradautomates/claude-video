#!/usr/bin/env python3
"""Local temporal video and non-speech audio analysis for short clips.

The language model sees the results as compact visual artifacts and timestamped
measurements:

- overview and adaptive sub-second motion strips preserve frame-to-frame order
  without spending one image input per sampled frame;
- changed-pixel coverage plus persistence separates cut/flash-like peaks from
  localized or sustained movement candidates;
- waveform and spectrogram images expose rhythm and frequency content;
- silence, loudness, spectral flux, and low/mid/high-band novelty expose sound
  beats that a speech transcript necessarily omits.

Everything runs through ffmpeg and the Python standard library. The numeric
signals are intentionally described as candidates: a pixel-difference spike can
be a cut, a flash, or fast motion, and audio statistics cannot name an exact
sound source on their own.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ANALYSIS_VERSION = 3
AUTO_ANALYSIS_MAX_SECONDS = 60.0
MOTION_SAMPLE_FPS = 10.0
MOTION_STRIP_FRAMES = 8
MAX_MOTION_STRIPS = 15
MAX_ZOOM_STRIPS = 8
ZOOM_STRIP_SECONDS = 0.9
DIFF_BLACK_THRESHOLD = 24
SILENCE_THRESHOLD_DB = -42
SILENCE_MIN_SECONDS = 0.12


def should_analyze(mode: str, duration_seconds: float) -> bool:
    """Resolve --analysis. Auto is intentionally optimized for short form."""
    if mode == "cinematic":
        return True
    if mode == "standard":
        return False
    return 0 < duration_seconds <= AUTO_ANALYSIS_MAX_SECONDS


def format_precise(seconds: float) -> str:
    """Format a timestamp without rounding away short-form timing."""
    seconds = max(0.0, seconds)
    whole = int(seconds)
    hundredths = int(round((seconds - whole) * 100))
    if hundredths == 100:
        whole += 1
        hundredths = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}.{hundredths:02d}"
    return f"{minutes:02d}:{secs:02d}.{hundredths:02d}"


def plan_strip_windows(
    start: float,
    end: float,
    max_strips: int = MAX_MOTION_STRIPS,
) -> list[tuple[float, float]]:
    """Split a range into compact windows, denser for very short clips."""
    duration = end - start
    if duration <= 0 or max_strips < 1:
        return []
    if duration <= 5:
        ideal_window = 1.0
    elif duration <= 15:
        ideal_window = 1.5
    elif duration <= 30:
        ideal_window = 2.0
    else:
        ideal_window = 3.0
    window = max(ideal_window, duration / max_strips)
    count = max(1, min(max_strips, math.ceil(duration / window)))
    # Equal windows avoid a tiny final strip that may not contain enough frames
    # for ffmpeg's tile filter to flush.
    return [
        (start + duration * i / count, start + duration * (i + 1) / count)
        for i in range(count)
    ]


def _active_span(
    points: list[dict],
    index: int,
    key: str,
    threshold: float,
) -> tuple[int, int]:
    """Return the hysteresis span around a peak while change stays elevated."""
    lo = index
    hi = index
    while lo > 0 and points[lo - 1].get(key, float("-inf")) >= threshold:
        lo -= 1
    while hi + 1 < len(points) and points[hi + 1].get(key, float("-inf")) >= threshold:
        hi += 1
    return lo, hi


def classify_visual_changes(
    points: list[dict],
    median_motion: float | None,
    p90_motion: float | None,
    limit: int = 10,
) -> list[dict]:
    """Classify strong local peaks as likely discontinuity or sustained motion.

    Difference magnitude alone over-fires on fast action. Changed-pixel coverage
    and a low hysteresis threshold tell us whether a peak is isolated across most
    of the frame (cut/flash-like), repeated briefly (impact/flash), or part of a
    sustained run (camera/subject motion or a transition). Labels remain
    candidates because a full-screen flash can be indistinguishable from a cut.
    """
    if not points:
        return []
    baseline = median_motion or 0.0
    upper = p90_motion if p90_motion is not None else baseline
    peak_threshold = max(0.5, upper)
    hysteresis = max(0.5, baseline + max(1.0, (upper - baseline) * 0.35))
    peaks = _pick_peaks(
        points,
        "motion",
        limit=max(limit * 3, limit),
        min_gap=0.2,
        minimum=peak_threshold,
        local_only=True,
    )
    by_time = {point["time"]: index for index, point in enumerate(points)}
    classified: list[dict] = []
    for peak in peaks:
        index = by_time.get(peak["time"])
        if index is None:
            continue
        lo, hi = _active_span(points, index, "motion", hysteresis)
        span_frames = hi - lo + 1
        coverage = float(peak.get("changed_percent") or 0.0)
        high_coverage_frames = sum(
            1 for point in points[lo:hi + 1]
            if float(point.get("changed_percent") or 0.0) >= 45.0
        )
        if coverage < 35.0:
            kind = (
                "localized_sustained_motion"
                if span_frames >= 4 else "localized_motion"
            )
        elif span_frames <= 2 and high_coverage_frames <= 1:
            kind = "cut_or_flash"
        elif span_frames <= 3 and high_coverage_frames >= 2:
            kind = "flash_or_impact"
        else:
            kind = "sustained_motion_or_transition"

        strength = min(1.0, float(peak["motion"]) / max(upper, 0.5))
        coverage_strength = min(1.0, coverage / 60.0)
        isolation = 1.0 / max(1.0, span_frames / 2.0)
        confidence = 0.25 + 0.3 * strength + 0.3 * coverage_strength + 0.15 * isolation
        classified.append({
            "time": peak["time"],
            "score": round(float(peak["motion"]), 2),
            "changed_percent": round(coverage, 1),
            "kind": kind,
            "confidence": round(min(0.99, confidence), 2),
            "active_frames": span_frames,
            "_span_start": lo,
            "_span_end": hi,
        })

    # Local maxima inside one continuous action should be one event, not a
    # stuttering list of near-identical timestamps. Preserve isolated cuts and
    # flashes even when they happen inside a generally active sequence.
    deduped: list[dict] = []
    mergeable_kinds = {"localized_sustained_motion", "sustained_motion_or_transition"}
    for item in classified:
        match_index = next(
            (
                index for index, existing in enumerate(deduped)
                if item["kind"] == existing["kind"]
                and item["kind"] in mergeable_kinds
                and item["_span_start"] <= existing["_span_end"]
                and item["_span_end"] >= existing["_span_start"]
            ),
            None,
        )
        if match_index is None:
            deduped.append(item)
        elif item["score"] > deduped[match_index]["score"]:
            deduped[match_index] = item

    ranked = sorted(
        deduped,
        key=lambda item: item["score"] * (0.65 + item["changed_percent"] / 100),
        reverse=True,
    )[:limit]
    for item in ranked:
        item.pop("_span_start", None)
        item.pop("_span_end", None)
    return sorted(ranked, key=lambda item: item["time"])


def plan_zoom_windows(
    events: list[dict],
    start: float,
    end: float,
    max_zooms: int = MAX_ZOOM_STRIPS,
    window_seconds: float = ZOOM_STRIP_SECONDS,
) -> list[dict]:
    """Plan non-overlapping sub-second strips around the strongest events."""
    duration = end - start
    if duration <= window_seconds or not events or max_zooms < 1:
        return []
    ranked = sorted(
        events,
        key=lambda item: (
            float(item.get("score") or 0)
            * (0.7 + float(item.get("changed_percent") or 0) / 100)
            * (1.1 if item.get("kind") in {"cut_or_flash", "flash_or_impact"} else 1.0)
        ),
        reverse=True,
    )
    selected: list[dict] = []
    half = window_seconds / 2
    for event in ranked:
        center = min(end, max(start, float(event.get("time") or start)))
        win_start = max(start, center - half)
        win_end = min(end, win_start + window_seconds)
        win_start = max(start, win_end - window_seconds)
        if any(not (win_end <= item["start"] or win_start >= item["end"]) for item in selected):
            continue
        selected.append({
            "start": round(win_start, 3),
            "end": round(win_end, 3),
            "focus_time": round(center, 3),
            "kind": event.get("kind") or "visual_change",
            "confidence": event.get("confidence"),
        })
        if len(selected) >= max_zooms:
            break
    return sorted(selected, key=lambda item: item["start"])


def _scope_args(start: float, end: float) -> list[str]:
    args: list[str] = []
    if start > 0:
        args += ["-ss", f"{start:.3f}"]
    args += ["-t", f"{max(0.001, end - start):.3f}"]
    return args


def _run_ffmpeg(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().replace("\n", " ")[:300]
        raise RuntimeError(f"{label} failed: {detail or f'ffmpeg exit {result.returncode}'}")
    return result


def _parse_frame_metadata(text: str, keys: dict[str, str], offset: float = 0.0) -> list[dict]:
    """Parse ffmpeg metadata=print output into timestamped measurements."""
    points: list[dict] = []
    current: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("frame:") and "pts_time:" in line:
            if current is not None:
                points.append(current)
            try:
                pts = float(line.rsplit("pts_time:", 1)[1].split()[0])
            except (ValueError, IndexError):
                current = None
            else:
                current = {"time": round(offset + pts, 3)}
            continue
        if current is None or "=" not in line:
            continue
        raw_key, raw_value = line.split("=", 1)
        output_key = keys.get(raw_key)
        if not output_key:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            value = float("-inf") if raw_value == "-inf" else float("nan")
        current[output_key] = value
    if current is not None:
        points.append(current)
    return points


def _percentile(values: list[float], fraction: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def _pick_peaks(
    points: list[dict],
    key: str,
    limit: int = 8,
    min_gap: float = 0.25,
    minimum: float | None = None,
    local_only: bool = False,
) -> list[dict]:
    candidates: list[dict] = []
    for index, point in enumerate(points):
        value = point.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        if minimum is not None and value < minimum:
            continue
        if local_only:
            previous = points[index - 1].get(key) if index > 0 else None
            following = points[index + 1].get(key) if index + 1 < len(points) else None
            if isinstance(previous, (int, float)) and math.isfinite(previous) and value < previous:
                continue
            if isinstance(following, (int, float)) and math.isfinite(following) and value <= following:
                continue
        candidates.append(point)
    ranked = sorted(candidates, key=lambda point: point[key], reverse=True)
    selected: list[dict] = []
    for point in ranked:
        if all(abs(point["time"] - other["time"]) >= min_gap for other in selected):
            selected.append(dict(point))
            if len(selected) >= limit:
                break
    return sorted(selected, key=lambda point: point["time"])


def _make_motion_strip(
    video_path: str,
    out_path: Path,
    start: float,
    end: float,
    tile_width: int,
) -> dict | None:
    duration = max(0.001, end - start)
    sample_fps = MOTION_STRIP_FRAMES / duration
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_scope_args(start, end),
        "-i", video_path,
        "-an",
        "-vf",
        (
            f"fps={sample_fps:.6f},scale={tile_width}:-2,"
            f"tile={MOTION_STRIP_FRAMES}x1:nb_frames={MOTION_STRIP_FRAMES}:"
            "padding=3:margin=4:color=black"
        ),
        "-frames:v", "1",
        str(out_path),
    ]
    try:
        _run_ffmpeg(command, "motion strip extraction")
    except RuntimeError as exc:
        print(f"[watch] {exc}", file=sys.stderr)
        return None
    if not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return {
        "path": str(out_path),
        "start": round(start, 3),
        "end": round(end, 3),
        "samples": MOTION_STRIP_FRAMES,
    }


def _analyze_video(
    video_path: str,
    out_dir: Path,
    start: float,
    end: float,
    resolution: int,
) -> dict:
    errors: list[str] = []
    points: list[dict] = []
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *_scope_args(start, end),
        "-i", video_path,
        "-an",
        "-vf",
        (
            f"fps={MOTION_SAMPLE_FPS:g},format=gray,"
            "tblend=all_mode=difference,signalstats,"
            f"blackframe=amount=0:threshold={DIFF_BLACK_THRESHOLD},"
            "metadata=print:file=-"
        ),
        "-f", "null", "-",
    ]
    try:
        result = _run_ffmpeg(command, "motion measurement")
        points = _parse_frame_metadata(
            result.stdout,
            {
                "lavfi.signalstats.YAVG": "motion",
                "lavfi.blackframe.pblack": "pblack",
            },
            offset=start,
        )
        for point in points:
            pblack = point.get("pblack")
            if isinstance(pblack, (int, float)) and math.isfinite(pblack):
                point["changed_percent"] = max(0.0, min(100.0, 100.0 - pblack))
    except RuntimeError as exc:
        errors.append(str(exc))

    motion_values = [point["motion"] for point in points if math.isfinite(point.get("motion", math.nan))]
    median_motion = statistics.median(motion_values) if motion_values else None
    p90_motion = _percentile(motion_values, 0.9)
    change_peaks = classify_visual_changes(points, median_motion, p90_motion)

    strips_dir = out_dir / "motion_strips"
    strips_dir.mkdir(parents=True, exist_ok=True)
    windows = plan_strip_windows(start, end)
    tile_width = max(140, min(260, resolution // 2))

    def create(item: tuple[int, tuple[float, float]]) -> dict | None:
        index, (win_start, win_end) = item
        return _make_motion_strip(
            video_path,
            strips_dir / f"motion_{index:03d}.jpg",
            win_start,
            win_end,
            tile_width,
        )

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(windows)))) as pool:
        strips = [strip for strip in pool.map(create, enumerate(windows)) if strip is not None]

    duration = end - start
    busy_motion = bool(
        duration > 15
        and median_motion is not None
        and p90_motion is not None
        and p90_motion >= max(median_motion * 2.2, median_motion + 8.0)
    )
    zoom_windows = plan_zoom_windows(change_peaks, start, end) if busy_motion else []

    def create_zoom(item: tuple[int, dict]) -> dict | None:
        index, window = item
        strip = _make_motion_strip(
            video_path,
            strips_dir / f"zoom_{index:03d}.jpg",
            window["start"],
            window["end"],
            tile_width,
        )
        if strip is not None:
            strip.update({
                "kind": "adaptive_zoom",
                "focus_time": window["focus_time"],
                "event_kind": window["kind"],
                "confidence": window.get("confidence"),
            })
        return strip

    if zoom_windows:
        with ThreadPoolExecutor(max_workers=min(4, len(zoom_windows))) as pool:
            zoom_strips = [
                strip for strip in pool.map(create_zoom, enumerate(zoom_windows))
                if strip is not None
            ]
    else:
        zoom_strips = []

    return {
        "sample_fps": MOTION_SAMPLE_FPS,
        "median_motion": round(median_motion, 2) if median_motion is not None else None,
        "p90_motion": round(p90_motion, 2) if p90_motion is not None else None,
        "change_peaks": change_peaks,
        "strips": strips,
        "zoom_strips": zoom_strips,
        "errors": errors,
    }


SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def _parse_silences(log: str, offset: float, end: float) -> list[dict]:
    ranges: list[dict] = []
    open_start: float | None = None
    for line in log.splitlines():
        start_match = SILENCE_START_RE.search(line)
        if start_match:
            open_start = offset + float(start_match.group(1))
        end_match = SILENCE_END_RE.search(line)
        if end_match and open_start is not None:
            silence_end = min(end, offset + float(end_match.group(1)))
            if silence_end > open_start:
                ranges.append({
                    "start": round(open_start, 3),
                    "end": round(silence_end, 3),
                    "duration": round(silence_end - open_start, 3),
                })
            open_start = None
    if open_start is not None and end > open_start:
        ranges.append({
            "start": round(open_start, 3),
            "end": round(end, 3),
            "duration": round(end - open_start, 3),
        })
    return ranges


def _sound_event_label(point: dict, median_rms: float | None) -> str:
    rms = point.get("rms_db")
    centroid = point.get("centroid_hz")
    flatness = point.get("flatness")
    if not isinstance(rms, (int, float)) or not math.isfinite(rms) or rms < -80:
        level = "near-silent"
    elif median_rms is not None and rms >= median_rms + 6:
        level = "prominent"
    else:
        level = "audible"
    if not isinstance(centroid, (int, float)) or not math.isfinite(centroid):
        band = "unknown-frequency"
    elif centroid < 600:
        band = "low-frequency"
    elif centroid < 2500:
        band = "mid-frequency"
    else:
        band = "high-frequency"
    if not isinstance(flatness, (int, float)) or not math.isfinite(flatness):
        texture = ""
    elif flatness < 0.08:
        texture = " tonal"
    elif flatness > 0.3:
        texture = " noise-like"
    else:
        texture = " mixed-texture"
    return f"{level} {band}{texture} transient candidate"


def _write_audio_picture(
    video_path: str,
    out_path: Path,
    start: float,
    end: float,
    filter_graph: str,
    label: str,
) -> str | None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *_scope_args(start, end),
        "-i", video_path,
        "-lavfi", filter_graph,
        "-frames:v", "1",
        str(out_path),
    ]
    try:
        _run_ffmpeg(command, label)
    except RuntimeError as exc:
        print(f"[watch] {exc}", file=sys.stderr)
        return None
    if not out_path.exists() or out_path.stat().st_size == 0:
        return None
    return str(out_path)


def _measure_band_rms(
    video_path: str,
    start: float,
    end: float,
    band_filter: str,
) -> list[dict]:
    """Measure short-time RMS after isolating one broad frequency band."""
    stats_filter = (
        f"aresample=16000,{band_filter},"
        "astats=metadata=1:reset=1:measure_perchannel=none:"
        "measure_overall=RMS_level,"
        "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *_scope_args(start, end),
        "-i", video_path,
        "-vn", "-af", stats_filter,
        "-f", "null", "-",
    ]
    result = _run_ffmpeg(command, "frequency-band audio measurement")
    return _parse_frame_metadata(
        result.stdout,
        {"lavfi.astats.Overall.RMS_level": "rms_db"},
        offset=start,
    )


def _band_novelty(points: list[dict], history: int = 4) -> list[dict]:
    """Return positive dB rises against a rolling pre-event baseline."""
    novelty: list[dict] = []
    recent: list[float] = []
    for point in points:
        rms = point.get("rms_db")
        if not isinstance(rms, (int, float)) or not math.isfinite(rms) or rms <= -120:
            continue
        baseline = statistics.median(recent[-history:]) if recent else rms
        novelty.append({
            "time": point["time"],
            "rms_db": rms,
            "rise_db": max(0.0, rms - baseline),
        })
        recent.append(rms)
    return novelty


def _detect_band_onsets(
    band_points: dict[str, list[dict]],
    limit: int = 12,
) -> list[dict]:
    """Detect and merge low/mid/high-band energy onsets.

    Full-band spectral flux can miss an effect layered over continuous music.
    Looking for positive novelty inside three broad bands exposes bass hits,
    midrange impacts, and high-frequency swishes independently.
    """
    candidates: list[dict] = []
    for band, points in band_points.items():
        novelty = _band_novelty(points)
        positive = [point["rise_db"] for point in novelty if point["rise_db"] > 0]
        threshold = max(2.5, _percentile(positive, 0.9) or 0.0)
        peaks = _pick_peaks(
            novelty,
            "rise_db",
            limit=max(limit * 2, limit),
            min_gap=0.18,
            minimum=threshold,
            local_only=True,
        )
        for peak in peaks:
            candidates.append({
                "time": peak["time"],
                "band": band,
                "rise_db": peak["rise_db"],
                "threshold_db": threshold,
            })

    clusters: list[list[dict]] = []
    for candidate in sorted(candidates, key=lambda item: item["time"]):
        if clusters and candidate["time"] - clusters[-1][-1]["time"] <= 0.12:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])

    merged: list[dict] = []
    for cluster in clusters:
        per_band: dict[str, dict] = {}
        for candidate in cluster:
            band = candidate["band"]
            if band not in per_band or candidate["rise_db"] > per_band[band]["rise_db"]:
                per_band[band] = candidate
        strongest = max(per_band.values(), key=lambda item: item["rise_db"])
        bands = [band for band in ("low", "mid", "high") if band in per_band]
        if len(bands) == 3:
            label = "broadband onset candidate"
        elif len(bands) == 2:
            label = "multi-band onset candidate"
        else:
            label = f"{bands[0]}-frequency onset candidate"
        relative_strength = max(
            item["rise_db"] / max(item["threshold_db"], 0.1)
            for item in per_band.values()
        )
        confidence = min(
            0.98,
            0.48 + 0.12 * len(bands) + 0.16 * min(2.0, relative_strength - 1.0),
        )
        merged.append({
            "time": round(strongest["time"], 3),
            "bands": bands,
            "rise_db": round(strongest["rise_db"], 2),
            "label": label,
            "confidence": round(max(0.5, confidence), 2),
        })

    ranked = sorted(
        merged,
        key=lambda item: item["rise_db"] * (1 + 0.2 * len(item["bands"])),
        reverse=True,
    )[:limit]
    return sorted(ranked, key=lambda item: item["time"])


def _analyze_audio(video_path: str, out_dir: Path, start: float, end: float) -> dict:
    errors: list[str] = []
    points: list[dict] = []
    stats_filter = (
        "aresample=16000,"
        "aspectralstats=win_size=1024:overlap=0:measure=centroid+flux+flatness,"
        "astats=metadata=1:reset=1:measure_perchannel=none:"
        "measure_overall=RMS_level+Peak_level,"
        "ametadata=print:file=-"
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        *_scope_args(start, end),
        "-i", video_path,
        "-vn", "-af", stats_filter,
        "-f", "null", "-",
    ]
    try:
        result = _run_ffmpeg(command, "audio measurement")
        points = _parse_frame_metadata(
            result.stdout,
            {
                "lavfi.astats.Overall.RMS_level": "rms_db",
                "lavfi.astats.Overall.Peak_level": "peak_db",
                "lavfi.aspectralstats.1.centroid": "centroid_hz",
                "lavfi.aspectralstats.1.flux": "flux",
                "lavfi.aspectralstats.1.flatness": "flatness",
            },
            offset=start,
        )
    except RuntimeError as exc:
        errors.append(str(exc))

    finite_rms = [
        point["rms_db"] for point in points
        if math.isfinite(point.get("rms_db", math.nan)) and point["rms_db"] > -120
    ]
    median_rms = statistics.median(finite_rms) if finite_rms else None
    p10_rms = _percentile(finite_rms, 0.1)
    p90_rms = _percentile(finite_rms, 0.9)
    dynamic_range = (
        p90_rms - p10_rms
        if p10_rms is not None and p90_rms is not None
        else None
    )
    energy_peaks = _pick_peaks(
        points,
        "rms_db",
        minimum=(median_rms + 3) if median_rms is not None else None,
        local_only=True,
    )
    finite_flux = [
        point["flux"] for point in points
        if math.isfinite(point.get("flux", math.nan))
    ]
    median_flux = statistics.median(finite_flux) if finite_flux else None
    p95_flux = _percentile(finite_flux, 0.95)
    flux_threshold = max(
        0.0001,
        p95_flux or 0.0,
        (median_flux or 0.0) * 8,
    )
    audible_floor = max(-80.0, (median_rms - 30) if median_rms is not None else -80.0)
    transient_points = [
        point for point in points
        if point.get("rms_db") is None
        or (
            math.isfinite(point.get("rms_db", math.nan))
            and point["rms_db"] >= audible_floor
        )
    ]
    transient_peaks = _pick_peaks(
        transient_points,
        "flux",
        minimum=flux_threshold,
        local_only=True,
    )
    for point in transient_peaks:
        point["label"] = _sound_event_label(point, median_rms)

    band_filters = {
        "low": "lowpass=f=250",
        "mid": "highpass=f=250,lowpass=f=2000",
        "high": "highpass=f=2000",
    }

    def measure_band(item: tuple[str, str]) -> tuple[str, list[dict], str | None]:
        band, band_filter = item
        try:
            return band, _measure_band_rms(video_path, start, end, band_filter), None
        except RuntimeError as exc:
            return band, [], str(exc)

    band_points: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for band, measured, error in pool.map(measure_band, band_filters.items()):
            band_points[band] = measured
            if error:
                errors.append(f"{band} band: {error}")
    band_onsets = _detect_band_onsets(band_points)

    silence_command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        *_scope_args(start, end),
        "-i", video_path,
        "-vn", "-af",
        f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={SILENCE_MIN_SECONDS}",
        "-f", "null", "-",
    ]
    silences: list[dict] = []
    try:
        silence_result = _run_ffmpeg(silence_command, "silence detection")
        silences = _parse_silences(silence_result.stderr, start, end)
    except RuntimeError as exc:
        errors.append(str(exc))

    waveform_path = _write_audio_picture(
        video_path,
        out_dir / "waveform.png",
        start,
        end,
        "showwavespic=s=1200x220:colors=0x4cc9f0:scale=sqrt:filter=peak",
        "waveform rendering",
    )
    if waveform_path is None:
        errors.append("waveform rendering produced no artifact")
    spectrogram_path = _write_audio_picture(
        video_path,
        out_dir / "spectrogram.png",
        start,
        end,
        (
            "showspectrumpic=s=1200x420:legend=1:color=viridis:scale=log:"
            "fscale=log:stop=8000:drange=80"
        ),
        "spectrogram rendering",
    )
    if spectrogram_path is None:
        errors.append("spectrogram rendering produced no artifact")

    return {
        "waveform_path": waveform_path,
        "spectrogram_path": spectrogram_path,
        "median_rms_db": round(median_rms, 2) if median_rms is not None else None,
        "dynamic_range_db": round(dynamic_range, 2) if dynamic_range is not None else None,
        "energy_peaks": [
            {"time": point["time"], "rms_db": round(point["rms_db"], 2)}
            for point in energy_peaks
        ],
        "transient_peaks": [
            {
                "time": point["time"],
                "flux": round(point["flux"], 5),
                "rms_db": round(point["rms_db"], 2)
                if math.isfinite(point.get("rms_db", math.nan)) else None,
                "centroid_hz": round(point["centroid_hz"])
                if math.isfinite(point.get("centroid_hz", math.nan)) else None,
                "label": point["label"],
            }
            for point in transient_peaks
        ],
        "band_onsets": band_onsets,
        "silences": silences,
        "errors": errors,
    }


def _artifact_paths(data: dict) -> list[Path]:
    paths: list[Path] = []
    video = data.get("video") or {}
    for key in ("strips", "zoom_strips"):
        paths.extend(Path(strip["path"]) for strip in video.get(key) or [])
    audio = data.get("audio") or {}
    for key in ("waveform_path", "spectrogram_path"):
        if audio.get(key):
            paths.append(Path(audio[key]))
    return paths


def _load_cache(index_path: Path) -> dict | None:
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("version") != ANALYSIS_VERSION:
        return None
    if not all(path.exists() and path.stat().st_size > 0 for path in _artifact_paths(data)):
        return None
    data["cache_hit"] = True
    return data


def analyze_media(
    video_path: str,
    out_dir: Path,
    *,
    has_video: bool,
    has_audio: bool,
    start: float,
    end: float,
    resolution: int = 512,
) -> dict:
    """Create/load the cinematic evidence bundle for a media range."""
    if shutil.which("ffmpeg") is None:
        return {
            "version": ANALYSIS_VERSION,
            "video": None,
            "audio": None,
            "errors": ["ffmpeg is not installed"],
            "cache_hit": False,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    cached = _load_cache(index_path)
    if cached is not None:
        return cached

    data = {
        "version": ANALYSIS_VERSION,
        "video": _analyze_video(video_path, out_dir, start, end, resolution)
        if has_video else None,
        "audio": _analyze_audio(video_path, out_dir, start, end)
        if has_audio else None,
        "errors": [],
        "cache_hit": False,
    }
    section_errors = [
        error
        for section in (data.get("video"), data.get("audio"))
        if section
        for error in section.get("errors") or []
    ]
    data["errors"] = section_errors
    # Retry partial analyses on the next run instead of caching a broken result.
    if not section_errors:
        try:
            index_path.write_text(json.dumps(data, indent=2))
        except OSError:
            pass
    return data
