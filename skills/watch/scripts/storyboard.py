#!/usr/bin/env python3
"""Recover frames from a YouTube storyboard mosaic.

Last-resort visual source for when media formats are bot-gated but the page
still resolves. YouTube ships a "storyboard": a handful of JPEG mosaics, each
a grid of thumbnails spanning a slice of the video. It is low resolution
(typically 320x180 per tile) but it is real frame coverage across the whole
runtime, which beats returning nothing.

Stdlib + ffmpeg only, matching the rest of the skill. Pillow is used as a fast
path when importable, but is never required.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:  # optional fast path
    from PIL import Image  # type: ignore
    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False

# Tile widths YouTube actually ships, most useful first.
_STD_TILE_WIDTHS = (320, 480, 160, 240, 80)


def _parse_slide_times(html: str) -> list[tuple[float, float]]:
    """Return [(start_seconds, duration_seconds)] from the mhtml figcaptions."""
    out: list[tuple[float, float]] = []
    pattern = re.compile(
        r"Slide #\d+:\s*(\d+):(\d+):(\d+),(\d+).*?\(duration:\s*([\d.]+)\)"
    )
    for m in pattern.finditer(html):
        h, mi, s, ms, dur = m.groups()
        start = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0
        out.append((start, float(dur)))
    return out


def _image_ext(body: bytes) -> str | None:
    """Identify a storyboard mosaic by magic bytes.

    YouTube serves these as JPEG or WebP depending on which player client
    answered, so keying off a single magic number silently yields zero frames.
    """
    if body.startswith(b"\xff\xd8"):
        return "jpg"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return "webp"
    if body.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return None


def parse_mhtml(path: Path) -> tuple[list[tuple[bytes, str]], list[tuple[float, float]]]:
    """Split the mhtml container into raw JPEG mosaics + slide timings.

    The images are stored as raw binary between MIME boundaries. Python's
    `email` parser mangles them (it does line-based text handling), so this
    splits on the boundary at the byte level instead.
    """
    data = path.read_bytes()
    head = data[:4096].decode("utf-8", "replace")
    bm = re.search(r'boundary="([^"]+)"', head)
    if not bm:
        return [], []
    boundary = ("--" + bm.group(1)).encode()

    mosaics: list[tuple[bytes, str]] = []
    html = ""
    for part in data.split(boundary):
        i = part.find(b"\r\n\r\n")
        if i == -1:
            continue
        hdr = part[:i]
        body = part[i + 4:].rstrip(b"\r\n")
        if b"text/html" in hdr:
            html = body.decode("utf-8", "replace")
        elif b"image/" in hdr:
            ext = _image_ext(body)
            if ext:
                mosaics.append((body, ext))
    return mosaics, _parse_slide_times(html)


def _grid_for(width: int, height: int) -> tuple[int, int]:
    """Infer (cols, rows) for a mosaic, preferring standard tile widths."""
    best = None
    for cols in range(1, 11):
        if width % cols:
            continue
        tw = width // cols
        for rows in range(1, 11):
            if height % rows:
                continue
            th = height // rows
            if th == 0 or abs((tw / th) - (16 / 9)) > 0.12:
                continue
            rank = (
                _STD_TILE_WIDTHS.index(tw) if tw in _STD_TILE_WIDTHS else 99,
                -(cols * rows),
            )
            if best is None or rank < best[0]:
                best = (rank, (cols, rows))
    return best[1] if best else (1, 1)


def _dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    w, _, h = out.partition(",")
    return int(w), int(h)


def _crop_ffmpeg(src: Path, dst: Path, x: int, y: int, w: int, h: int, out_w: int) -> bool:
    vf = f"crop={w}:{h}:{x}:{y}"
    if out_w and out_w != w:
        vf += f",scale={out_w}:-2:flags=lanczos"
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-vf", vf, "-q:v", "3", str(dst)],
        capture_output=True,
    )
    return r.returncode == 0 and dst.exists()


def _sig(path: Path):
    if not _HAVE_PIL:
        return None
    try:
        im = Image.open(path).convert("L").resize((32, 18))
        return list(im.getdata())
    except Exception:
        return None


def _dist(a, b) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def extract_frames(
    mhtml: Path,
    out_dir: Path,
    duration: float | None = None,
    max_frames: int = 60,
    resolution: int = 512,
    dedup: bool = True,
) -> list[dict]:
    """Split the storyboard into timestamped frames. Returns watch.py frame dicts."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("[watch] storyboard: ffmpeg/ffprobe missing", file=sys.stderr)
        return []

    mosaics, slides = parse_mhtml(mhtml)
    if not mosaics:
        print("[watch] storyboard: no mosaics found in container", file=sys.stderr)
        return []

    work = out_dir / "_mosaics"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[float, Path]] = []
    for mi, (blob, ext) in enumerate(mosaics):
        mp = work / f"m{mi:03d}.{ext}"
        mp.write_bytes(blob)
        try:
            W, H = _dims(mp)
        except Exception:
            continue
        cols, rows = _grid_for(W, H)
        tw, th = W // cols, H // rows
        per = cols * rows

        if mi < len(slides):
            s_start, s_dur = slides[mi]
        else:
            s_dur = slides[0][1] if slides else 0.0
            s_start = mi * s_dur
        tile_dt = (s_dur / per) if per else 0.0

        for idx in range(per):
            ts = s_start + idx * tile_dt
            if duration and ts > duration:
                continue
            r, c = divmod(idx, cols)
            dst = out_dir / f"sb_{int(ts)//60:02d}m{int(ts)%60:02d}s_{idx}.jpg"
            if _crop_ffmpeg(mp, dst, c * tw, r * th, tw, th, resolution):
                candidates.append((ts, dst))

    candidates.sort(key=lambda x: x[0])

    # Drop blank padding tiles and near-duplicates (needs PIL; skipped without it).
    kept = candidates
    if dedup and _HAVE_PIL:
        kept, prev = [], None
        for ts, p in candidates:
            s = _sig(p)
            if s is None:
                kept.append((ts, p))
                continue
            if max(s) - min(s) < 8:      # blank/black padding tile
                p.unlink(missing_ok=True)
                continue
            if prev is None or _dist(s, prev) > 6.5:
                kept.append((ts, p))
                prev = s
            else:
                p.unlink(missing_ok=True)

    if max_frames and len(kept) > max_frames:
        step = (len(kept) - 1) / (max_frames - 1) if max_frames > 1 else 1
        idxs = sorted({round(i * step) for i in range(max_frames)})
        drop = [kept[i] for i in range(len(kept)) if i not in idxs]
        for _, p in drop:
            p.unlink(missing_ok=True)
        kept = [kept[i] for i in idxs]

    shutil.rmtree(work, ignore_errors=True)
    return [
        {"path": str(p), "timestamp_seconds": ts, "reason": "storyboard"}
        for ts, p in kept
    ]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: storyboard.py <storyboard.mhtml> <out-dir> [max_frames]", file=sys.stderr)
        raise SystemExit(2)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    for f in extract_frames(Path(sys.argv[1]), Path(sys.argv[2]), max_frames=n):
        print(f"{f['timestamp_seconds']:8.2f}  {f['path']}")
