#!/usr/bin/env python3
"""verify.py — audible-track verification for /watch mining.

Aligns a native/downloaded transcript (SRT) against a Whisper STT transcript (SRT),
flags low-similarity segments as candidate transcription errors, and writes:

  candidates.json  — [{timecode, native, whisper, ratio, verdict: "unset"}]
  aligned.srt      — merged provisional transcript (whisper text on mismatch)

Usage:
  python3 verify.py <native.srt> <whisper.srt> <outdir> [--ratio 0.8]

The agent (DeepSeek per model routing) then judges each candidates.json entry
against surrounding context and writes the final transcript.verified.srt.
Pure stdlib (difflib) — no pip installs.
"""
import argparse
import difflib
import json
import os
import re
import sys

SRT_TS = r'^(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})'


def parse_srt(path):
    """Parse SRT into [{index, start, end, text}]."""
    with open(path, encoding='utf-8', errors='replace') as f:
        raw = f.read()
    out = []
    for block in re.split(r'\n\s*\n', raw.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        if not re.match(r'^\d+\s*$', lines[0]):
            continue
        tm = re.match(SRT_TS, lines[1])
        if not tm:
            continue
        out.append({'index': int(lines[0]), 'start': tm.group(1),
                    'end': tm.group(2), 'text': ' '.join(lines[2:])})
    return out


def norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def ts_to_sec(ts):
    ts = ts.replace(',', '.')
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def write_srt(segments, path):
    with open(path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{seg['start']} --> {seg['end']}\n{seg['text']}\n\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('native_srt', help='downloaded/native transcript (SRT)')
    ap.add_argument('whisper_srt', help='whisper STT transcript (SRT)')
    ap.add_argument('outdir', help='where candidates.json + aligned.srt go')
    ap.add_argument('--ratio', type=float, default=0.8,
                    help='similarity threshold; below = candidate (default 0.8)')
    ap.add_argument('--window', type=float, default=8.0,
                    help='max time delta (s) between matched segments (default 8)')
    a = ap.parse_args()

    if not os.path.exists(a.native_srt) or not os.path.exists(a.whisper_srt):
        sys.exit('missing input: %s or %s' % (a.native_srt, a.whisper_srt))
    os.makedirs(a.outdir, exist_ok=True)

    native = parse_srt(a.native_srt)
    whisper = parse_srt(a.whisper_srt)
    if not native or not whisper:
        sys.exit('no segments parsed — check both SRT files')

    candidates = []
    aligned = []
    used = set()

    for n in native:
        nnorm = norm(n['text'])
        if not nnorm:
            aligned.append({'start': n['start'], 'end': n['end'], 'text': n['text']})
            continue
        nts = ts_to_sec(n['start'])
        best, bestr = None, 0.0
        for w in whisper:
            if id(w) in used:
                continue
            if abs(ts_to_sec(w['start']) - nts) > a.window:
                continue
            r = difflib.SequenceMatcher(None, nnorm, norm(w['text'])).ratio()
            if r > bestr:
                best, bestr = w, r
        if best is not None:
            used.add(id(best))
            if bestr < a.ratio:
                candidates.append({'timecode': n['start'], 'native': n['text'],
                                   'whisper': best['text'], 'ratio': round(bestr, 3),
                                   'verdict': 'unset'})
                aligned.append({'start': n['start'], 'end': n['end'],
                                'text': best['text']})
            else:
                aligned.append({'start': n['start'], 'end': n['end'],
                                'text': n['text']})
        else:
            aligned.append({'start': n['start'], 'end': n['end'], 'text': n['text']})

    with open(os.path.join(a.outdir, 'candidates.json'), 'w', encoding='utf-8') as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    write_srt(aligned, os.path.join(a.outdir, 'aligned.srt'))

    print(json.dumps({'native_segments': len(native),
                      'whisper_segments': len(whisper),
                      'candidates': len(candidates),
                      'aligned': len(aligned)}))


if __name__ == '__main__':
    main()
