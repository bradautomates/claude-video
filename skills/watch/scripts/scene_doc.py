#!/usr/bin/env python3
"""scene_doc.py — combination track for /watch mining.

Merges the VERIFIED transcript (SRT) with the visual index (index.json from
vision_extract.py) into a timecoded scene document:

  doc.md — [Scene N @MM:SS] scene direction + corrected transcript + graphical
           content (extracted text + frame refs) inserted inline at timecodes

Scene boundaries = informational keyframes (type in slide|code|diagram|chart|ui).
broll/face frames are skipped (no info). Pure stdlib, deterministic; the agent
(DeepSeek) may polish wording afterward, never invent content.

Usage:
  python3 scene_doc.py <transcript.verified.srt> <frames/index.json> <doc.md>
"""
import argparse
import json
import os
import re

INFO_TYPES = {'slide', 'code', 'diagram', 'chart', 'ui'}


def ts_to_sec(ts):
    ts = ts.replace(',', '.')
    parts = [float(p) for p in ts.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def parse_srt(path):
    with open(path, encoding='utf-8', errors='replace') as f:
        raw = f.read()
    out = []
    for block in re.split(r'\n\s*\n', raw.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 2 or not re.match(r'^\d+\s*$', lines[0]):
            continue
        tm = re.match(r'^(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})', lines[1])
        if not tm:
            continue
        out.append({'start': tm.group(1), 'end': tm.group(2),
                    'text': ' '.join(lines[2:])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('verified_srt')
    ap.add_argument('index_json')
    ap.add_argument('out_doc')
    a = ap.parse_args()

    transcript = parse_srt(a.verified_srt)
    with open(a.index_json, encoding='utf-8') as f:
        index = json.load(f)

    # informational frames only, sorted by timecode
    frames = []
    for it in index:
        if 'error' in it or it.get('type') not in INFO_TYPES:
            continue
        try:
            sec = ts_to_sec(it['timecode'])
        except Exception:
            continue
        frames.append({'sec': sec, **it})
    frames.sort(key=lambda f: f['sec'])

    if not frames:
        print('warning: no informational frames in index — writing transcript-only doc',
              file=sys.stderr)

    # scene boundaries = frame times; transcript segments assigned by time
    bounds = [f['sec'] for f in frames] + [float('inf')]
    scenes = []
    for i, f in enumerate(frames):
        scenes.append({'start': f['sec'], 'end': bounds[i + 1] if i + 1 < len(bounds) else f['sec'],
                       'frames': [f]})
    # transcript-only segments before the first frame or between gaps still belong
    # to the nearest following scene; unassigned tail gets its own scene
    tail = []
    for seg in transcript:
        sec = ts_to_sec(seg['start'])
        target = None
        for s in scenes:
            if sec < s['end'] and (s['start'] == s['end'] or sec >= s['start'] - 2):
                target = s
                break
        if target is None and scenes and sec >= scenes[-1]['end']:
            tail.append(seg)
        elif target is not None:
            target.setdefault('texts', []).append(seg)
    if tail:
        scenes.append({'start': scenes[-1]['end'] if scenes else 0.0,
                       'end': ts_to_sec(tail[-1]['end']), 'frames': [], 'texts': tail})

    lines = ['# Video Scene Document', '']
    for n, s in enumerate(scenes, 1):
        end_ts = fmt_ts(s['end']) if s['end'] != float('inf') else 'end'
        lines.append(f"## Scene {n} — {fmt_ts(s['start'])}–{end_ts}")
        lines.append('')
        for f in s.get('frames', []):
            if f.get('graphics'):
                lines.append(f"[direction @{f['timecode']}] {f['graphics']}")
            if f.get('text'):
                lines.append(f"> {f['text']}  —  _frame: {f['file']} @{f['timecode']}_")
            lines.append(f"![[{f['file']}]] @{f['timecode']}")
        lines.append('')
        for seg in s.get('texts', []):
            lines.append(seg['text'])
        lines.append('')

    os.makedirs(os.path.dirname(os.path.abspath(a.out_doc)), exist_ok=True)
    with open(a.out_doc, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps({'scenes': len(scenes), 'frames_used': len(frames),
                      'transcript_segments': len(transcript), 'out': a.out_doc}))


if __name__ == '__main__':
    import sys
    main()
