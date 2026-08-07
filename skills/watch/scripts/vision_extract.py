#!/usr/bin/env python3
"""vision_extract.py — visual-track extraction for /watch mining.

Sends each extracted keyframe to the vision model (kimi-k2.7-code via OpenCode Go,
the ONLY vision-capable model in the catalog — per model routing, vision is used
for frames only; all text work stays on DeepSeek) and writes:

  index.json — [{timecode, file, type, text, graphics}]

Frame types: slide | code | diagram | chart | ui | broll | face
(broll/face carry no information — text/graphics are empty and downstream skips them.)

Usage:
  python3 vision_extract.py <frames_dir> [--times times.log] [--out index.json]
                             [--model kimi-k2.7-code] [--concurrency 4] [--max-frames N]

Input: frames as f_00001.jpg etc. Optionally a times.log with one pts_time per
extracted frame (ffmpeg showinfo format) to stamp timecodes; without it, timecodes
are derived from the sorted filename order (approx).

Pure stdlib. Requires OPENCODE_API_KEY in the environment.
"""
import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_BASE = 'https://opencode.ai/zen/go/v1'
DEFAULT_MODEL = 'kimi-k2.7-code'

PROMPT = (
    'You are analyzing a single video keyframe. Respond with STRICT JSON only, '
    'no prose, no markdown. Schema: {"type": string, "text": string, "graphics": string}. '
    '1) type: one of slide, code, diagram, chart, ui, broll, face. '
    '2) text: ALL on-screen text verbatim (titles, labels, code, numbers) — empty if none. '
    '3) graphics: short description of graphical content (charts, diagrams, UI, '
    'schematics — what it SHOWS, not aesthetics) — empty if none. '
    'If the frame is broll or face (no informational content), return '
    '{"type": "broll", "text": "", "graphics": ""} (or "face").'
)


def parse_times(path):
    """Parse ffmpeg showinfo pts_time log into a list of float seconds."""
    times = []
    if path and os.path.exists(path):
        with open(path, encoding='utf-8', errors='replace') as f:
            for m in re.finditer(r'pts_time:([\d.]+)', f.read()):
                times.append(float(m.group(1)))
    return times


def fmt_ts(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def analyze(base, model, api_key, frame_path):
    with open(frame_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        'model': model,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': PROMPT},
                {'type': 'image_url',
                 'image_url': {'url': 'data:image/jpeg;base64,' + b64}},
            ],
        }],
        'temperature': 1,  # kimi-k2.7-code rejects anything but 1
        'max_tokens': 800,
    }
    req = urllib.request.Request(
        base.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json',
                 'Authorization': 'Bearer ' + api_key,
                 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    content = resp['choices'][0]['message']['content']
    content = re.sub(r'^```(json)?|```$', '', content.strip(), flags=re.M).strip()
    parsed = json.loads(content)
    return {
        'type': parsed.get('type', 'unknown'),
        'text': parsed.get('text', ''),
        'graphics': parsed.get('graphics', ''),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames_dir')
    ap.add_argument('--times', default=None, help='ffmpeg showinfo pts_time log')
    ap.add_argument('--out', default='index.json')
    ap.add_argument('--base', default=DEFAULT_BASE)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--max-frames', type=int, default=0, help='0 = all')
    a = ap.parse_args()

    api_key = os.environ.get('OPENCODE_API_KEY')
    if not api_key:
        sys.exit('OPENCODE_API_KEY not set')

    frames = sorted(
        f for f in os.listdir(a.frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if a.max_frames:
        frames = frames[:a.max_frames]
    if not frames:
        sys.exit('no frames found in ' + a.frames_dir)

    times = parse_times(a.times)
    if times and len(times) < len(frames):
        print('warning: %d times for %d frames — padding with filename order'
              % (len(times), len(frames)), file=sys.stderr)

    results, errors = [], []

    def work(i, name):
        path = os.path.join(a.frames_dir, name)
        try:
            info = analyze(a.base, a.model, api_key, path)
            ts = fmt_ts(times[i]) if i < len(times) else fmt_ts(i * 5)
            return {'timecode': ts, 'file': name, **info}
        except Exception as e:
            return {'timecode': '?', 'file': name, 'error': str(e)[:200]}

    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = {ex.submit(work, i, name): name for i, name in enumerate(frames)}
        for fut in as_completed(futs):
            r = fut.result()
            if 'error' in r:
                errors.append(r)
            else:
                results.append(r)

    results.sort(key=lambda r: r.get('timecode', ''))
    index = results + errors
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(json.dumps({'frames': len(frames), 'analyzed': len(results),
                      'errors': len(errors), 'out': a.out}))


if __name__ == '__main__':
    main()
