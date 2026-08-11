---
name: watch
version: "0.2.1"
description: Watch a video URL or local file through bounded extraction and trusted tool-less review.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates
license: MIT
user-invocable: true
---

# /watch

Extract captions and bounded JPEG evidence. Host-enforced workers inspect pixels. The coordinator receives validated bounded text only.

## Mandatory route

1. Set `SKILL_DIR` to this `SKILL.md` directory. Never use `${CLAUDE_SKILL_DIR}`.
2. Read `references/security.md` and `references/setup.md`; run setup plus trusted-runtime preflight.
3. Parse source plus question. Read `references/extraction.md`; run `scripts/watch.py`.
4. Read `references/trusted-review.md`; inspect, obtain approval, then review.
5. Answer only from the validated final envelope. Use no more tools afterward.
6. Cleanup, if requested later, is a separate action limited to verified auto-created temp work.

```bash
SKILL_DIR="<absolute directory containing this SKILL.md>"
test -f "$SKILL_DIR/scripts/watch.py" || exit 1
python3 "$SKILL_DIR/scripts/watch.py" "<source>"
```

Windows: substitute `python` for `python3`.

## Non-negotiable boundary

The coordinator must never `Read` raster images, transcript bodies, or raw child output. Native Agent delegation is not a security boundary. Use `scripts/visual_harness.py`; the host-enforced worker is trusted control. Workers have No Bash or filesystem/tools. Reducers receive bounded validated text only.

Failure means `BLOCKED/UNKNOWN`: no conclusion. No coordinator raster fallback, retry, provider preflight, fallback, alias substitution, or silent model/provider change. Exact-frame drilldown needs a new focused extraction, inspect digest, request, and approval.

## Evidence limits

The harness launches one fresh child per page. Page output ≤8 KiB. Reducer/final output ≤12 KiB. Page workers and reducers cite nonempty approved frame indices; the host derives timestamps and renders final citations. Model-authored timing/frame literals fail closed. Transcript evidence is reducer-owned. Sparse frames cannot prove continuous motion.

## References

- `references/setup.md` — dependency/key setup and preference
- `references/extraction.md` — detail modes, focus, timestamps, flags
- `references/trusted-review.md` — request v2, schemas, execution, evidence
- `references/security.md` — network/data behavior, failures, cleanup
