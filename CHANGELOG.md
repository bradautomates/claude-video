# Changelog

All notable changes to `/watch` are documented here.

## Unreleased

### Added
- Scene-aware frame selection with a reserved timeline-coverage floor, exact frame timestamps, uniform fallback, and persistent frame caching.
- Multi-pass deep analysis guidance for long videos, using source chapters when available and focused chapter agents within the host's concurrency limit.
- Persistent download caching, source chapter reporting, audio-only inputs, non-English caption fallback, Whisper vocabulary biasing, overlap-aware long-audio stitching, and adaptive bisection recovery.
- Optional semantic sound timelines for clips and focused ranges up to 60 seconds, using OpenAI or Gemini audio models with validated events, source timestamps, local audio/visual timing support, and persistent result caching.
- Adaptive 0.9-second motion detail strips for dense short-form action, changed-pixel coverage with hysteresis-based cut/flash-versus-motion candidates, and low/mid/high-band audio onset detection for effects layered over continuous music.

### Changed
- Long videos now receive the documented 100-frame default instead of silently stopping at 80.
- Auto-generated rolling captions remove repeated word overlap before analysis.
- Audio-only transcription failures no longer tell the agent to proceed with nonexistent frames.

## [0.1.2] — 2026-04-24

### Fixed
- Windows console crash: removed the emoji from the long-video warning in `watch.py`; cp1252 consoles couldn't encode it.
- `setup.py` now prints `winget` / `pip` install commands on Windows instead of "unsupported platform" — matches what the README already promised.

### Changed
- `SKILL.md` notes that on Windows the scripts must be invoked with `python`, not `python3` (the latter is the Microsoft Store stub on Windows).

## [0.1.1] — 2026-04-24

### Fixed
- Added `commands/watch.md` shim so `/watch` is callable when installed as a Claude Code plugin. Without it, the plugin loaded but the skill wasn't exposed as a slash command.
- `scripts/build-skill.sh` now strips `commands/` from the claude.ai `.skill` bundle alongside `hooks/` and `.claude-plugin/`.

## [0.1.0] — 2026-04-24

Initial marketplace release.

### Added
- `/watch <url-or-path> [question]` slash command.
- yt-dlp download with native caption extraction (manual + auto-subs).
- ffmpeg frame extraction with auto-scaled fps (≤2 fps, ≤100 frames, duration-aware budget).
- `--start` / `--end` focused mode with denser frame budget and transcript range filtering.
- Whisper fallback (Groq preferred, OpenAI secondary) for videos without captions.
- `setup.py` preflight: silent `--check`, structured `--json`, and installer that auto-runs `brew install` on macOS.
- Session-start hook that prints a one-line status on first run / partial config.
- `.skill` bundle packaging for claude.ai upload via `scripts/build-skill.sh`.
