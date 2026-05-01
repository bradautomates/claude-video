# Changelog

All notable changes to `/watch` are documented here.

## [Unreleased]

### Added
- Local whisper-cli (whisper.cpp) as a third transcription backend — free, private, no API key needed. Auto-detected when `whisper-cli` is in PATH and a GGML model is found. Preferred over API backends by default.
- `--whisper local` flag to force local transcription.
- `WHISPER_MODEL` env var / config option to point at a specific GGML model file.
- `WHISPER_LOCAL=false` env var / config option to disable local whisper auto-detection.
- WAV audio extraction path for whisper.cpp (16kHz mono PCM, native format).

### Changed
- Backend priority is now: local > Groq > OpenAI (previously Groq > OpenAI).
- `setup.py` installer and `--check` / `--json` modes now detect and report local whisper availability.
- Session-start hook mentions `whisper-cpp` as an alternative when no API keys are configured.

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
