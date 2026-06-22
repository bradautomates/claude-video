# Changelog

All notable changes to `/watch` are documented here.

## [Unreleased]

### Added
- **TwelveLabs Pegasus provider** — `--provider twelvelabs` swaps the frames+Whisper pipeline for [TwelveLabs](https://twelvelabs.io) Pegasus on-the-fly analysis. The video is analyzed server-side and Claude receives **text** — a verbatim, timestamped transcript plus a scene-by-scene visual walkthrough — instead of 80-100 frame images. This removes the per-frame image-token cost and the context-length ceiling that make long videos expensive in the default mode, and needs **no Whisper key** (Pegasus does its own ASR).
  - New flags: `--provider {frames,twelvelabs}` (default `frames`, unchanged behavior), `--tl-model {pegasus1.5,pegasus1.2}`, `--tl-prompt`, `--tl-max-tokens`, `--chunk-minutes`.
  - Videos longer than `--chunk-minutes` (default 30) are split with `ffmpeg -c copy` and analyzed per-chunk, merged into one report with absolute-timestamp segment headings. Chunk length also auto-shrinks to keep each piece under the 200 MB direct-upload cap.
  - New `scripts/twelvelabs.py` (pure-stdlib REST client: asset upload, async analyze task, polling) and `scripts/chunk.py` (segmenting + trimming). `setup.py` scaffolds an optional `TWELVELABS_API_KEY` and reports `has_twelvelabs_key` in `--json`.
  - `tests/test_provider.py` covers chunk planning, segment-offset parsing, prompt building, and result extraction.

## [0.1.3] — 2026-05-09

### Fixed
- Windows: `video.info.json` is read as UTF-8 (#4). Previously `Path.read_text()` defaulted to cp1252 on Windows and crashed on yt-dlp's UTF-8 output, silently dropping Title/Uploader from the report. Same fix applied to `.env` reads/writes in `whisper.py` and `setup.py`.
- `download.py` now logs info.json parse failures to stderr instead of swallowing them.

### Security
- Hardened subprocess argv against option injection (#2): inserted `--` before the URL in the yt-dlp argv, and tightened `is_url` to reject `-`-prefixed sources and require a non-empty netloc. Resolved video/audio paths to absolute via `Path.resolve()` before passing to `ffmpeg`/`ffprobe`, so a relative path starting with `-` can't be misinterpreted as a flag.

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
