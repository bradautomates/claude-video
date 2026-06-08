# Changelog

All notable changes to `/watch` are documented here.

## [0.1.7] — 2026-06-08

### Added
- `scripts/macp_adapter.py` — MACP cloud ingestion adapter. Registers completed `/watch` UCID folder publication assets (transcript, article md/docx/pdf, hi-res frames) into MACP using presigned R2 uploads. Direct CLI: `python3 scripts/macp_adapter.py register --folder <UCID-folder> [--dry-run]`.
- `--macp-register`, `--macp-dry-run`, `--macp-base-url`, `--macp-brand-id`, `--macp-created-by`, `--macp-editorial-brief-id` flags added to `watch.py`. Watch.py accepts and preserves intent; network calls are made by the Skill after Step 4.5 completes.
- Step 4.6 (MACP cloud registration) added to `SKILL.md` after sentinel deletion.
- MACP flags section and failure modes added to `REFERENCE.md`.
- MACP env var presence check added to `scripts/setup.py` (`--json` output includes `macp_env_vars_set` / `macp_env_vars_missing`).
- `tests/test_macp_adapter.py` — 20 adapter unit tests covering all required scenarios.
- `SKILL.md` version bumped to 0.4.2.

## [0.1.6] — 2026-05-19

### Added
- Step 4.5 now mandates a `/humanizer` pass on `business-article.md` between the initial write and the docx/PDF render. Applies to every run, with no opt-out. Documented constraints for preserving image embeds, blockquotes, numbers, tool names, creator attributions, and MOR-950 voice rules across the pass.

## [0.1.5] — 2026-05-18

### Added
- Step 4.5: UCID folder allocation, `--save-dir` flag, renderer hand-off.

## [0.1.4] — 2026-05-18

### Changed
- Forked to `moreresults/claude-video`.

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
