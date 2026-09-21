# Changelog

All notable changes to `/watch` are documented here.

## [0.3.0] — 2026-09-21

Community release from the [frinsen/claude-video](https://github.com/frinsen/claude-video) fork. Upstream had 50 open pull requests and 41 open issues with no maintainer response since July; this release lands the fixes from those PRs (deduplicated — the ffmpeg `-vsync` bug alone had 13 PRs and 15 issues), resolves the remaining open issues, and adds tests for each. Merged PRs are credited in the git history; where several PRs fixed the same thing the most complete one was taken and the others' extra cases folded in.

### Fixed
- **Frame extraction on ffmpeg 8 and newer** (#99 #101 #117 #122 #126 #134 #141 #143 #149 #161 #163 #174 #180 #195 #229; PR #219). `-vsync` was removed in ffmpeg 8.0, so every frame-producing mode aborted with `Unrecognized option 'vsync'` on current Homebrew/winget builds — `/watch` returned no frames at all. The flag is now probed once (`-fps_mode` on 5.0+, `-vsync` on older builds) and cached; a missing ffmpeg no longer crashes the probe.
- **Windows: `UnicodeEncodeError` printing the report** (#51 #109 #150 #134 #67; PR #192). stdout/stderr are reconfigured to UTF-8 with `errors="replace"`, so the arrow/em-dash/ellipsis glyphs no longer kill a run *after* the download and Whisper spend.
- **Windows: `UnicodeDecodeError` reading ffmpeg output** (#108). Every `subprocess.run(text=True)` now decodes as UTF-8 regardless of console codepage, so a filename with an emoji no longer crashes frame extraction.
- **Windows: permanent false "permissions 644 (should be 600)" warning** (#47 #107 #189; PR #206). POSIX mode bits are meaningless on NTFS and `chmod` is a no-op there, so the hook and `setup.py` skip the check and the chmod on Windows and document the ACL posture instead.
- **SessionStart hook aborted when `.env` existed but was unreadable** (#120). `read_key()` checks `-r` and tolerates awk failing under `set -e`.
- **Non-English videos got YouTube's machine-translated English captions, or none at all** (#144 #153; PRs #212 #221 #234 #187). English is still requested first, but the video's own language (from yt-dlp's metadata) is then fetched with one bounded extra call, and uploader-supplied tracks rank above auto-generated ones. `--lang` overrides detection. `all` is never requested.
- **YouTube auto-caption transcripts were ~2× their real length** (PR #182; also #225). Rolling cues repeat the tail of the previous cue; the parser now uses the inline word-timing tags to identify genuinely new text and a word-overlap guard for the rest. Whitespace-padded cue bodies are no longer dropped; HTML entities are unescaped.
- **Whisper hallucinations were presented as a normal transcript** (#222; PR #223). Segments with high `no_speech_prob`, or a phrase looping at regular intervals, mark the transcript LOW CONFIDENCE with an explicit warning to judge it against the frames.
- **`--whisper openai` with only a Groq key posted the Groq key to OpenAI** (PR #214). Key lookup is now scoped to the requested backend.
- **Three divergent `.env` parsers** (PRs #232 #218). `setup.py` and `whisper.py` used their own copies that lacked inline-comment handling, so `SETUP_COMPLETE=true  # done` never registered and a commented key reached the API with the comment attached. One parser now; quoted values with a trailing comment unquote correctly.
- **`--fps` override sampled only the head of the video** (#178; PR #226). `fps=N` plus `-frames:v cap` stops ffmpeg after `cap/N` seconds; the effective fps is now lowered so the cap spreads across the range.
- **Capped frame selection spaced by list index, not time** (PR #205). A cluster of cuts kept its proportional share of the budget; selection now pins first/last and picks the candidate nearest each evenly spaced target time.
- **Scene engine "succeeded" with all its frames bunched in one spot** (PR #210). Clearing `SCENE_MIN_FRAMES` said nothing about coverage; a worst-gap check now triggers the uniform fallback and the report says why.
- **Frame report arithmetic that could not hold** (PR #224). Fallbacks reported the rejected detector's candidates, so "12 selected from 3 candidates" was possible; counts now reconcile.
- **Bare "yt-dlp did not produce a video file"** (PR #228). A 403 now explains the likely cause (stale yt-dlp, no impersonation, no JS runtime) with the matching upgrade command, and when captions came down the run finishes in clearly labelled transcript-only mode instead of dying.
- **Re-running on the downloaded file inside the same `--out-dir` could destroy it** (#80). The report footer now says whether the work dir is temporary or user-supplied, the script warns when the source lives inside it, and SKILL.md forbids deleting anything but an auto-created temp dir.
- **`ffprobe.exe` blocked by Windows App Control aborted the run** (#128). Metadata falls back to parsing `ffmpeg -i`'s banner; `setup.py` treats ffprobe as optional.
- **Audio-only downloads crashed frame extraction with `Input #0, mp3`** (from PR #220). Sources without a video stream finish in transcript-only mode.
- **Test suite depended on the developer's machine** (#96; PRs #231 #208). Host binaries are stubbed for argv/state-machine tests; the frame-line matcher accepts Windows path separators.

### Added
- **`--lang CODE`** and **`--force-whisper`** (PRs #187 #207).
- **Self-hosted Whisper backend** (#137; PR #199; also closed PRs #5 #18). `WATCH_WHISPER_BASE_URL` points at any OpenAI-compatible `/v1/audio/transcriptions` server; local wins over Groq/OpenAI when set, so audio never has to leave the machine.
- **Concurrent Whisper chunk uploads** (PR #233). Long audio is bounded by the slowest chunk, not their sum; output order is unchanged.
- **`WATCH_MAX_FPS`** (#37; PR #205). The 2 fps ceiling is now overridable for sub-second / fast-action clips.
- **Opt-in yt-dlp cookies** (#48; PR #184). `WATCH_COOKIES_FROM_BROWSER` / `WATCH_COOKIES_FILE`, off by default; SKILL.md tells the model to ask first.
- **Preflight warnings for a yt-dlp likely to 403** (#67 #93 #156; PR #227): more than 60 days old, built without browser impersonation (Homebrew's formula omits `curl_cffi`), or no JavaScript runtime (deno/node) for YouTube's challenge solver. Still exit 0 — a warning, not a blocker.
- **YouTube player-client retry** (#156; from PR #200). A media fetch refused with 403/429/bot-check on the default client is retried through `mweb`, `tv`, `web_embedded` — media only, since alternate clients drop caption tracks.
- **Transcript proper nouns flagged as unverified** in SKILL.md (PR #236).
- **`WATCH_YTDLP`** — choose which yt-dlp runs (a path, or a command such as `python -m yt_dlp`) for machines with several copies; the resolved binary is always the one executed.
- **CI** on every push/PR and weekly: pytest on Ubuntu 22.04 (ffmpeg 4.4), Ubuntu 24.04 (6.1), macOS (9.x), Windows (8.1.2, latest).
- `AUTHORS.md`, and the MIT notice inside the skill package so bundles and `npx skills add` installs carry it.
- Docs: sandbox egress blocks (`CERTIFICATE_VERIFY_FAILED` from an allowlist proxy) explained as environmental (#83 #135).

### Not taken (deferred, with reasons)
- PR #215 (screen-recording bundle, 2k lines: quality dial, tile-based screen detection, run manifest) — too entangled to review safely; its yt-dlp capability detection is covered by the preflight warnings above.
- PR #235 (download cache under `~/.cache/watch`) — a feature with a growth/cleanup design question, not a bug.
- PR #220's gallery-dl slideshow support, PR #200's storyboard fallback (needs Pillow), PRs #196 (DashScope SDK) and #193 (MuAPI) — extra dependencies or vendor-specific services; the generic local backend covers self-hosted transcription.
- PR #205's faster-whisper commits — replaced by the dependency-free OpenAI-compatible local backend.

## [0.2.0] — 2026-06-29

### Added
- **`--detail` dial** with four modes — `transcript` (captions only, no frames), `efficient` (fast keyframe pass, cap 50), `balanced` (scene-aware, cap 100, default), and `token-burner` (scene-aware, uncapped). Set the default with `WATCH_DETAIL` in `~/.config/watch/.env`.
- **Frame deduplication** (default on; `--no-dedup` to disable). Before the budget cap, a pass downscales each frame to a 16×16 grayscale thumbnail and drops frames whose mean per-pixel difference from the last *kept* frame is within threshold — so the budget goes to distinct content instead of held slides and static recordings. The **Frames** report line shows how many near-duplicates were dropped.
- **Whisper auto-chunking.** Audio over the 25 MB upload cap is split into evenly sized chunks, transcribed per chunk, with segment timestamps shifted back into source time. Partial failures are tolerated — transcription only fails if *every* chunk fails, so length alone no longer breaks it.
- **`--timestamps T1,T2,…`** — grab a frame at each absolute timestamp; reserved against the cap, and the only frames produced under `--detail transcript`.
- **`--no-whisper`** — disable transcription entirely (frames only).
- pytest suite covering config, dedup, download, fixtures, frames, setup, timestamps, watch, and whisper (no network; ffmpeg-synthesized clips).

### Changed
- **Restructured into a self-contained `skills/watch/` package** so `SKILL.md` and its `scripts/` runtime are siblings in one folder. This fixes installs on Codex, Cursor, Copilot, and other Agent Skills hosts: `npx skills add` now copies the skill as a working unit instead of grabbing the root `SKILL.md` without its scripts.
- **Harness-agnostic path resolution** — `SKILL.md` resolves `$SKILL_DIR` from where it was Read instead of the Claude-Code-only `${CLAUDE_SKILL_DIR}`, so script calls work on every host.
- `/watch` is now derived from `SKILL.md` frontmatter; the separate `commands/watch.md` wrapper was dropped to avoid a duplicate slash command.
- `balanced` now full-decodes to detect every scene cut across the whole video. The previous early-exit was faster but kept only the first cuts and dropped the tail of long videos.
- `token-burner` is exempt from the long-video "sparse scan" warning, since it keeps every scene-change frame.
- `--max-frames` is now an override on top of each mode's default cap, rather than a fixed default of 80.

### Fixed
- Non-Claude installs (`npx skills add`) were dead on arrival — the installer copied `SKILL.md` without the `scripts/` it shells out to. The self-contained package layout resolves this.

### Removed
- `V2_PLAN.md` and `V2_CONCERNS.md` planning docs.

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
