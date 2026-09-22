# Changelog

All notable changes to `/watch` are documented here.

## [Unreleased]

### Added
- **Structural boundaries pass** — new `scripts/boundaries.py`. One low-resolution ffmpeg pass (`blackdetect` + `freezedetect` + `silencedetect`, `-f null`, no files written) adds a **## Timeline** section giving fades to black, held pictures and silence to ~0.1s. A scene score measures the *rate* of pixel change, so a fade (change spread over 20-30 frames) and a held card (no change at all) are invisible to it by construction, while a spinning reel clears it on every frame — the interval detectors see exactly what it cannot. On by default for `balanced`/`token-burner`, off for `efficient` and `transcript`; `--boundaries` / `--no-boundaries` override. Skips `-af` when the source has no audio stream, and reports "no audio" distinctly from "no silence detected".
- **Post-fade frames** (`reason=post-fade`). A frame is pinned 0.5s after each fade-up out of black — the incoming shot, where title and loader cards live, and the frame scene detection structurally cannot find. Reserved against the cap like transcript cues, limited to a third of the budget so a fade-heavy edit cannot crowd out the detail engine.
- **`dedupe_perceptual(..., protect=...)`** — reasons that are never dropped as near-duplicates. A pinned frame was selected for *when* it is, so an identical neighbour is not grounds to delete it.
- 42 tests: `tests/test_boundaries.py`, `tests/test_hybrid.py`, `tests/test_report_timeline.py`, plus `clustered_clip` / `clustered_clip_silent` fixtures reproducing the failure shape (cuts bunched at the head, a fade mid-clip).

### Fixed
- **JS-runtime preflight had a false negative** (review of upstream #237 by @Verohomie). yt-dlp enables only `deno` by default; a machine with `node`/`bun`/`qjs` but no `deno` passed the check and still lost formats. When `deno` is absent but another supported runtime is on PATH, every yt-dlp call now passes `--js-runtimes <runtime>` so the finding is actually used, and the preflight says which runtime is in use. Wording softened to match yt-dlp's own: a missing runtime degrades (formats missing, lower quality), it does not fail outright.
- **Docs: `-vsync` was removed in ffmpeg 9.0, not 8.0** (also @Verohomie). Verified against the FFmpeg source: the option is present in `n8.1.2`, gone in `n9.0`; 8.x prints a deprecation notice and works. The probe-based fix was already right for both; the affected population is "ffmpeg 9", which is what Homebrew and winget ship today.
- **A clustered scene detector had its cuts deleted instead of kept.** When the coverage check (0.3.0, PR #210) found the cuts bunched into part of the range, the response was to discard them and sample uniformly — and `extract()` wipes `frame_*.jpg` on entry, so every cut frame was unlinked. On a montage this is exactly backwards: hard cuts between cards while long segments only fade is a *finding about the edit*, not a detector failure, and the discarded frames were the only ones landing on a boundary. That branch is now a **hybrid** (`engine: scene+uniform`): cuts are reserved against the cap, uniform sampling gets the remaining budget, the two merge chronologically. Measured on a 12-cut clustered clip: 12 cuts kept + 13 uniform fill, where before it was 0 cuts and 40 uniform. `extract()` gained a `prefix` parameter so the fill pass cannot delete the pinned frames.
- **Uniform fill landing on moments a cut already pinned** (`t=0.0` beside the first-frame pin, `2.12s` beside a cut at `2.10s`). Content dedup cannot catch these — the two engines write different filename prefixes and `_thumb_frames` reads a single numbered sequence — so they are pruned by *time*, within half a fill-interval of any pinned frame.
- **A rejected detector's measurements died with its JPEGs.** The sparse fallback now reports **Detected scene cuts (not sampled)** with the times, and the keyframe fallback carries its timestamps the same way. "No frame there" is not "nothing happened there".
- **Timeline times render as `MM:SS.dd`.** `format_time` rounds to whole seconds, which would print a 0.70s card as "00:01 → 00:01" — erasing the precision that is the entire reason to measure intervals.

### Docs
- SKILL.md: the three selection outcomes (`scene` / `scene+uniform` / `uniform`) and how to read each; a **Structural boundaries** section; Timeline added as a third evidence stream in Step 4; and an explicit rule that a timing claim from frames carries `±` half the sampling interval, and that anything shorter than that interval is **"not observed"**, never "absent".

## [0.4.0] — 2026-09-22

Bump because Claude Code keys plugin updates on the `plugin.json` version string: 0.3.0 users saw "already at the latest version" while `main` had moved. Every release bumps it from now on.

### Fork additions (originating here — tracked with the `fork-feature` label)
- **On-device transcription with NVIDIA Parakeet** ([#19](https://github.com/frinsen/claude-video/issues/19)). `--whisper parakeet` runs Parakeet TDT 0.6B v3 through [parakeet-mlx](https://github.com/senstella/parakeet-mlx) on Apple Silicon — 25 European languages, faster than real time, no key, no network after the one-time model download. Model via `WATCH_PARAKEET_MODEL`, command via `WATCH_PARAKEET_CMD`. Install: `uv tool install parakeet-mlx` (or pipx).
- **Generic on-device backend** `--whisper cli`: `WATCH_TRANSCRIBE_CMD` is any command with `{audio}` and `{out_dir}` placeholders that writes a `.vtt`/`.srt` (sherpa-onnx, whisper-cli, …); parakeet is a preset on it. CLI backends skip the 25 MB chunking — they handle long audio themselves.
- **`WATCH_WHISPER_BACKEND`** sets the default backend persistently, so `--whisper parakeet` need not be typed every run. Cloud defaults are unchanged unless set.
- **Report footer shows the skill version** (read from `SKILL.md`'s frontmatter), so "which version am I running?" has an answer in every run.
- README: **Updating** section for all four install paths — Claude Code marketplaces from third parties do not auto-update unless enabled in `/plugin` → Marketplaces; `npx skills update`; re-upload the `.skill`; `git pull`. On-device transcription has its own section.
- **Focused runs transcribe only the window.** With `--start`/`--end`, audio is extracted for that range (`-ss`/`-t`) and timestamps shifted back to source time, for every backend. Measured: a 30 s window of a 29-minute German video went from transcribing all 246 segments to 5, in 5.6 s wall on Parakeet; a Groq upload shrinks the same way.

### Upstream PRs landed after 0.3.0
- Merged #97 (keyframe-less range → uniform fallback), #147 (sidecar `.vtt` for local files), #175 (RGB dedup), #176 (image-token arithmetic), #154 (skill description says when to invoke); adapted #119 (`.env` in Windows encodings), #136 (preflight reads `./.env`), #127/#179 (`android` client fallback), #106 (test isolation). Full disposition of all 99 upstream PRs: [UPSTREAM.md](UPSTREAM.md).

### Fixed
- **Test suite read the developer's real `~/.config/watch/.env`** (#96 part 1; from PR #106, @greekr4). A configured `WATCH_DETAIL` made five tests fail; HOME is now redirected for the whole suite.

## [0.3.0] — 2026-09-21

Community release from the [frinsen/claude-video](https://github.com/frinsen/claude-video) fork. Upstream had 50 open pull requests and 41 open issues with no maintainer response since July; this release lands the fixes from those PRs (deduplicated — the ffmpeg `-vsync` bug alone had 13 PRs and 15 issues), resolves the remaining open issues, and adds tests for each. Merged PRs are credited in the git history; where several PRs fixed the same thing the most complete one was taken and the others' extra cases folded in.

### Fixed
- **Frame extraction on ffmpeg 9** (#99 #101 #117 #122 #126 #134 #141 #143 #149 #161 #163 #174 #180 #195 #229; PR #219). `-vsync` (deprecated since 5.1) was removed in ffmpeg 9.0, so every frame-producing mode aborted with `Unrecognized option 'vsync'` on current Homebrew/winget builds — `/watch` returned no frames at all. The flag is now probed once (`-fps_mode` on 5.0+, `-vsync` on older builds) and cached; a missing ffmpeg no longer crashes the probe.
- **Windows: `UnicodeEncodeError` printing the report** (#51 #109 #150 #134 #67; PR #192). stdout/stderr are reconfigured to UTF-8 with `errors="replace"`, so the arrow/em-dash/ellipsis glyphs no longer kill a run *after* the download and Whisper spend.
- **Windows: `UnicodeDecodeError` reading ffmpeg output** (#108; fork fix, no upstream PR). Every `subprocess.run(text=True)` now decodes as UTF-8 regardless of console codepage, so a filename with an emoji no longer crashes frame extraction.
- **Windows: permanent false "permissions 644 (should be 600)" warning** (#47 #107 #189; PR #206). POSIX mode bits are meaningless on NTFS and `chmod` is a no-op there, so the hook and `setup.py` skip the check and the chmod on Windows and document the ACL posture instead.
- **SessionStart hook aborted when `.env` existed but was unreadable** (#120; fork fix). `read_key()` checks `-r` and tolerates awk failing under `set -e`.
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
- **Re-running on the downloaded file inside the same `--out-dir` could destroy it** (#80; fork fix). The report footer now says whether the work dir is temporary or user-supplied, the script warns when the source lives inside it, and SKILL.md forbids deleting anything but an auto-created temp dir.
- **`ffprobe.exe` blocked by Windows App Control aborted the run** (#128; fork fix). Metadata falls back to parsing `ffmpeg -i`'s banner; `setup.py` treats ffprobe as optional.
- **Audio-only downloads crashed frame extraction with `Input #0, mp3`** (from PR #220). Sources without a video stream finish in transcript-only mode.
- **Test suite depended on the developer's machine** (#96; PRs #231 #208).
- **`efficient` detail crashed on a range with no keyframes** (PR #97). ffmpeg fails at encoder init when a short `--start/--end` window holds no keyframe; that now falls back to uniform sampling like the too-sparse case it is.
- **`.env` written by PowerShell or Notepad could not be read** (PR #119). UTF-16 (± BOM), UTF-8 BOM and ANSI-codepage files raised `UnicodeDecodeError` (a `ValueError`, so `except OSError` missed it) or silently lost every key; one decoder now sits behind the single parser.
- **Preflight ignored `./.env` while transcription read it** (PR #136), so `setup.py --check` could demand a key that `whisper.py` would have found. Host binaries are stubbed for argv/state-machine tests; the frame-line matcher accepts Windows path separators.

### Added
- **`--lang CODE`** and **`--force-whisper`** (PRs #187 #207).
- **Self-hosted Whisper backend** (#137; PR #199; also closed PRs #5 #18). `WATCH_WHISPER_BASE_URL` points at any OpenAI-compatible `/v1/audio/transcriptions` server; local wins over Groq/OpenAI when set, so audio never has to leave the machine.
- **Concurrent Whisper chunk uploads** (PR #233). Long audio is bounded by the slowest chunk, not their sum; output order is unchanged.
- **`WATCH_MAX_FPS`** (#37; PR #205). The 2 fps ceiling is now overridable for sub-second / fast-action clips.
- **Opt-in yt-dlp cookies** (#48; PR #184). `WATCH_COOKIES_FROM_BROWSER` / `WATCH_COOKIES_FILE`, off by default; SKILL.md tells the model to ask first.
- **Preflight warnings for a yt-dlp likely to 403** (#67 #93 #156; PR #227): more than 60 days old, built without browser impersonation (Homebrew's formula omits `curl_cffi`), or no JavaScript runtime (deno/node) for YouTube's challenge solver. Still exit 0 — a warning, not a blocker.
- **YouTube player-client retry** (#156; from PR #200). A media fetch refused with 403/429/bot-check on the default client is retried through `mweb`, `tv`, `web_embedded` — media only, since alternate clients drop caption tracks.
- **Transcript proper nouns flagged as unverified** in SKILL.md (PR #236).
- **Sidecar captions for local files** (PR #147): `clip.vtt` or `clip.<lang>.vtt` next to a local video is used as the transcript.
- **Frame dedup compares RGB, not grayscale** (PR #175), so an equal-luma hue change still counts as a distinct frame.
- **Correct image-token arithmetic** in SKILL.md (PR #176): 28-px patches, so a 512×288 frame is 209 tokens and contact sheets save nothing.
- **Skill description says when to invoke** (PR #154).
- `android` joins the YouTube media-fetch client fallbacks (PRs #127 #179).
- Docs: sandbox egress blocks (`CERTIFICATE_VERIFY_FAILED` from an allowlist proxy) explained as environmental (#83 #135).

#### Fork additions (not from an upstream PR)
- `UPSTREAM.md`: disposition of all 99 upstream PRs and 41 issues open at fork time; a weekly upstream-watch workflow opens a tracking issue for new activity.- **`WATCH_YTDLP`** — choose which yt-dlp runs (a path, or a command such as `python -m yt_dlp`) for machines with several copies; the resolved binary is always the one executed.
- **CI** on every push/PR and weekly: pytest on Ubuntu 22.04 (ffmpeg 4.4), Ubuntu 24.04 (6.1), macOS (9.x), Windows (8.1.2, latest).
- `AUTHORS.md`, and the MIT notice inside the skill package so bundles and `npx skills add` installs carry it.

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
