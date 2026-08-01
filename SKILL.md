---
name: watch
description: Watch a video or audio recording (URL or local path). Downloads supported URLs with yt-dlp, extracts scene-aware frames, adds motion strips and non-speech sound evidence for short clips, pulls and cleans captions (or uses Whisper), and combines visuals, sound, chapters, and timestamps. Supports cinematic short-form analysis and multi-pass deep analysis for long videos.
argument-hint: "<video-or-audio-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion, Task
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates
license: MIT
user-invocable: true
---

# /watch — Claude watches a video

You don't have a video input; this skill gives you one. A Python script downloads the video, extracts visual and sound evidence, gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints paths to the evidence. You then `Read` the images and combine them with the transcript and timestamped signal analysis to answer the user.

Frame selection is **scene-aware by default**: a detection pass finds visual change points (cuts, slide flips, UI actions), reserves 25% of the budget for evenly spaced timeline coverage, spends the rest on the strongest changes, and uses any leftover slots to split the largest gaps. This captures visual events without letting a dense intro starve a quiet later section. Frame timestamps are exact seek points — no drift.

For clips or focused ranges up to 60 seconds, the script also runs **cinematic analysis by default**. It creates eight-frame overview motion strips that preserve left-to-right temporal order, measures changed-pixel coverage and persistence at 10 samples/second, and adds adaptive 0.9-second detail strips around dense action peaks. It analyzes the actual audio rather than only its words: waveform, log-frequency spectrogram, silence ranges, loudness, transients, and low/mid/high-band onset novelty. This pass is local and uses FFmpeg only; it does not require a transcription key. An optional learned semantic layer can label audible effects, foley, ambience, music, texture, and their likely filmmaking functions through OpenAI or Gemini.

Downloads, frames, cinematic evidence, and Whisper transcripts are **cached persistently** under `~/.cache/watch/`. Re-running on the same video — including focused zooms with different ranges — skips the download and often the extraction entirely. This makes multi-pass analysis cheap by design.

When the source publishes chapters, the report includes those chapter titles and ranges. Use them as the first-pass structure for summaries and deep analysis; refine or replace boundaries only when the transcript and visuals show that the source chapters are misleading or too broad.

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python` — the `python3` command on Windows is the Microsoft Store stub and will not run the script.

Before every `/watch` run, verify that dependencies and an API key are in place:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. On exit 0, the script emits **nothing** — proceed to Step 1 without comment. **Do NOT announce "setup is complete" to the user** — they don't need a status message on every turn. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit, follow the table:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | No Whisper API key | Run installer to scaffold `.env`, then ask user for a key |
| `4` | Both missing | Run installer, then ask for a key |

The installer is idempotent — safe to re-run:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders at `0600` perms, and writes `SETUP_COMPLETE=true` once deps + a key are in place so the next session knows this user has already been through the wizard.

**If an API key is still missing after install:** use `AskUserQuestion` to ask the user whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key. Then write it into `~/.config/watch/.env` — set the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line. If they don't want to set up Whisper, proceed with `--no-whisper` and tell them speech text will be unavailable when native captions are missing; short-form visual and sound evidence still works.

**Structured mode (optional):** `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --json` emits `{status, first_run, missing_binaries, whisper_backend, has_api_key, config_file, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. Use this when you need to branch on specifics (e.g. "is this the user's very first run?" → `first_run: true`).

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User points at an **audio file** (`.m4a`, `.mp3`, `.wav`, `.ogg`, etc.) — voice notes, podcasts, meeting recordings. The script detects the missing video stream and produces transcript plus non-speech sound evidence for a short source/range.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best single-pass accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration; use deep analysis for complete coverage of longer videos.
- **Hard caps: 100 frames total and 2 fps.** Token cost grows with frame count, so the script targets a frame budget by duration (and never exceeds 2 fps even when the budget would imply more):
  - ≤30s → ~1-2 fps (up to 30 frames)
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → 100 frames, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke

**Step 1 — parse the user input.** Separate the video source (URL or path) from any question the user asked. Example: `/watch https://youtu.be/abc what language is this in?` → source = `https://youtu.be/abc`, question = `what language is this in?`.

**Step 2 — run the watch script.** Pass the source verbatim. Do not shell-escape it yourself beyond normal quoting:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. When either is set, the frame budget scales denser (see "Focusing on a section" below).
- `--max-frames N` — lower the 100-frame cap for a tighter token budget (e.g. `--max-frames 40`)
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text)
- `--fps F` — force uniform sampling at an explicit rate (clamped to 2 fps max)
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir)
- `--whisper groq|openai` — force a specific Whisper backend (default: prefer Groq if both keys exist)
- `--no-whisper` — disable the Whisper fallback entirely; caption-less sources have no speech text, while short-form visual and non-speech sound evidence still works
- `--sampling scene|uniform` — frame selection strategy (default `scene`; falls back to uniform automatically when no changes are detected). Passing `--fps` forces uniform.
- `--analysis auto|standard|cinematic` — `auto` adds motion + non-speech sound evidence for clips/focused ranges up to 60s (default); `standard` disables that pass; `cinematic` forces it for a longer range.
- `--sound-semantics auto|api|off` — learned semantic sound events for a range up to 60s. `auto` runs only when `SOUND_SEMANTICS_PROVIDER=openai|gemini` is configured. Use `api` when the user explicitly asks to identify or interpret sound design; this uploads only the selected audio range even if captions exist. `off` disables it.
- `--sound-provider auto|openai|gemini` — choose the semantic provider. `api` + `auto` uses the configured provider or an available matching key.
- `--sound-vocab "impact, fabric movement, forest ambience"` — plausible domain-specific sound concepts. Treat these as hints, never forced labels.
- `--no-cache` — bypass the persistent cache; download and extract into the temp work dir
- `--vocab "term1, term2, …"` — proper nouns/terms expected in the audio, passed as the Whisper prompt to bias spelling. **Use this proactively**: when the context tells you which product names, people, or tools will be spoken (from the user's question, the video title, or the conversation), pass them — otherwise Whisper garbles them ("ComfyUI" → "Confiby", "Weavy" → "webby"). Only affects Whisper runs, not caption pulls.

### Focusing on a section (denser frame coverage)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets:

- ≤5s → up to 10 frames
- 5-15s → up to 30 frames
- 15-30s → up to 60 frames
- 30-60s → up to 80 frames
- 60-180s → up to 100 frames

Focused mode is the right call for:
- Any moment/range the user names explicitly ("around 2:30", "the intro", "the last 30 seconds").
- Any video longer than ~10 minutes where the user's question is about a specific part — running focused on the relevant section is far more useful than a sparse scan of the whole thing.
- Re-runs after a full scan didn't have enough detail in some region.

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline, not offset-from-start).

Examples:
```bash
# Last 10 seconds of a 1 minute video
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at the 2 fps hard cap (60 frames)
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 2

# From 1h12m to the end of the video
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every visual evidence path the script lists.** The Read tool renders the keyframe JPEGs, motion-strip JPEGs, waveform PNG, and spectrogram PNG as images. Read them in parallel where possible. Keyframes and cinematic artifacts use precise timestamps so distinct events inside a five-second clip do not collapse to the same second.

**Step 4 — answer the user.** Use every available evidence stream:
- **Frames** — what's on screen at each timestamp
- **Motion strips** — how the camera and subjects move, how a transition unfolds, and what happens between isolated keyframes; each strip is ordered left-to-right. Read both whole-sequence overview strips and any adaptive 0.9-second detail strips.
- **Visual-event evidence** — use changed-pixel coverage, persistence, and the corresponding strip together. `cut or flash`, `localized motion`, and `sustained motion or transition` are candidate classes, not ground truth.
- **Sound evidence** — rhythm, energy, silence, impacts/onsets, low/mid/high-band novelty, frequency texture, and likely audiovisual sync points. A waveform or spectrogram cannot identify an exact sound source by itself, so keep unsupported labels tentative.
- **Semantic sound events** — when present, these are learned labels, audible cues, possible diegetic status, and story/edit functions. Their timestamps are source-relative and may include deterministic links to nearby local audio/visual change candidates. Confidence values are model estimates, not calibrated probabilities.
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

**Step 5 — clean up.** The script prints a working directory at the end. If the user isn't going to ask follow-ups about this video, delete it with `rm -rf <dir>`. If they might, leave it in place. **Never delete anything under `~/.cache/watch/`** — that is the persistent cache (videos, frames, cinematic evidence, transcripts) that makes re-runs and focused zooms near-instant. If the user wants to reclaim disk space, they can clear it themselves with `rm -rf ~/.cache/watch`.

## Short-form cinematic analysis (5-60 seconds)

Use this mode for AI-generated clips, ads, music videos, trailers, visual effects, or any request about filmmaking, editing, camera work, movement, sound design, or storytelling. `--analysis auto` already enables it for a short source or focused range. Pass `--analysis cinematic` when the requested range is longer than 60 seconds and the extra local processing is justified.

When the user explicitly wants sound identification, semantic sound understanding, sound-design interpretation, or audiovisual storytelling, add `--sound-semantics api` if a configured OpenAI or Gemini key is available. This request is the scope for the selected-audio upload. For general video questions, leave the default `auto`; it will not run unless the user has previously opted in by setting `SOUND_SEMANTICS_PROVIDER`.

When the user wants a filmmaking/story analysis, build a timestamped shot breakdown from the evidence:

1. **Shots and transitions** — mark likely in/out points and distinguish hard cut, dissolve/fade, flash, wipe, match cut, or continuous take. The report's visual-change peaks are candidates, not final labels; verify them in the surrounding motion strip.
2. **Framing and angle** — identify establishing/wide/medium/close/detail framing, high/low/eye-level/overhead angle, composition, depth, lighting, color, and visible lens cues. Do not invent an exact focal length.
3. **Movement** — separate camera movement (pan, tilt, push/pull, truck, crane, orbit, handheld shake, locked-off) from subject movement. Note direction, speed changes, foreground/background parallax, and whether motion motivates the edit.
4. **Sound design** — combine the transcript with waveform/spectrogram evidence. Note voice, music, ambience, silence, impacts, whooshes, risers, drops, and likely sync between an onset and a cut or movement. Only name an exact source when visuals, words, or direct audio inspection support it.
5. **Story beat and function** — for each moment, explain what changes for the viewer: setup, reveal, escalation, turn, payoff, loop, emotional shift, or attention reset. A five-second clip can still have several beats; use hundredths-of-a-second timestamps when timing matters.

Keep observations separate from inferences. A strong pixel-change peak proves that the image changed quickly; the strip tells you whether it was a cut, flash, camera move, or fast subject action. A spectral-change peak proves that the sound changed quickly; it does not by itself prove “door slam” or “explosion.”

## Deep analysis mode (multi-pass, complete coverage)

A single pass is capped at 100 frames in one context — fine for Q&A, lossy for a 30-minute lesson. Deep mode removes that ceiling by chaptering the video and giving each chapter its own pass in its own subagent context.

**When to use:** the user asks for a deep/complete/detailed analysis, wants to document a workflow or course lesson, says "don't lose details", or asks an exhaustive question about a video longer than ~10 minutes. For short videos (<10 min) a single scene-aware pass is usually enough — don't multi-pass unless asked.

**Warn once about cost:** deep mode spends N chapters × ~60-100 frames of image tokens across subagents. Say roughly what it will cost ("~6 chapters, each a dense pass") before launching, unless the user already opted in.

**The protocol:**

1. **Pass 1 — scan.** Run the script normally on the full video. Read the sparse frames and the full transcript. This pass is mostly for the transcript and the shape of the video; it also warms the cache (download + transcript are now free for every later pass).
2. **Chapter the video.** Prefer source chapters printed in the report. Otherwise infer chapters from transcript topic shifts and sparse frames. Aim for roughly 3-6 minutes with a one-line description each, aligned to natural transitions rather than arbitrary round numbers. Split source chapters that are too long or visually busy.
3. **Pass 2 — one subagent per chapter.** Launch chapter agents in parallel up to the host's concurrency limit and queue the rest. Each subagent's prompt must include: the video source, its chapter range, the chapter description, what the user wants, and these instructions:
   - Run `python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>" --start <chapter-start> --end <chapter-end>` (add `--resolution 768` if the chapter shows screen content: slides, UI, code, drawing software).
   - Read every frame, read the transcript section, and return exhaustive timestamped notes: what is shown, what is said, every technique/step/setting/tool visible, notable quotes. Return raw detailed notes, not a polished summary — synthesis happens later.
   - The download and transcript come from cache; the subagent only pays for its own frames.
4. **Synthesize.** Merge the chapter notes into one structured document ordered by timeline. Preserve timestamps, keep every concrete detail (tool names, settings, techniques, quotes), and flag anything a subagent reported as unclear so the user can zoom in further.
5. **Follow-ups are cheap.** Everything is cached — if the user asks about a moment, re-run focused on that range at high resolution instead of guessing from the notes.

**Chapter sizing:** 3-6 minutes keeps each focused pass dense (~0.3-0.6 fps scene-aware budget). Shorter chapters for visually busy content (drawing demos, fast UI work), longer for talking-head sections.

## Transcription

The script gets a timestamped transcript in one of two ways:

1. **Native captions (free, preferred).** yt-dlp pulls English subtitles from the source platform if available. When no English track exists, the script checks what the source actually has and fetches the best alternative — manual captions in any language beat auto-generated ones, and the original language beats translations. The report labels what it got, e.g. `captions (manual, es)` or `captions (auto, en)`. Rolling overlap in auto-caption cues is removed before the transcript is printed. **Treat auto-generated captions as lower quality** — punctuation and proper nouns can still be wrong; if the user wants deep/accurate analysis and a Whisper key exists, consider re-running with Whisper.
2. **Whisper API fallback.** If no captions came back (or the source is a local file), the script extracts mono 16 kHz audio in the backend's upload format (Opus ~0.18 MB/min for Groq, MP3 ~0.5 MB/min for OpenAI) and uploads it to whichever Whisper API has a key configured:
   - **Groq** — `whisper-large-v3`. Preferred default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

   Most content uploads as **one seamless request** — the chunk budget is derived from the encoded bitrate against the 25 MB API cap (≈1 hour for Groq/Opus, ≈40 min for OpenAI/MP3). Only longer audio is pre-split into overlapping windows (8s overlap, stitched at the midpoint, each primed with the previous window's text tail). If an upload fails after retries, the span is bisected and each half retried independently, so a bad region is isolated instead of losing the whole transcript. Pass `--vocab` to bias spelling of proper nouns (see flags above).

Both keys live in `~/.config/watch/.env`. The script prefers Groq when both are set; override with `--whisper openai` to force OpenAI. Use `--no-whisper` to skip the fallback entirely.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND (no Whisper key OR Whisper API failed). Script prints a hint pointing to setup. For video, proceed with visual and non-speech sound evidence. For a short audio-only source, proceed with waveform/spectrogram and timestamped sound evidence while stating that speech content is unavailable. In `standard` mode, audio-only input still requires transcription.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying.
- **Whisper request fails** → the error is printed to stderr (likely: invalid key or rate limit; uploads are auto-sized under the 25 MB cap and bisected on failure, so size errors and total losses are rare). The report will say "none available" for transcript, or list the failed regions if part of it survived. You can retry with `--whisper openai` if Groq failed (or vice versa).
- **Semantic sound request fails** → local waveform/spectrogram, transient, silence, and motion evidence still remains. Report the provider error and analyze from those local artifacts; do not convert a tentative signal-based candidate into an exact source label.

## Token efficiency

This skill burns tokens primarily on images. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- Cinematic mode adds up to 15 wide overview motion strips, up to eight adaptive detail strips for genuinely dense action, plus a waveform and spectrogram. Each strip compresses eight ordered samples into one image, preserving temporal evidence more efficiently than sending every sample as a separate full-size frame.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.

If you already watched a video this session and the user asks a follow-up you can answer from context, answer from what you have. But if the follow-up needs detail you don't have (a moment that fell between frames, on-screen text too small at 512px), re-running focused is cheap: the download and transcript are cached, so a `--start/--end` zoom only pays for the new frames.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames, motion strips, waveform/spectrogram images, timestamped motion/audio measurements, and, when Whisper is needed, a mono 16 kHz audio clip
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set (preferred — cheaper, faster)
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and Groq is not, or when `--whisper openai` is forced
- Sends only the selected audio range (maximum 60s) to OpenAI Chat Completions or Gemini `generateContent` when semantic sound is explicitly enabled with `--sound-semantics api`, or previously opted into with `SOUND_SEMANTICS_PROVIDER`; this may happen even when captions exist because it analyzes non-speech sound
- Writes the downloaded video, extracted frames, cinematic evidence, semantic JSON, and Whisper transcripts to a persistent cache under `~/.cache/watch/` (keyed by URL / file identity + extraction params) so re-runs skip the network and repeated local analysis; audio and other intermediates go to a working directory under the system temp dir (or `--out-dir` if specified). Semantic input MP3s are removed immediately after the request. `--no-cache` keeps results in the temp working directory instead
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the Whisper key(s), optional semantic provider/key, and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory

**What this skill does NOT do:**
- Does not upload the video itself or visual frames to any API. Extracted audio goes out for missing-caption Whisper fallback, or for a selected range when semantic sound has been explicitly enabled/configured
- Does not access any platform account (no login, no session cookies, no posting)
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory, the `~/.cache/watch/` cache, and `~/.config/watch/.env` — clean up the working directory when you're done (Step 5), but leave the cache alone

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction), `scripts/media_analysis.py` (motion strips + non-speech sound evidence), `scripts/sound_semantics.py` (OpenAI/Gemini semantic sound timeline), `scripts/transcribe.py` (caption cleanup), `scripts/whisper.py` (Groq / OpenAI transcription clients), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
