---
name: watch
version: "0.3.0"
description: Watch a video (URL or local path). Downloads with yt-dlp, resolves a transcript via youtube-data MCP / native captions / local Voicebox (no paid API), extracts auto-scaled frames only once a transcript is secured (or the user asked for visual-only analysis), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates (fork: local-first transcript resolution, see DLU-302)
license: MIT
user-invocable: true
---

# /watch

You don't have a video input; this skill gives you one. A Python script downloads, extracts frames as JPEGs (scene-aware, or fast keyframes at `efficient` detail), and gets native captions when available. Transcript resolution beyond that — youtube-data MCP, Voicebox — is **your** job, done before frames, not the script's. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

## Resolve `SKILL_DIR` (do this before any command)

Every `python3 ...` / `bash ...` command below runs a bundled script under `SKILL_DIR/scripts/`. Set `SKILL_DIR` to the **absolute path of the directory containing THIS SKILL.md you just Read** — your harness told you that path in the Read result. The scripts are always a direct sibling of this file (`SKILL_DIR/scripts/watch.py`), in every install layout:

```
Read ~/.claude/skills/watch/SKILL.md    → SKILL_DIR=~/.claude/skills/watch
Read ~/.codex/skills/watch/SKILL.md     → SKILL_DIR=~/.codex/skills/watch
```

Substitute that literal path for `${SKILL_DIR}` in every command. Guard once at the start of a run:

```bash
SKILL_DIR="<absolute path of the directory containing the SKILL.md you Read>"
if [ ! -f "$SKILL_DIR/scripts/watch.py" ]; then
  echo "ERROR: scripts/watch.py not found under SKILL_DIR=$SKILL_DIR" >&2
  echo "Re-check the directory of the SKILL.md you Read and substitute it as SKILL_DIR." >&2
  exit 1
fi
```

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python`.

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. Exit 0 means required binaries (`ffmpeg`, `ffprobe`, `yt-dlp`) are present — the script emits **nothing**, proceed without comment. On non-zero exit (missing binaries), run the installer:

```bash
python3 "${SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. No API key setup, no first-run wizard — the only thing gating `/watch` is the three binaries.

Within a single session, skip Step 0 on follow-up `/watch` calls — nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Transcript resolution (do this BEFORE frame extraction)

A failed transcript means there's usually no point burning image tokens on frames — resolve the transcript first, gate frame extraction on the result.

**Step A — YouTube URL? Try the MCP transcript first, no download needed.**

Extract the video ID from the URL (`youtube.com/watch?v=<id>`, `youtu.be/<id>`, etc.), then:

```
mcp__youtube-data__transcripts_getTranscript(videoId="<id>")
```

If it returns usable text (not empty, not garbled, right language), that's your transcript — `transcript_source = "youtube-data MCP"`. Skip to the frame-extraction gate below.

**Step B — Still no transcript (non-YouTube source, or MCP came back empty). Run a cheap transcript-only probe:**

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "<source>" --detail transcript
```

This does **not** extract frames. If the source has native captions, this returns them without downloading video at all (`transcript_source = "captions"`). If not, it downloads **audio only** — no frames, no full video — and the report tells you where that audio landed.

**Step C — Still no transcript. Extract audio (if step B didn't already produce it) and hand it to Voicebox:**

```bash
bash "${SKILL_DIR}/scripts/extract-audio.sh" "<video-or-audio-path-from-report>" "<workdir>/audio.mp3"
```

```
mcp__voicebox__voicebox_transcribe(audio_path="<workdir>/audio.mp3")
```

If Voicebox returns usable text, `transcript_source = "voicebox"`. If it errors (e.g. `voicebox.service` not running) or returns nothing usable, transcript resolution has **failed** — say so, don't silently swallow it.

**"Usable" / "poor quality" is your own judgment** reading the text — garbled auto-captions, wrong language, or empty output are all things you can just see. No scripted heuristic.

## Frame-extraction gate

```
transcript secured                              → proceed to frame extraction (Step 1 below,
                                                    now downloading the full video)
transcript failed, user did NOT ask visual-only → STOP. Tell the user no transcript is
                                                    available for this video and you're not
                                                    pulling frames without one. Do not run
                                                    the script again, do not Read any frames.
transcript failed, user DID ask visual-only      → proceed to frame extraction anyway
                                                    (bug-repro screen recordings, UI/diagram
                                                    review — content that's visual by nature)
```

Note: this means the "no captions, needs Voicebox" path costs **two** `watch.py` invocations (the Step B probe, then the real frame-extraction run below) instead of one — accepted tradeoff for not burning image tokens on videos with no transcript backing.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
- **Universal rate cap: 2 fps.** The script never samples faster than 2 fps, even when a budget or `--fps` would imply more.
- **The frame ceiling is set by the detail mode** (`WATCH_DETAIL` in `~/.config/watch/.env`, or `--detail`), not a single global cap:
  - `transcript` → no frames
  - `efficient` → up to **50** (keyframes)
  - `balanced` (default) → up to **100** (scene-aware)
  - `token-burner` → **uncapped** (scene-aware; a soft warning prints past 250 frames)
  - `--max-frames N` overrides whichever cap the mode would otherwise use.
- **Full-video frame budget by duration.** Token cost grows with frame count, so the script targets a budget by duration:
  - ≤30s → ~12-30 frames
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → up to the detail cap, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke (once transcript resolution has cleared the gate)

**Step 1 — run the watch script at the real detail level.** Pass the source verbatim:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:
- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial (see Recommended limits above).
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`.
- `--timestamps T1,T2,…` — grab a frame at each of these absolute timestamps. Use this after reading the transcript to capture deictic moments ("look here", "as you can see") that visual selection alone may miss. See "Transcript-cue frames" below.
- `--max-frames N` — override the preset cap for tighter token budget.
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text).
- `--fps F` — override auto-fps (clamped to 2 fps max).
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir).
- `--no-dedup` — keep near-duplicate frames instead of collapsing held slides / static screen recordings.

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00" — pass `--start` and/or `--end`. Still capped at 2 fps and the detail-mode cap:

- ≤5s → 2 fps (up to 10 frames)
- 5-15s → 2 fps (up to 30 frames)
- 15-30s → ~2 fps (up to 60 frames)
- 30-60s → ~1.3 fps (up to 80 frames)
- 60-180s → ~0.6 fps (100 frames, capped)

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline).

```bash
# Last 10 seconds of a 1 minute video
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 2
```

**Step 2 — Read every frame path the script lists.** The Read tool renders JPEGs directly. Read all frames in a single message (parallel tool calls). Frames are chronological with a `t=MM:SS` timestamp.

**Step 3 — answer the user.** You have two streams of evidence:
- **Frames** — what's on screen at each timestamp.
- **Transcript** — whichever you resolved in the transcript-resolution steps above (`youtube-data MCP` / `captions` / `voicebox`), not necessarily whatever this second script run's own report says (it may report "none available" even though you already have one from MCP/Voicebox — that's expected, use what you resolved).

If the user asked a specific question, answer it directly citing timestamps. If not, summarize structure, key moments, notable visuals, spoken content.

This holds for `transcript` detail too: produce a **summary**, don't paste the full transcript into chat. Offer the raw transcript only if explicitly asked.

**Step 4 — clean up, always, no exceptions.**

```bash
rm -rf <work_dir>
```

Do this as your last action every single time, regardless of whether the user might ask a follow-up. (A startup safety net in `watch.py` also prunes any `watch-*` tmp dirs older than 1 hour, in case a run gets interrupted before this step runs — but don't rely on that; always run this explicitly.) If the `rm -rf` itself fails, say so — don't silently claim the disk space was reclaimed.

## Detail and frames

Default behavior comes from `~/.config/watch/.env`: `WATCH_DETAIL=transcript|efficient|balanced|token-burner` (default: `balanced`).

At `efficient` detail, the script extracts **keyframes only** (`ffmpeg -skip_frame nokey`) — near-instant, lands on scene cuts. Falls back to uniform sampling below 4 keyframes.

At `balanced` / `token-burner` detail, the script extracts **scene-aware** frames, falling back to uniform sampling only when the video is effectively static. `balanced` caps at 100 frames; `token-burner` is uncapped. Extracted images are clamped to a maximum 1998px height.

## Transcript-cue frames

Visual frame selection can miss moments a presenter explicitly flags ("look here", "notice this") because pointing at a slide is often a *low* visual change. `--timestamps` forces a frame at exact moments. **You** decide which moments matter, by reading the transcript:

1. Get the timestamped transcript (already done in transcript resolution, above).
2. Scan it for deictic cues — judgment call, not a regex.
3. Re-run with `--timestamps 4:32,7:10,9:55` (absolute source times). For a URL, point at the **downloaded local file** in the work dir so it doesn't re-download.

Behavior:
- **Additive by default.** Cue frames (`reason=transcript-cue`) merge into whatever `--detail` already selected, chronologically.
- **Pinned and counted first.** Reserved against the frame cap before the detail engine runs.
- **Honors focus mode.** With `--start/--end`, cue timestamps outside the window are dropped.
- **Cue-only frames.** `--detail transcript --timestamps …` skips scene/keyframe sampling, returns only cue frames.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS).
- **youtube-data MCP call errors or returns empty** → fall through to Step B (native captions), no retry, no need to surface this to the user — it's just one link in the chain.
- **Voicebox call errors, or `voicebox.service` isn't running** → this is the terminal fallback. Surface it: tell the user Voicebox was unavailable and no transcript could be resolved, then apply the frame-extraction gate (stop, unless visual-only was requested).
- **No transcript available after all three steps, visual-only NOT requested** → stop per the frame-extraction gate. Tell the user plainly; do not pull frames "just in case."
- **Long video warning printed** → acknowledge it. Offer to re-run focused via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's login-required or region-locked, tell the user plainly; do not keep retrying.

## Token efficiency

This skill burns tokens primarily on frames. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.
- The frame-extraction gate above is itself a token-efficiency measure — no transcript, no visual-only ask, no frames.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when Voicebox is needed, a mono 16 kHz audio clip
- Calls the `youtube-data` MCP's official transcript endpoint for YouTube video IDs — read-only, no video/audio upload, just requesting the transcript text for a public video
- Calls the local `voicebox` MCP (`voicebox_transcribe`) with an extracted audio clip when no captions exist and visual-only wasn't the ask — this never leaves the machine (Voicebox runs on `127.0.0.1:17493`)
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the default `WATCH_DETAIL` preference

**What this skill does NOT do:**
- Does not send audio or video to any paid third-party API — the Groq/OpenAI Whisper fallback from upstream was deleted; the only external network call is the read-only `youtube-data` MCP transcript lookup
- Does not access any platform account (no login, no session cookies, no posting) — yt-dlp only ever requests public data
- Does not log, cache, or write API keys anywhere — there are none
- Does not persist anything outside the working directory and `~/.config/watch/.env` — Step 4 above cleans up the working directory unconditionally, every run

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction), `scripts/transcribe.py` (caption parsing), `scripts/whisper.py` (audio extraction for the Voicebox fallback — no longer a Whisper API client, name kept for minimal diff from upstream), `scripts/extract-audio.sh` (CLI wrapper around `whisper.py`), `scripts/setup.py` (binary preflight + installer)

Review scripts before first use to verify behavior.
