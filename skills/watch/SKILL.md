---
name: watch
description: Watch a video from a URL or local path and answer questions about its content. Use when the user shares a video and asks what is in it, wants it summarised, or asks about something shown or said on screen.
license: MIT
allowed-tools: Bash Read AskUserQuestion
metadata:
  version: "0.4.0"
  author: "frinsen (fork of bradautomates/claude-video by Bradley Bonanno, MIT)"
  homepage: "https://github.com/frinsen/claude-video"
  repository: "https://github.com/frinsen/claude-video"
---

# /watch

You don't have a video input; this skill gives you one. A Python script gets captions first, optionally downloads the video, extracts frames as JPEGs (scene-aware, or fast keyframes at `efficient` detail), gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

## Resolve `SKILL_DIR` (do this before any command)

Every `python3 ...` command below runs a bundled script under `SKILL_DIR/scripts/`. Set `SKILL_DIR` to the **absolute path of the directory containing THIS SKILL.md you just Read** — your harness told you that path in the Read result. The scripts are always a direct sibling of this file (`SKILL_DIR/scripts/watch.py`), in every install layout:

```
Read ~/.claude/plugins/cache/claude-video/watch/<ver>/skills/watch/SKILL.md → SKILL_DIR=…/skills/watch
Read ~/.codex/skills/watch/SKILL.md                                          → SKILL_DIR=~/.codex/skills/watch
Read ~/.agents/skills/watch/SKILL.md                                         → SKILL_DIR=~/.agents/skills/watch
```

Substitute that literal path for `${SKILL_DIR}` in every command. This works on every harness (Claude Code, Codex, Cursor, Gemini CLI, …) without relying on any harness-specific environment variable. Guard once at the start of a run:

```bash
SKILL_DIR="<absolute path of the directory containing the SKILL.md you Read>"
if [ ! -f "$SKILL_DIR/scripts/watch.py" ]; then
  echo "ERROR: scripts/watch.py not found under SKILL_DIR=$SKILL_DIR" >&2
  echo "Re-check the directory of the SKILL.md you Read and substitute it as SKILL_DIR." >&2
  exit 1
fi
```

## Step 0 — Setup preflight (runs every `/watch` invocation, silent on success)

**Python interpreter:** every `python3 ...` command in this skill is for macOS/Linux. On **Windows**, substitute `python` — the `python3` command on Windows is the Microsoft Store stub and will not run the script.

On the first `/watch` invocation in a session, use structured preflight so you can detect first-run setup:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --json
```

Branch on two fields:

- **`can_proceed: true` and `first_run: false`** → setup is already done (the user may have deliberately skipped a Whisper key — that's allowed). Proceed to Step 1 without comment.
- **`first_run: true`** → genuine first-time setup. Do these in order:
  1. If `missing_binaries` is non-empty, run the installer first (it auto-installs on macOS / prints commands elsewhere — see below) and confirm the binaries land. **Do not skip this and jump to preferences.**
  2. Run the installer once more if needed so it scaffolds `~/.config/watch/.env` (it only writes the template when the file is absent, so let it create the file *before* you write any values into it).
  3. Encourage a Whisper API key and ask the watch-preference questions below, then write the selected values into `~/.config/watch/.env` and set `SETUP_COMPLETE=true`.
- **`can_proceed: false` and `first_run: false`** → setup was finished before but the environment regressed (e.g. `missing_binaries` after an OS change). Run the installer to remediate, then proceed. Don't re-ask preferences.

A missing Whisper key is *encouraged to fix, not required*: on a genuine first run `status` will read `needs_key` even when binaries are present — that's your cue to encourage a key, not a blocker.

On follow-up `/watch` calls in the same session, use the silent check:

```bash
python3 "${SKILL_DIR}/scripts/setup.py" --check
```

This is a <100ms lookup. Exit 0 means /watch can run — this **includes a user who finished setup without a Whisper key** (keyless is allowed). On exit 0 the script emits **nothing** — proceed to Step 1 without comment. **Do NOT announce "setup is complete" to the user** — they don't need a status message on every turn. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit, follow the table:

| Exit | Meaning | Action |
|------|---------|--------|
| `2` | Missing binaries (`ffmpeg` / `ffprobe` / `yt-dlp`) | Run installer |
| `3` | Genuine first run with no Whisper API key | Run installer to scaffold `.env`, then encourage a key (the user may decline — proceed with `--no-whisper`) |
| `4` | Both missing | Run installer, then encourage a key |

Exit `3` only fires before the user has completed setup. Once `SETUP_COMPLETE=true` is written, a keyless install returns exit 0 and is never nagged again.

The installer is idempotent — safe to re-run:

```bash
python3 "${SKILL_DIR}/scripts/setup.py"
```

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders and default watch settings, restricted to `0600` on macOS/Linux (on Windows the file keeps the inherited user-profile ACL — POSIX modes do not apply).

**If an API key is still missing after install:** use `AskUserQuestion` to ask the user whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key. Then write it into `~/.config/watch/.env` — set the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line. If they don't want to set up Whisper, proceed with `--no-whisper` and tell them videos without native captions will come back frames-only.

**First-run watch preference:** after the installer has scaffolded `~/.config/watch/.env`, use `AskUserQuestion` to ask one question:

- Default detail (one dial). Present these as `AskUserQuestion` options in this exact order — lightest to heaviest — and keep `(recommended)` on `balanced` even though it is not first (do **not** reorder to put the recommended option first):
  - `transcript` — no frames at all, transcript only (skips video download when captions exist).
  - `efficient` — fast keyframe pass (cap 50).
  - `balanced` (recommended) — scene-aware frames (cap 100, default).
  - `token-burner` — scene-aware, uncapped (maximum fidelity; high token cost).

Write the answer directly into `~/.config/watch/.env` by setting the bare key on its own line — **no trailing inline comment** (a `# note` after the value can break parsing):

```bash
WATCH_DETAIL=balanced
```

Use the user's selected value. If they skip the question, keep the recommended default. Once dependencies, the API-key choice, and this preference are handled, write or update `SETUP_COMPLETE=true` in the same file. Do not ask this preference question again when `SETUP_COMPLETE=true`.

**Structured mode (optional):** `python3 "${SKILL_DIR}/scripts/setup.py" --json` emits `{status, can_proceed, first_run, setup_complete, missing_binaries, whisper_backend, has_api_key, config_file, watch_detail, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. `status` describes the *ideal* state (a key is encouraged, so a keyless first run reads `needs_key`); `can_proceed` is the operational gate (binaries present AND a key is set OR setup was already completed). Branch on `can_proceed`/`first_run` to decide whether to run; use `status` to decide what to encourage.

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
- **Universal rate cap: 2 fps.** The script never samples faster than 2 fps, even when a budget or `--fps` would imply more.
- **The frame ceiling is set by the detail mode** (`WATCH_DETAIL` in `~/.config/watch/.env`, or `--detail`), not a single global cap:
  - `transcript` → no frames
  - `efficient` → up to **50** (keyframes)
  - `balanced` (default) → up to **100** (scene-aware)
  - `token-burner` → **uncapped** (scene-aware; a soft warning prints past 250 frames)
  - `--max-frames N` overrides whichever cap the mode would otherwise use.
- **Full-video frame budget by duration.** Token cost grows with frame count, so the script targets a budget by duration. This budget sets the fps and the uniform-sampling fallback; scene-aware selection can fill up to the detail cap above, whichever is lower:
  - ≤30s → ~12-30 frames
  - 30s-1min → ~40 frames
  - 1-3min → ~60 frames
  - 3-10min → ~80 frames
  - \>10min → up to the detail cap, sparsely spaced (warning printed)
- If the user hands you a long video, consider asking whether they want a specific section before burning tokens on a sparse scan.

## How to invoke

**Step 1 — parse the user input.** Separate the video source (URL or path) from any question the user asked. Example: `/watch https://youtu.be/abc what language is this in?` → source = `https://youtu.be/abc`, question = `what language is this in?`.

**Step 2 — run the watch script.** Pass the source verbatim. Do not shell-escape it yourself beyond normal quoting:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "<source>"
```

Optional flags:
- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial. `transcript` = no frames (transcript only, skips video download when captions exist); `efficient` = fast keyframes (cap 50); `balanced` = scene-aware frames (cap 100); `token-burner` = scene-aware, uncapped.
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. When either is set, fps auto-scales denser (see "Focusing on a section" below).
- `--timestamps T1,T2,…` — grab a frame at each of these absolute timestamps (`SS`, `MM:SS`, or `HH:MM:SS`). Use this after reading the transcript to capture deictic moments the presenter flags ("look here", "as you can see", "notice this") that visual selection alone may miss. See "Transcript-cue frames" below.
- `--max-frames N` — override the preset cap for tighter token budget (e.g. `--max-frames 40`)
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text)
- `--fps F` — override auto-fps (clamped to 2 fps max)
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir)
- `--whisper groq|openai` — force a specific Whisper backend (default: prefer Groq if both keys exist)
- `--no-whisper` — disable the Whisper fallback entirely (frames-only if no captions)
- `--boundaries` / `--no-boundaries` — the structural boundaries pass (fades, freezes, silence). On by default for `balanced` / `token-burner`, off for `efficient` (whose promise is speed) and `transcript` (no video). See "Structural boundaries" below.
- `--no-dedup` — keep near-duplicate frames. By default a frame-delta pass drops frames that are visually near-identical to the previous kept one (held slides, static screen recordings, paused video) so the frame budget goes to distinct content; the report's **Frames** line notes how many were dropped. Pass this only if the user needs every sampled frame (e.g. judging subtle frame-to-frame motion).

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets (still capped at 2 fps, and still bounded by the detail-mode cap — the counts below assume the default `balanced` cap of 100; `efficient` tops out at 50):

- ≤5s → 2 fps (up to 10 frames)
- 5-15s → 2 fps (up to 30 frames)
- 15-30s → ~2 fps (up to 60 frames)
- 30-60s → ~1.3 fps (up to 80 frames)
- 60-180s → ~0.6 fps (100 frames, capped)

Focused mode is the right call for:
- Any moment/range the user names explicitly ("around 2:30", "the intro", "the last 30 seconds").
- Any video longer than ~10 minutes where the user's question is about a specific part — running focused on the relevant section is far more useful than a sparse scan of the whole thing.
- Re-runs after a full scan didn't have enough detail in some region.

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline, not offset-from-start).

Examples:
```bash
# Last 10 seconds of a 1 minute video
python3 "${SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at 2 fps (60 frames)
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 2

# From 1h12m to the end of the video
python3 "${SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every frame path the script lists.** The Read tool renders JPEGs directly as images for you. Read all frames in a single message (parallel tool calls) so you see them together. The frames are in chronological order with a `t=MM:SS` timestamp so you can align them to the transcript.

**Step 4 — answer the user.** You now have three streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).
- **Timeline** (when present) — *when* the structure changes, measured to ~0.1s: fades to black, held/frozen pictures, stretches of silence. This is far more precise than the frame interval, so prefer it for any timing claim.

**Timing claims: state the uncertainty, and never read absence as absence.** Your frames are samples. The gap between them is your timing error, so a boundary you know only from frames is `±` half that gap — say so rather than implying a precision you do not have. Use the **Timeline** and the **Detected scene cuts** line for exact times instead wherever they cover the moment. And when something is shorter than the sampling interval — a 3-second title card between 7-second samples — the honest answer is **"not observed"**, never "absent": you cannot prove a gap empty by not having looked into it.

**Proper nouns in a transcript are unverified.** Captions and Whisper both transcribe phonetically, so names come back wrong or invented — "Diogo Almeida" as "Dooo Almeida", "Vercel" as "Verscell", ChatGPT as "chatbt". Frames are the corrective: title cards, slides and on-screen UI usually spell a name correctly, so check it against a frame before stating it. Where no frame confirms it — including every `transcript`-detail run, which has no frames at all — give the name as-heard and say it is from the transcript and unverified. This applies to people, companies, products, model names, URLs, prices and figures: anything the user might repeat, quote or act on.

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

This holds for `transcript` detail too: even with no frames, produce a **summary** like the other modes — do not paste the full transcript into chat. Synthesize structure, key moments, and spoken content with timestamps; quote only the lines that matter. Offer the raw transcript only if the user explicitly asks for it.

**Step 5 — clean up.** The script's last line names the working directory and says whether it is *temporary* or *user-supplied*. Only a **temporary** dir (auto-created under the system temp dir, prefix `watch-`) may be deleted, and only if the user isn't going to ask follow-ups — then `rm -rf <dir>`. **Never delete a user-supplied `--out-dir`**, and never delete a directory that contains the source file you were given (the script warns when it does): that is the user's data, and a local recording may have no upstream copy.

## Detail and frames

Default behavior comes from `~/.config/watch/.env`:

- `WATCH_DETAIL=transcript|efficient|balanced|token-burner` (default: `balanced`)

At `transcript` detail, captions are enough to return a report without downloading video. If captions are missing, the script downloads audio only and tries Whisper. If no transcript can be produced, it reports the limitation clearly; re-run with `--detail balanced` for frames.

At `efficient` detail, the script downloads the video and extracts **keyframes only** (`ffmpeg -skip_frame nokey`) — a near-instant pass that lands frames on scene cuts. If a clip has fewer than 4 keyframes it falls back to uniform sampling.

At `balanced` / `token-burner` detail, the script extracts **scene-aware** frames: ffmpeg scene-change selection first, falling back to uniform sampling only when the video is effectively static. `balanced` caps at 100 frames; `token-burner` is uncapped. Frame report lines include both timestamp and selection reason. Extracted images are clamped to a maximum 1998px height for Claude Read compatibility.

Three outcomes are possible, and the **Frames** line names which one you got:

- `scene` — the cuts spanned the range; they are your frames.
- `scene+uniform` — the cuts were real but **bunched into part of the range** (a montage cuts hard between its cards and only *fades* between its long segments). The cuts are kept and pinned, uniform sampling fills the rest. Reported as "*N* scene cuts pinned, *M* uniform fill".
- `uniform` — too few cuts to be worth keeping; pure uniform sampling. The report then prints a **Detected scene cuts (not sampled)** line giving the times anyway: those are real boundaries with no frame on them. Treat them as evidence of *when* something changed, never as moments you have seen.

## Structural boundaries

A scene score measures the *rate* of pixel change per frame, which makes it blind to two things by construction: a **fade** spreads its change over 20-30 frames so no frame crosses the threshold, and a **held card** changes nothing at all. Meanwhile a spinning reel or a panning camera clears the threshold constantly. This is why a montage's real section breaks are exactly what scene detection misses.

So `balanced` / `token-burner` also run one cheap low-resolution pass with ffmpeg's *interval* detectors — `blackdetect`, `freezedetect`, `silencedetect` — and the report gains a **## Timeline** section:

- **Fades / cuts to black** — section boundaries, to ~0.1s. A frame is automatically pinned 0.5s after each fade-up (`reason=post-fade`): that is the incoming shot, where title and loader cards live.
- **Held / frozen picture** — title cards, paused playback, a slide left up.
- **Silence** — stretches below the speech floor. A transcript with content over a silent stretch is suspect. "No audio stream" and "no silence detected" are reported as different things, because they are.

Use the Timeline for every timing question; use frames for what is actually on screen. `--no-boundaries` skips the pass, `--boundaries` forces it on for `efficient`.

## Transcript-cue frames

Visual frame selection (scene/keyframe) can miss the moments a presenter explicitly flags — "look here", "as you can see", "notice this", "watch what happens" — because pointing at a slide is often a *low* visual change. `--timestamps` lets you force a frame at those exact moments. **You** decide which moments matter, by reading the transcript:

1. Run once at `--detail transcript` (or any detail) to get the timestamped transcript.
2. Scan it for deictic cues — phrases where the speaker directs attention to something on screen. This is a judgment call (ignore rhetorical "look, the point is…"); that's why it's done by you, not a regex.
3. Re-run with `--timestamps 4:32,7:10,9:55` (absolute source times). For a URL, point the second run at the **downloaded local file** in the work dir so it doesn't re-download.

Behavior:
- **Additive by default.** Cue frames (`reason=transcript-cue`) are merged into whatever `--detail` already selected, in chronological order.
- **Pinned and counted first.** Cue frames are reserved against the frame cap before the detail engine runs, so they're never evicted by even-sampling.
- **Honors focus mode.** With `--start/--end`, any cue timestamp outside the window is dropped (reported in the summary). Coordinates are always absolute source time.
- **Cue-only frames.** `--detail transcript --timestamps …` skips scene/keyframe sampling and returns *only* the cue frames (it will download the video to do so, since frames need pixels).

## Transcription

The script gets a timestamped transcript in one of three ways:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Sidecar VTT (free, local files).** For a local path, a `.vtt` next to the video (`clip.vtt`, or a language-tagged `clip.en.vtt`) is used as the transcript. This is the only free transcript route for a file that never came from a caption-bearing platform — put your own transcription output there and no API key is needed.
3. **Whisper API fallback.** If no captions came back (or the source is a local file), the script extracts audio (`ffmpeg -vn -ac 1 -ar 16000 -b:a 64k`, ~0.5 MB/min) and uploads it to whichever Whisper API has a key configured:
   - **On-device alternative:** `--whisper parakeet` (or `WATCH_WHISPER_BACKEND=parakeet` in the config) transcribes with NVIDIA Parakeet TDT 0.6B v3 via `parakeet-mlx` on Apple Silicon — no key, no upload. If the user asks for offline/local transcription and `parakeet-mlx` is missing, tell them: `uv tool install parakeet-mlx` (first run downloads the model once). `--whisper cli` runs any `WATCH_TRANSCRIBE_CMD` that writes a `.vtt`/`.srt`.
   - **Groq** — `whisper-large-v3`. Preferred default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

Both keys live in `~/.config/watch/.env`. The script prefers Groq when both are set; override with `--whisper openai` to force OpenAI. Use `--no-whisper` to skip the fallback entirely.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND (no Whisper key OR Whisper API failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **HTTP 403 on the media stream** → almost always a yt-dlp that has fallen behind the site, not a blocked video. The report prints the update command for the package manager that owns *this* yt-dlp (brew / pipx / uv / pip / winget / the release binary). Run that command **once, visibly**. It changes the user's installed software, so it goes through the normal Bash permission prompt rather than happening silently inside the script. Then re-run the same `/watch` **once**. If it still 403s, stop and relay the report's *if it persists* guidance: no loop, no cookies, no disabling TLS. Don't substitute `yt-dlp -U` for the printed command: yt-dlp refuses to self-update a package-manager install.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying. If the site needs a signed-in session (or is challenging the request as a bot), tell the user they can re-run with `WATCH_COOKIES_FROM_BROWSER=chrome` or `WATCH_COOKIES_FILE=/path/to/cookies.txt` — ask before reading their cookies, never do it unprompted.
- **`SSL: CERTIFICATE_VERIFY_FAILED` on the very first request, or every host unreachable** → you are in a network-restricted sandbox whose proxy blocks the video host (typical of Claude's cloud/web sandbox: PyPI and GitHub allowed, youtube.com not). It is not a certificate problem and not fixable from here — tell the user plainly, suggest adding the host to the egress allowlist or running on their own machine, and stop retrying.
- **Whisper request fails** → the error is printed to stderr (likely: invalid key or rate limit). Audio over the API's 25 MB upload cap is split into chunks and transcribed automatically, so length alone won't fail it; if some chunks fail the transcript is partial and the dropped chunks are noted on stderr. The report will say "none available" only if every chunk fails. You can retry with `--whisper openai` if Groq failed (or vice versa).

## Token efficiency

This skill burns tokens primarily on frames. Claude prices an image in 28x28-pixel
patches, so one frame costs `ceil(width / 28) * ceil(height / 28)` visual tokens
([vision docs](https://platform.claude.com/docs/en/build-with-claude/vision#resolution-and-token-cost)).
Cost tracks pixel area, not image count:
- At the default 512px width a 16:9 frame is 512x288, so `19 * 11 = 209` tokens.
  80 of them is about **17k**.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 makes a frame `37 * 21 = 777` tokens - 3.7x more.
  Only do it when necessary.
- Because cost follows pixel area, tiling frames into contact sheets saves nothing:
  the same nine frames cost the same whether sent as nine images or one 3x3 sheet.
  Lower `--resolution` or `--max-frames` to actually spend less.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when Whisper is needed, a mono 16 kHz audio clip
- Runs one extra local `ffmpeg` analysis pass (`blackdetect` / `freezedetect` / `silencedetect`) that writes no files and produces only timings
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set (preferred — cheaper, faster)
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and Groq is not, or when `--whisper openai` is forced
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600` on macOS/Linux; inherited user-profile ACL on Windows) to store the Whisper API key(s) and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory

**What this skill does NOT do:**
- Does not upload the video itself to any API — only the extracted audio goes out, and only when native captions are missing AND Whisper is not disabled with `--no-whisper`
- Does not access any platform account by default — yt-dlp runs signed-out and requests only public data. It uses a signed-in session **only** when the user explicitly sets `WATCH_COOKIES_FROM_BROWSER=chrome|safari|firefox` (yt-dlp `--cookies-from-browser`) or `WATCH_COOKIES_FILE=/path/to/cookies.txt` (`--cookies`), and even then it only reads — it never posts, comments, or changes account state. Cookies go only to the host the URL points at and are never copied into the working directory or printed
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory and `~/.config/watch/.env` — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction), `scripts/boundaries.py` (fade / freeze / silence detection), `scripts/transcribe.py` (caption selection + Whisper orchestration), `scripts/whisper.py` (Groq / OpenAI clients), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
