---
name: watch
version: "0.6.0"
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg, pulls the transcript from captions (or Whisper API fallback), and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/bradautomates/claude-video
repository: https://github.com/bradautomates/claude-video
author: bradautomates
license: MIT
user-invocable: true
---

# /watch

> **⚡ LOGIN-WALLED SITES (Douyin, Bilibili, Weibo, etc.) — CHECK PERSISTED COOKIES FIRST!**
>
> Before asking the user for cookies, check if `~/.workbuddy/cookies/` already has a `<domain>_cookies.txt` file for the site. If it exists, the script **auto-loads it** — just run the watch command directly with **no cookie flags**. Only ask the user for cookies if the file is missing or yt-dlp reports a login/verification error.
>
> Quick check:
> ```bash
> ls ~/.workbuddy/cookies/
> ```
> If you see e.g. `douyin_cookies.txt`, just send the link to `watch.py` — done. No need to bother the user.

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

On macOS with Homebrew, it auto-installs `ffmpeg` and `yt-dlp`. On Linux/Windows, it prints the exact install commands for the user to run. It scaffolds `~/.config/watch/.env` with commented placeholders and default watch settings at `0600` perms.

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
- `--no-dedup` — keep near-duplicate frames. By default a frame-delta pass drops frames that are visually near-identical to the previous kept one (held slides, static screen recordings, paused video) so the frame budget goes to distinct content; the report's **Frames** line notes how many were dropped. Pass this only if the user needs every sampled frame (e.g. judging subtle frame-to-frame motion).
- `--cookies FILE` — path to a Netscape-format cookies.txt file for login-walled sites (Douyin, Bilibili, etc.). See [Login-walled videos](#login-walled-videos-douyin-bilibili-etc) below.
- `--cookie-string "k=v; k2=v2"` — raw Cookie header string (e.g. copied from F12 → Network → Request Headers → Cookie). Automatically converted to a temp cookies.txt. Use this when `--cookies-from-browser` fails due to Chrome App-Bound encryption (Chrome 127+ on Windows).
- `--cdp-port PORT` — Chrome DevTools Protocol port for **automatic** cookie extraction (default 9222, set to 0 to disable). When Chrome is running with `--remote-debugging-port=PORT`, the script pulls cookies directly from the browser process via CDP — bypassing App-Bound encryption entirely. No manual cookie copying needed. See [Login-walled videos](#login-walled-videos-douyin-bilibili-etc) below.
- `--save-cookie "k=v; k2=v2"` — save a raw Cookie header string to `~/.workbuddy/cookies/<domain>_cookies.txt` for **automatic reuse**. After saving once, future runs auto-load cookies for that domain without any flags. Get the string via F12 Console: `copy(document.cookie)`. See [Cookie persistence](#cookie-persistence-save-once-auto-reuse) below.

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

**Step 4 — answer the user.** You now have two streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

This holds for `transcript` detail too: even with no frames, produce a **summary** like the other modes — do not paste the full transcript into chat. Synthesize structure, key moments, and spoken content with timestamps; quote only the lines that matter. Offer the raw transcript only if the user explicitly asks for it.

**Step 5 — clean up.** The script prints a working directory at the end. If the user isn't going to ask follow-ups about this video, delete it with `rm -rf <dir>`. If they might, leave it in place.

## Detail and frames

Default behavior comes from `~/.config/watch/.env`:

- `WATCH_DETAIL=transcript|efficient|balanced|token-burner` (default: `balanced`)

At `transcript` detail, captions are enough to return a report without downloading video. If captions are missing, the script downloads audio only and tries Whisper. If no transcript can be produced, it reports the limitation clearly; re-run with `--detail balanced` for frames.

At `efficient` detail, the script downloads the video and extracts **keyframes only** (`ffmpeg -skip_frame nokey`) — a near-instant pass that lands frames on scene cuts. If a clip has fewer than 4 keyframes it falls back to uniform sampling.

At `balanced` / `token-burner` detail, the script extracts **scene-aware** frames: ffmpeg scene-change selection first, falling back to uniform sampling only when the video is effectively static. `balanced` caps at 100 frames; `token-burner` is uncapped. Frame report lines include both timestamp and selection reason. Extracted images are clamped to a maximum 1998px height for Claude Read compatibility.

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

## Login-walled videos (Douyin, Bilibili, etc.)

Some platforms require a logged-in session for yt-dlp to access video metadata or download streams. When this happens, yt-dlp returns empty results or a verification challenge, and the watch script reports "no transcript available" or a download failure.

**Symptom:** yt-dlp produces no video file, no captions, and no meaningful metadata — even though the URL works fine in a browser where you're logged in.

**Root cause:** The site serves a bot-detection challenge to anonymous (cookie-less) requests. yt-dlp's built-in `--cookies-from-browser` option should work, but on **Windows with Chrome 127+** it fails because Chrome introduced [App-Bound encryption](https://developer.chrome.com/blog/app-bound-encryption) — cookies are encrypted with a key that only the Chrome process itself can access. External tools (yt-dlp, any Python library) read ciphertext and get DPAPI decryption errors (`Failed to decrypt with DPAPI`). Closing Chrome doesn't help; the encryption mechanism blocks all external access regardless of whether Chrome is running.

**Solution — four methods, try the persistence method first (save once, then just send links):**

### Cookie persistence (save once, auto-reuse)

This is the **recommended workflow** for login-walled sites. Save cookies once with `--save-cookie`, and all future runs automatically load them — no need to pass cookies every time.

**Step 1 — get the cookie string (one-time, repeat when cookies expire):**

Open the video site in your browser (logged in), press **F12** → **Console** tab, type:

```js
copy(document.cookie)
```

This copies the cookie string to your clipboard.

**Step 2 — save it:**

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "https://www.douyin.com/video/xxxxx" \
  --save-cookie "ttwid=1|abc; msToken=xyz; sid_guard=def; ..."
```

You'll see `[watch] cookies saved to: ~/.workbuddy/cookies/douyin_cookies.txt`.

**Step 3 — from now on, just send the link:**

```bash
# No cookie flags needed — auto-loaded from ~/.workbuddy/cookies/
python3 "${SKILL_DIR}/scripts/watch.py" "https://www.douyin.com/video/yyyyy"
```

You'll see `[watch] using persisted cookies: ~/.workbuddy/cookies/douyin_cookies.txt` on stderr.

**When cookies expire (every few hours to ~1 day):** yt-dlp will fail with a login/verification error. Just repeat Step 1–2 to refresh. The old cookie file is overwritten.

**Supported domains:** Douyin, Bilibili, Weibo, Xiaohongshu, Zhihu, Twitter/X, Instagram, Facebook. For other domains, the script derives a filename from the hostname automatically.

**To clear saved cookies:** delete the file at `~/.workbuddy/cookies/<domain>_cookies.txt`.

### Method 0: `--cdp-port` (automatic, no manual cookie copying)

This is the recommended method on Windows + Chrome 127+. It extracts cookies **directly from the Chrome process** via the DevTools Protocol, completely bypassing App-Bound encryption — no DPAPI, no manual copying.

**One-time setup — launch Chrome with remote debugging:**

> **Important (Windows):** Chrome/Edge on Windows requires a **non-default** `--user-data-dir` to enable remote debugging. Use a dedicated profile directory:

```bash
# Windows (PowerShell)
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="$env:USERPROFILE\.workbuddy\chrome-debug-profile"

# macOS (default profile works)
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222

# Linux (default profile works)
google-chrome --remote-debugging-port=9222
```

Then log in to the video site (Douyin, Bilibili, etc.) in that Chrome instance. The watch script will automatically detect the debug port, extract matching cookies via CDP, and feed them to yt-dlp — no extra flags needed. Extracted cookies are **auto-persisted** to `~/.workbuddy/cookies/` so they remain available even after Chrome is closed:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "https://www.douyin.com/video/xxxxx"
```

You'll see `[watch] auto-extracted cookies via Chrome CDP (port 9222)` on stderr when it works.

**How it works:** The script queries `http://localhost:9222/json` to discover **page-level** WebSocket endpoints (browser-level `Network.getAllCookies` returns empty results on some Chrome versions). It connects to a page tab — preferring one whose URL matches the target domain — enables the `Network` domain, then calls `Network.getAllCookies` over a minimal stdlib WebSocket client (no third-party dependencies). Cookies are filtered to the target domain (including subdomain matching), converted to Netscape format, and passed to yt-dlp via `--cookies`. The temp file is deleted after use, and a copy is persisted to `~/.workbuddy/cookies/<domain>_cookies.txt` for offline reuse.

> **Note:** Chrome must already be running with `--remote-debugging-port` before you invoke the watch script. If Chrome is not in debug mode, the script silently falls back to running yt-dlp without cookies (which may fail on login-walled sites — in that case, use Method 1 or 2 below).

### Method 1: `--cookie-string` (quick manual fallback)

1. Open the video page in your browser (where you're logged in).
2. Press **F12** → **Network** tab.
3. Refresh the page, click any request to the video's domain.
4. In **Request Headers**, find the `Cookie:` line — copy the **entire value** (everything after `Cookie:`).
5. Pass it to the watch script:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "https://www.douyin.com/video/xxxxx" \
  --cookie-string "ttwid=1|abc; msToken=xyz; sid_guard=def; ..."
```

The script auto-converts the raw string to a temporary Netscape-format cookies.txt, feeds it to yt-dlp, and deletes the temp file when done.

### Method 2: `--cookies FILE` (for repeated use)

1. Install a browser extension that exports cookies in Netscape format:
   - Chrome: [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookies-txt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - Firefox: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)
2. Navigate to the video site while logged in, click the extension, and **Export** → save as `cookies.txt`.
3. Pass the file path:

```bash
python3 "${SKILL_DIR}/scripts/watch.py" "https://www.douyin.com/video/xxxxx" \
  --cookies /path/to/cookies.txt
```

### When to use which

| Situation | Recommended method |
|-----------|-------------------|
| **Any login-walled site (recommended default)** | `--save-cookie` — save once, then just send links |
| Windows + Chrome 127+ (DPAPI error), Chrome can be restarted with debug port | `--cdp-port` (automatic, no manual steps) |
| Quick one-off, no saved cookies, Chrome debug port not available | `--cookie-string` |
| Repeated use across multiple videos from the same site | `--save-cookie` (save once, auto-reuse) |
| `--cookies-from-browser` fails with DPAPI/encryption error | `--save-cookie` (preferred) or `--cdp-port` or `--cookie-string` |

**Security note:** Cookie strings and CDP-extracted cookies grant the same access as your logged-in session. Persisted cookies are stored as plain-text Netscape files in `~/.workbuddy/cookies/` — ensure this directory is not shared or synced publicly. Temp files (from `--cookie-string` and CDP) are deleted after the yt-dlp subprocess completes. CDP extraction connects only to `localhost` and never transmits cookies over the network. For maximum safety, use a fresh incognito session to capture cookies rather than your main session.

## Transcription

The script gets a timestamped transcript in one of two ways:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Whisper API fallback.** If no captions came back (or the source is a local file), the script extracts audio (`ffmpeg -vn -ac 1 -ar 16000 -b:a 64k`, ~0.5 MB/min) and uploads it to whichever Whisper API has a key configured:
   - **Groq** — `whisper-large-v3`. Preferred default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

Both keys live in `~/.config/watch/.env`. The script prefers Groq when both are set; override with `--whisper openai` to force OpenAI. Use `--no-whisper` to skip the fallback entirely.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND (no Whisper key OR Whisper API failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying. For login-walled sites (Douyin, Bilibili, etc.) where the browser works but yt-dlp gets a blank/blocked response: if persisted cookies exist at `~/.workbuddy/cookies/<domain>_cookies.txt`, they may have expired — ask the user to refresh via `--save-cookie`. If no persisted cookies exist, guide the user through the [Cookie persistence](#cookie-persistence-save-once-auto-reuse) workflow. On Windows with Chrome 127+, `--cookies-from-browser` will fail with DPAPI errors — the cookie methods above are the reliable workaround.
- **Whisper request fails** → the error is printed to stderr (likely: invalid key or rate limit). Audio over the API's 25 MB upload cap is split into chunks and transcribed automatically, so length alone won't fail it; if some chunks fail the transcript is partial and the dropped chunks are noted on stderr. The report will say "none available" only if every chunk fails. You can retry with `--whisper openai` if Groq failed (or vice versa).

## Token efficiency

This skill burns tokens primarily on frames. Order of magnitude:
- 80 frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Bumping `--resolution` to 1024 roughly quadruples the image tokens per frame. Only do it when necessary.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- When the user explicitly provides cookies via `--cookies`, `--cookie-string`, or `--save-cookie`, passes them to yt-dlp to access login-walled content (e.g. Douyin, Bilibili). Cookie strings from `--cookie-string` are written to a temporary file that is deleted immediately after the yt-dlp subprocess completes. Cookies saved via `--save-cookie` are persisted to `~/.workbuddy/cookies/<domain>_cookies.txt` as plain-text Netscape files for automatic reuse across runs.
- When no explicit cookies are provided, automatically attempts to load persisted cookies from `~/.workbuddy/cookies/` (if a file for the domain exists), then falls back to CDP extraction (if Chrome is running with `--remote-debugging-port`). Neither step requires user action.
- When Chrome is running with `--remote-debugging-port` (default 9222), automatically extracts cookies from the browser process via the DevTools Protocol (CDP) to bypass App-Bound encryption on Windows + Chrome 127+. CDP communication is local-only (localhost WebSocket). Extracted cookies are written to a temporary file that is deleted after use.
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when Whisper is needed, a mono 16 kHz audio clip
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set (preferred — cheaper, faster)
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and Groq is not, or when `--whisper openai` is forced
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the Whisper API key(s) and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory

**What this skill does NOT do:**
- Does not upload the video itself to any API — only the extracted audio goes out, and only when native captions are missing AND Whisper is not disabled with `--no-whisper`
- Does not access any platform account unless the user explicitly provides cookies via `--cookies` or `--cookie-string`, or Chrome is running with `--remote-debugging-port` (CDP auto-extraction). Without cookie input and without a debug port, yt-dlp only requests public data.
- Does not log, cache, or write cookies to stdout or stderr — temp cookie files are deleted after use. CDP-extracted cookies are persisted to `~/.workbuddy/cookies/` for offline reuse (same as `--save-cookie`).
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory and `~/.config/watch/.env` — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction), `scripts/transcribe.py` (caption selection + Whisper orchestration), `scripts/whisper.py` (Groq / OpenAI clients), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
