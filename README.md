# /watch

**Give Claude the ability to watch any video.**

A maintained community fork of [bradautomates/claude-video](https://github.com/bradautomates/claude-video), the `/watch` skill created by **Bradley Bonanno**. Same tool, same MIT license, same credit — plus the fixes the upstream queue was waiting for, and a place to keep developing it.

## Why this fork exists

We use `/watch` daily and want to keep building on it. In September 2026 that was not possible upstream:

- **The original project stopped being maintained.** The last maintainer commit was 2026-06-30 and the last maintainer comment 2026-07-11. Since then **50 pull requests and 41 issues** have accumulated with no response; not a single PR has ever been merged, and contributors have started withdrawing theirs.
- **The published release is broken on every current install.** ffmpeg 8 removed the `-vsync` flag; upstream 0.2.0 still passes it, so `brew install ffmpeg` / `winget install Gyan.FFmpeg` users get **zero frames** from a tool whose whole point is frames. The one-line fix sat in 13 duplicate PRs and 15 duplicate issues.
- **We wanted every open issue fixed, not just that one.** Windows crashes (cp1252, ACL false positives, blocked `ffprobe.exe`), non-English videos getting YouTube's machine-translated captions, transcripts twice their real length, Whisper hallucinating dialogue over music, one provider's API key being sent to another, frame budgets spent on the first 30 seconds, data loss on re-runs — all of it is addressed in [0.3.0](CHANGELOG.md), each with a test.
- **Good community work deserved to land.** The upstream PR authors did real diagnosis and wrote real tests. We merged the best PR per problem, folded the others' edge cases in, re-implemented where five PRs fixed the same file five different ways, and credit every author in [AUTHORS.md](AUTHORS.md) and the git history.
- **Nothing was verified before.** Upstream has no CI; the test suite could not pass on a developer machine (#96). We fixed the harness (71 → 186 tests, all passing on ffmpeg 9), verified the release end-to-end against real YouTube videos, and ship a `.skill` bundle that carries its license.
- **It has to stay installable everywhere.** Claude Code marketplace, Codex/Cursor/Copilot via `npx skills add`, and the claude.ai web bundle all point at this repo, so what the manifest names is what you get.
- **We want to keep going.** See the [roadmap](#roadmap). If upstream resumes, we will track it and offer our changes back as a single PR; this fork is not a hostile split.

## Quick install

Claude Code (recommended — auto-updates via marketplace):
```
/plugin marketplace add frinsen/claude-video
/plugin install watch@claude-video
```

Codex, Cursor, Copilot, Gemini CLI, or any of 50+ [Agent Skills](https://agentskills.io) hosts:
```bash
npx skills add frinsen/claude-video -g
```
(`-g` installs globally for your user, available across all projects. Drop it to scope per-project.)

More install options (claude.ai web, manual) in the [Install](#install) section below.

Zero config to start — `yt-dlp` and `ffmpeg` install on first run via `brew` on macOS (Linux/Windows print exact commands). Captions cover most public videos for free. Whisper API key is only needed when a video has no captions.

---

Claude can read a webpage, run a script, browse a repo. What it can't do, out of the box, is *watch a video*. You paste a YouTube link and it has to either guess from the title or pull a transcript that's missing 90% of what's on screen.

With Claude Video `/watch` you can paste a URL or a local path, ask a question, and Claude fetches captions first, downloads only what it needs, extracts frames (scene-aware, or fast keyframes at `efficient` detail), pulls a timestamped transcript (free captions when available, Whisper API as fallback), and `Read`s every frame as an image. By the time it answers, it has *seen* the video and *heard* the audio.

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## What people actually use it for

**Analyze someone else's content.** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude looks at the first frames, reads the opening transcript, breaks down the structure. Same for ad creative, competitor launches, podcast intros, anything where the *how* matters as much as the *what*.

**Diagnose a bug from a video.** Someone sends you a screen recording of something broken. `/watch bug-repro.mov what's going wrong?` Claude watches the recording, finds the frame where the issue appears, describes what's on screen, often catches the cause without you ever opening the file.

**Summarize a video.** `/watch https://youtu.be/<long-thing> summarize this` does the obvious thing — pulls the structure, the key moments, what was actually said and shown. Faster than watching at 2x.

**Cut the hype out of an update video.** `/watch https://youtu.be/<launch-video> what's actually new — skip the hype` Strip a "game-changer" feature drop down to the few things that matter, so you get the substance without ten minutes of intro and overselling.

**Turn a playlist into notes.** `/watch https://youtu.be/<video> summarize this to a note` Run it across a series and file a per-video summary, so a channel or course becomes a searchable set of notes instead of hours you have to sit through.

## How it works

1. **You paste a video and a question.** URL (anything yt-dlp supports — YouTube, Loom, TikTok, X, Instagram, plus a few hundred more) or a local path (`.mp4`, `.mov`, `.mkv`, `.webm`).
2. **`yt-dlp` checks captions first.** At `transcript` detail, captioned URLs return without downloading video. Otherwise, or when Whisper needs audio, it downloads only what the run needs.
3. **`ffmpeg` extracts frames at the chosen detail.** `efficient` decodes keyframes only (near-instant); `balanced`/`token-burner` prefer scene-change frames and fall back to the duration-aware uniform sampler when they under-produce. JPEGs are 512px wide by default and clamped to 1998px tall for Claude Read compatibility.
4. **The transcript comes from one of two places.** First try: `yt-dlp` pulls native captions (manual or auto-generated) from the source. Free, instant, accurate-ish. Fallback: extract a mono 16 kHz 64 kbps mp3 audio clip (~480 kB/min) and ship it to Whisper — Groq's `whisper-large-v3` (preferred — cheaper and faster) or OpenAI's `whisper-1`.
5. **Frames + transcript are handed to Claude.** The script prints frame paths with `t=MM:SS` markers and the transcript with timestamps. Claude `Read`s each frame in parallel — JPEGs render directly as images in its context.
6. **Claude answers grounded in what's actually on screen and in the audio.** Not "based on the description" or "according to the title." It saw the frames. It heard the transcript. It answers the way someone who watched the video would.
7. **Cleanup.** The script prints a working directory at the end. If you're not asking follow-ups, Claude removes it.

## Frame budget — why it matters

Token cost is dominated by frames. Every frame is an image; image tokens add up fast. The script's auto-fps logic exists so you don't blow your context budget on a sparse scan of a 30-minute video that would have been better answered by a focused 30-second window.

| Duration | Default frame budget | What you get |
|----------|---------------------|--------------|
| ≤30 s | ~30 frames | Dense — basically every key moment |
| 30 s - 1 min | ~40 frames | Still dense |
| 1 - 3 min | ~60 frames | Comfortable |
| 3 - 10 min | ~80 frames | Sparse but workable |
| > 10 min | 100 frames (capped modes) | "Sparse scan" warning — re-run focused, or `--detail token-burner` for full uncapped coverage |

When the user names a moment ("around 2:30", "the last 30 seconds", "from 0:45 to 1:00"), pass `--start` / `--end`. Focused mode gets denser per-second budgets, capped at 2 fps. Far more useful than a sparse pass over the whole thing.

## Frame deduplication

Frame selection — keyframes (`efficient`), scene-change detection (`balanced`/`token-burner`), or the uniform sampler it falls back to — can still surface near-identical frames: a screen recording that holds one slide for 90 seconds produces a dozen, each billed as a separate image. A dedup pass drops them before frames reach Claude. It runs by default on every frame mode (`--no-dedup` turns it off):

1. One `ffmpeg` call scales each extracted JPEG to a 16×16 grayscale thumbnail. Everything after is pure-stdlib Python — no image libraries.
2. For each frame, compute the **mean absolute difference** against the *last frame that was kept* (average per-pixel brightness change, 0–255 scale).
3. If that difference is at or below the threshold (`2.0`), the frame is a near-duplicate and is dropped. Otherwise it's kept and becomes the new reference.
4. The frame-budget cap applies *after* dedup, so the budget is spent on distinct frames.

Comparing against the last *kept* frame (not the previous one) catches slow fades that never trip a frame-to-frame threshold. The threshold is deliberately low and measures absolute brightness rather than structure, so a one-line code diff, a terminal scrolling a row, or two differently-colored flat slides all survive.

The **Frames** line reports what was collapsed, e.g. `6 selected from 14 candidates (… 8 near-duplicates dropped …)`. On always-moving footage nothing is dropped and you pay what you would have anyway.

## Detail modes — measured

The `--detail` dial trades speed and token cost for visual fidelity. Numbers below are from a real run against a **49:08** YouTube video (1280×720, English auto-captions) — a long, mostly-static screen recording, the case that stresses the caps hardest. Extraction times are local CPU against a pre-downloaded copy; the one-time download was **~37 s** / 76 MB, shared by the three frame modes.

| Mode | Engine | Frames | Cap | Extraction time | Temporal coverage | Est. image tokens |
|------|--------|--------|-----|-----------------|-------------------|-------------------|
| `transcript` | none (captions) | 0 | — | **~4.5 s** (one yt-dlp call, no download) | full (text) | 0 (≈26.6k text tokens) |
| `efficient` | keyframe (`-skip_frame nokey`) | 50 | 50 | **~0.5 s** | 0:00 → 49:04 (full) | **~9.8k** |
| `balanced` | scene-change | 100 | 100 | **~20.9 s** | 0:00 → 48:38 (full) | **~19.7k** |
| `token-burner` | scene-change | 116 | uncapped | **~21.0 s** | 0:00 → 48:38 (full) | **~22.8k** |

- **Image tokens** use Anthropic's `(width × height) / 750` — at the default 512px width these 720p frames are 512×288, **≈197 tokens/frame**; `--resolution 1024` roughly 4×s that. The transcript is surfaced in every captioned mode and on long videos is often the larger cost.
- **One sampling rule across frame modes.** Each detects all candidates across the full range, then even-samples (first + last always kept) down to its cap. The modes differ only in candidate *source* (keyframes vs. scene cuts) and cap, never in how coverage is spread — so the last frame always lands at the end, not partway through.
- **`efficient` is the speed tier** (~0.5 s) — it only reconstructs keyframes, so it's ~40× faster than the scene modes, which decode every frame to find cuts. It can also return *more* frames than `balanced` on low-motion footage (keyframes outnumber scene cuts); "efficient" means fast extraction, not fewer frames.
- **`token-burner` only diverges from `balanced` past the cap.** This clip had 116 cuts, so `balanced` sampled 100 and `token-burner` kept all 116. On high-motion video with hundreds of cuts, `token-burner` keeps everything (and trips the >250-frame token warning) while `balanced` thins to 100.

End-to-end from a cold URL, `transcript` is the cheapest mode by far; the frame modes add the shared ~37 s download on top of the extraction times above.

## Install

| Surface | Install |
|---------|---------|
| **Claude Code** | `/plugin marketplace add frinsen/claude-video` then `/plugin install watch@claude-video` |
| **Codex, Cursor, Copilot, Gemini CLI, +50 more** | `npx skills add frinsen/claude-video -g` |
| **claude.ai** (web) | [Download `watch.skill`](https://github.com/frinsen/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **Manual / dev** | `git clone` then symlink `skills/watch` into your host's skills dir (see below) |

### Claude Code

```
/plugin marketplace add frinsen/claude-video
/plugin install watch@claude-video
```

Update later with `/plugin update watch@claude-video`.

### Codex, Cursor, Copilot, Gemini CLI, and 50+ other hosts

The [Agent Skills](https://agentskills.io) CLI installs the skill into whatever agents it detects:

```bash
npx skills add frinsen/claude-video -g
```

`-g` installs globally for your user (`~/.codex/skills`, `~/.cursor/skills`, etc.); drop it to install into the current project instead. Useful flags:

- `-a, --agent <names…>` — target specific hosts, e.g. `-a codex -a cursor`
- `-l, --list` — list the skills in this repo without installing
- `--copy` — copy files instead of symlinking (for filesystems without symlink support)

The CLI discovers the skill from `skills/watch/SKILL.md` and copies the whole folder — `SKILL.md` plus its `scripts/` runtime — as a self-contained unit. `SKILL.md` resolves its own scripts relative to wherever it was installed, so it works the same on every host.

Update later with `npx skills update watch -g`.

### claude.ai (web)

1. [Download `watch.skill`](https://github.com/frinsen/claude-video/releases/latest) from the latest release.
2. Go to Settings → Capabilities → Skills.
3. Click `+` and drop the file in.

Enable "Code execution and file creation" under Capabilities first — the skill shells out to `ffmpeg` and `yt-dlp`, so it won't run without it.

### Manual (developer)

Clone the repo and symlink the self-contained skill folder into your host's skills directory — the symlink keeps the install in sync with your working tree as you edit:

```bash
git clone https://github.com/frinsen/claude-video.git
ln -s "$(pwd)/claude-video/skills/watch" ~/.claude/skills/watch   # or ~/.codex/skills/watch
```

For claude.ai, build the `.skill` bundle from source: `bash skills/watch/scripts/build-skill.sh` produces `dist/watch.skill`.

## First run

On the first `/watch` call, the skill runs `scripts/setup.py --check`. If `ffmpeg` / `yt-dlp` aren't on your PATH, or no Whisper API key is set, it walks you through fixing it:

- **macOS** — auto-runs `brew install ffmpeg yt-dlp`.
- **Linux** — prints the exact `apt` / `dnf` / `pipx` commands.
- **Windows** — prints the `winget` / `pip` commands.
- **API key** — scaffolds `~/.config/watch/.env` with commented placeholders for `GROQ_API_KEY` (preferred) and `OPENAI_API_KEY`. Restricted to mode `0600` on macOS/Linux; on Windows it keeps the inherited profile ACL (you, SYSTEM, Administrators) — POSIX modes are not settable there.

After setup, preflight is silent and `/watch` just works. The check is a sub-100ms lookup, so it doesn't slow you down on subsequent runs.

## Bring your own keys

Captions cover the majority of public videos for free. The Whisper fallback only kicks in when a video genuinely has no caption track — typically local files, TikToks, some Vimeos, and the occasional caption-less YouTube upload.

| Capability | What you need | Cost |
|------------|---------------|------|
| Download + native captions | `yt-dlp` + `ffmpeg` | Free |
| Whisper fallback (preferred) | [Groq API key](https://console.groq.com/keys) — `whisper-large-v3` | Cheap, fast |
| Whisper fallback (alt) | [OpenAI API key](https://platform.openai.com/api-keys) — `whisper-1` | Standard pricing |
| Whisper fallback (self-hosted) | `WATCH_WHISPER_BASE_URL` → any OpenAI-compatible transcription server | Free, fully offline |
| Disable Whisper entirely | `--no-whisper` | Free, frames-only when no captions |

## Usage

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
/watch https://www.tiktok.com/@user/video/123 summarize this
/watch ~/Movies/screen-recording.mp4 when does the UI break?
/watch https://vimeo.com/123 what tools does she mention?
```

Focused on a specific section — denser frame budget, lower token cost:
```
/watch https://youtu.be/abc --start 2:15 --end 2:45
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00            # from 1h12m to end
```

Other knobs (passed to `scripts/watch.py`):

- `--detail transcript|efficient|balanced|token-burner` — fidelity/speed dial. `transcript` skips frames (transcript only); `efficient` uses fast keyframes (cap 50); `balanced` uses scene-aware frames (cap 100); `token-burner` is scene-aware and uncapped.
- `--timestamps T1,T2,…` — grab a frame at each absolute timestamp (`SS`/`MM:SS`/`HH:MM:SS`). Claude reads the transcript first, then targets the moments the presenter flags ("look here", "as you can see"). Added on top of the detail frames (reserved against the cap); out-of-window cues are dropped in focus mode; with `--detail transcript` these become the only frames.
- `--max-frames N` — lower the frame cap for a tighter token budget.
- `--resolution W` — bump frame width to 1024 px when Claude needs to read on-screen text (slides, terminals, code).
- `--fps F` — override the auto-fps calculation (capped at 2 fps by default; raise the ceiling with `WATCH_MAX_FPS=24` for sub-second or fast-action clips, paired with `--start`/`--end` and `--max-frames`).
- `--lang de` — prefer captions in a given language. By default the video's own language is used (read from yt-dlp's metadata) so non-English videos get their real captions instead of YouTube's machine-translated English track; English is the fallback.
- `--force-whisper` — ignore native captions and transcribe with Whisper. For sources whose auto-captions are too poor to use.
- `--whisper groq|openai|local` — force a specific Whisper backend. `local` is any server exposing OpenAI's `/v1/audio/transcriptions` route (whisper.cpp `server`, faster-whisper-server, speaches, LM Studio): set `WATCH_WHISPER_BASE_URL=http://localhost:8080` (optional `WATCH_WHISPER_MODEL`, `WATCH_WHISPER_API_KEY`) and audio never leaves the machine.
- `--no-whisper` — disable transcription entirely; frames only.
- `--no-dedup` — keep near-duplicate frames. By default a frame-delta pass drops frames that are visually near-identical to the one before them (held slides, static screen recordings, paused video), so the frame budget is spent on distinct content; this flag turns that off.
- `--out-dir DIR` — keep working files somewhere specific (default: auto-generated tmp dir).

Environment / `~/.config/watch/.env` settings: `WATCH_DETAIL` (default detail mode), `WATCH_MAX_FPS` (fps ceiling), `WATCH_WHISPER_BASE_URL` (local Whisper server), `WATCH_YTDLP` (which yt-dlp to run — a path, or a command like `python -m yt_dlp`, for machines with several copies), and opt-in `WATCH_COOKIES_FROM_BROWSER=chrome|safari|firefox` / `WATCH_COOKIES_FILE=/path/to/cookies.txt` for sites that refuse signed-out requests (Instagram, YouTube's bot gate). Cookies are never read unless you set one of these.

## Limits

- **Long-video accuracy depends on the detail mode.** On the capped modes (`efficient`, default `balanced`) coverage thins out past ~10 minutes — the frame cap spreads across the whole clip, so the script prints a "sparse scan" warning and you're better off re-running focused with `--start`/`--end`. `token-burner` lifts the cap and keeps *every* scene-change frame across the full video, so it stays complete on longer clips at the cost of more image tokens. The 10-minute mark is guidance for the capped modes, not a hard ceiling.
- **Needs outbound network to the video host.** Sandboxed environments with a domain allowlist — Claude's cloud/web sandbox, locked-down CI — typically allow PyPI and GitHub but not `youtube.com`, and the proxy surfaces the block as `SSL: CERTIFICATE_VERIFY_FAILED` (a self-signed interception certificate), not as a clear "blocked" error. That is the environment, not the skill: add the host to your egress allowlist, or run `/watch` on a machine with normal internet and pass a local file. Datacenter IPs may additionally hit YouTube's bot gate; the script retries with alternate player clients and explains the failure if all are refused.
- **Detail is one dial.** Defaults are balanced: scene-aware frames, 2 fps max, 100-frame cap. Use `--detail efficient` for a fast 50-frame keyframe pass, or `--detail token-burner` for uncapped scene candidates. Set `WATCH_DETAIL` in `~/.config/watch/.env` to change the default.

## Structure

```
.
├── skills/watch/                 # self-contained skill — copied as a unit by every installer
│   ├── SKILL.md                  # skill contract — the source of truth across all surfaces
│   └── scripts/
│       ├── watch.py              # entry point — orchestrates download → frames → transcript
│       ├── download.py           # yt-dlp wrapper
│       ├── frames.py             # ffmpeg frame extraction + auto-fps logic
│       ├── transcribe.py         # VTT parsing + dedupe + Whisper orchestration
│       ├── whisper.py            # Groq / OpenAI clients (pure stdlib)
│       ├── config.py             # shared config (~/.config/watch/.env)
│       ├── setup.py              # preflight + installer
│       └── build-skill.sh        # build dist/watch.skill for claude.ai upload (dev-only)
├── hooks/                        # SessionStart status hook (Claude Code only)
├── .claude-plugin/               # plugin.json + marketplace.json (Claude Code)
├── .codex-plugin/                # plugin.json — Codex/agents manifest ("skills": "./skills/")
├── .agents/plugins/              # marketplace.json — Agent Skills marketplace listing
├── AGENTS.md → CLAUDE.md         # generic-agent entry point
├── tests/                        # pytest suite (ffmpeg-synthesized clips, no network)
└── .github/workflows/            # release.yml — auto-builds watch.skill on tag push
```

## Develop

```bash
# Run the test suite (stdlib + pytest; ffmpeg required for frame tests):
python3 -m pytest -q

# Build the claude.ai upload bundle:
bash skills/watch/scripts/build-skill.sh      # → dist/watch.skill
```

Releasing: tag `vX.Y.Z`, push the tag. The workflow builds `dist/watch.skill` and attaches it to the GitHub release. Keep the version in sync across `skills/watch/SKILL.md`, `.claude-plugin/plugin.json`, and `.codex-plugin/plugin.json`.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Roadmap

CI across ffmpeg 5/7/9 · the deferred upstream PRs (download cache #235, quality dial #215, TikTok slideshows #220) · an opt-in `--report` editorial mode (hook analysis, pacing, structured report) · upstream sync if `bradautomates/claude-video` resumes. Issues and PRs welcome.

## Credits & license

Created by [Bradley Bonanno](https://github.com/bradautomates) ([bradautomates/claude-video](https://github.com/bradautomates/claude-video), [@bradbonanno](https://www.youtube.com/@bradbonanno)). Fork maintained by [frinsen](https://github.com/frinsen); community fixes credited in [AUTHORS.md](AUTHORS.md). [MIT](LICENSE), © 2026 Bradley Bonanno — not affiliated with or endorsed by the original author. Built on `yt-dlp`, `ffmpeg`, Claude's `Read` tool, and Whisper via Groq / OpenAI / any OpenAI-compatible server.

## Star History

<a href="https://www.star-history.com/?repos=frinsen%2Fclaude-video&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=frinsen/claude-video&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=frinsen/claude-video&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=frinsen/claude-video&type=date&legend=top-left" />
 </picture>
</a>
