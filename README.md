# /watch

**Give Claude the ability to watch any video.**

Claude Code:
```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

claude.ai (web): [download `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) and drop it into Settings → Capabilities → Skills.

Codex / generic skills:
```bash
git clone https://github.com/bradautomates/claude-video.git ~/.codex/skills/watch
```

Zero config to start — `yt-dlp` and `ffmpeg` install on first run via `brew` on macOS (Linux/Windows print exact commands). Captions cover most public videos for free. Whisper API key is only needed when a video has no captions.

---

Claude can read a webpage, run a script, browse a repo. What it can't do, out of the box, is *watch a video*. You paste a YouTube link and it has to either guess from the title or pull a transcript that's missing 90% of what's on screen.

With Claude Video `/watch` you can paste a URL or a local path, ask a question, and Claude downloads the source, extracts coverage-preserving scene-aware frames, adds motion and sound evidence for short clips, reports source chapters, and cleans a timestamped caption transcript (or uses Whisper). By the time it answers, it has evidence for what was shown, how it moved, what was said, and how the soundtrack changed.

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## Why this exists

I built this because I'm constantly using video to keep up with content. If I see a YouTube video that's blowing up, I want to know how the creator structured the hook — what's on screen in the first 3 seconds, what they said, why it worked. That used to mean watching it myself with a notepad. Now I just paste the URL and ask.

The other half is summarization. Most YouTube videos don't deserve 20 minutes of my attention. I hand the URL to Claude, it pulls the transcript, and tells me what actually happened. If the visual matters, frames come along too. If it's a podcast or a talking head, transcript is enough.

Claude is great at reading and synthesizing — but until now, video was the one input I couldn't hand it. Pasting a YouTube link got you nothing useful. `/watch` closes that gap.

## What people actually use it for

**Analyze someone else's content.** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude looks at the first frames, reads the opening transcript, breaks down the structure. Same for ad creative, competitor launches, podcast intros, anything where the *how* matters as much as the *what*.

**Break down a short film or AI-generated clip.** `/watch seedance-clip.mp4 analyze the camera work, edit, sound design, and story beats` adds eight-frame motion strips, adaptive 0.9-second detail strips for dense action, cut/flash-versus-motion evidence, waveform, spectrogram, silence, and low/mid/high-band onset timing. A five-second clip is treated as a temporal sequence, not five unrelated screenshots.

**Diagnose a bug from a video.** Someone sends you a screen recording of something broken. `/watch bug-repro.mov what's going wrong?` Claude watches the recording, finds the frame where the issue appears, describes what's on screen, often catches the cause without you ever opening the file.

**Summarize a video.** `/watch https://youtu.be/<long-thing> summarize this` does the obvious thing — pulls the structure, the key moments, what was actually said and shown. Faster than watching at 2x.

**Transcribe and analyze audio.** Voice notes, podcasts, meeting recordings — `/watch meeting.m4a what did they decide?` detects the missing video stream and produces a transcript. For a short source or focused range it also reports waveform, spectrogram, silence, loudness, and transient evidence, even when speech transcription is unavailable.

## How it works

1. **You paste a video and a question.** URL (anything yt-dlp supports — YouTube, Loom, TikTok, X, Instagram, plus a few hundred more) or a local path (`.mp4`, `.mov`, `.mkv`, `.webm`).
2. **`yt-dlp` downloads it.** For URLs, into a temp working directory. For local files, no download — just probed in place.
3. **`ffmpeg` extracts scene-aware frames on a duration-based budget.** A detection pass finds visual change points (cuts, slide flips, UI actions), reserves 25% of the budget for evenly spaced coverage, spends the rest on the strongest changes, and splits any remaining gaps. A visually dense intro cannot consume every frame while later sections disappear. The budget is duration-aware: ≤30s gets roughly one keyframe per second (up to 30; the shortest clips can reach 2 fps), 30-60s gets ~40, 1-3min gets ~60, 3-10min gets ~80, longer gets 100. Hard ceiling: 100 frames per pass. JPEGs are 512px wide by default; use 768-1024px for slides, terminals, or UI text. `--sampling uniform` restores constant-rate sampling.
4. **Short clips get a cinematic evidence pass.** For a source or focused range up to 60 seconds, FFmpeg samples visual change at 10 Hz, measures changed-pixel coverage and temporal persistence, and packs ordered frames into overview motion strips. Dense action also gets up to eight adaptive 0.9-second detail strips around the strongest events. The audio pass renders waveform and log-frequency spectrogram images, reports silence and loudness, and detects novelty independently in low, mid, and high frequency bands. It is local, free, and independent of Whisper. An optional learned layer can then listen to that selected audio and label effects, foley, ambience, music, texture, vocalizations, and likely story/edit functions through OpenAI or Gemini.
5. **The transcript comes from one of two places.** First try: `yt-dlp` pulls native captions — English if available, otherwise the best track the source has (manual beats auto-generated, original language beats translations), and the report says which kind you got. Rolling duplicate phrases from auto-caption cues are removed before the report is printed. Fallback: extract a mono 16 kHz audio clip (Opus for Groq, MP3 for OpenAI) and ship it to Whisper — Groq's `whisper-large-v3` (preferred — cheaper and faster) or OpenAI's `whisper-1`. Uploads are sized to the API's 25 MB cap, so up to ~1 hour of audio goes as a single seamless request; longer audio is split into overlapping windows stitched at the seam midpoints, each primed on the previous window's text. Failed uploads are bisected and retried, isolating bad regions instead of losing the transcript. Pass `--vocab "ComfyUI, Seedance, …"` to stop Whisper from garbling proper nouns.
6. **All evidence is handed to Claude.** The report lists precise keyframe timestamps, left-to-right motion strips, waveform/spectrogram paths, local sound and visual change markers, optional semantic sound events, and transcript segments. Claude reads the images together and aligns them on one timeline.
7. **Claude answers from the combined evidence.** It can separate camera movement from subject movement, inspect transition shape, connect a cut to a sound onset, and explain the story function of a sub-second beat. Signal-based labels stay tentative when the exact source cannot be verified.
8. **Everything is cached.** Downloads, frames, cinematic evidence, and Whisper transcripts persist under `~/.cache/watch/`, keyed by media identity and extraction params. Watching the same video again — or zooming into a section — skips the download and usually the extraction. The temp working directory is still cleaned up; the cache is not.

Source-provided chapter titles and ranges are included in the report. For long videos where you want complete coverage, the skill has a **deep analysis mode**: a scan pass uses those chapters (or infers better boundaries), chapter agents run dense focused passes in parallel within the host's concurrency limit, and the results are synthesized into a single timestamped analysis document. The cache makes those N passes affordable because the video downloads once.

## Frame budget — why it matters

Token cost is dominated by frames. Every frame is an image; image tokens add up fast. The duration-based budget keeps a 30-minute scan bounded and makes a focused 30-second window much denser.

| Duration | Default frame budget | What you get |
|----------|---------------------|--------------|
| ≤5 s | Up to 10 keyframes + motion strips | Sub-second temporal detail |
| 5 - 30 s | ~1 keyframe/s + motion strips | Dense — every key moment plus movement |
| 30 s - 1 min | ~40 frames | Still dense |
| 1 - 3 min | ~60 frames | Comfortable |
| 3 - 10 min | ~80 frames | Sparse but workable |
| > 10 min | 100 frames | "Sparse scan" warning — re-run focused |

When the user names a moment ("around 2:30", "the last 30 seconds", "from 0:45 to 1:00"), pass `--start` / `--end`. Focused mode gets denser per-second budgets, capped at 2 fps. Far more useful than a sparse pass over the whole thing.

## Short-form and cinematic analysis

`--analysis auto` is the default. It enables the cinematic pass whenever the full source or selected range is 60 seconds or shorter:

- **5-second clip:** up to five one-second motion strips, eight ordered frames per strip, plus sub-second visual and sound-change timing.
- **15-second clip:** roughly ten 1.5-second motion strips.
- **30-second clip:** roughly fifteen two-second motion strips.
- **Dense action:** up to eight additional 0.9-second strips centered on the strongest non-overlapping visual events.
- **Visual events:** changed-pixel coverage plus temporal hysteresis distinguishes isolated cut/flash-like discontinuities from localized or sustained movement.
- **Sound:** waveform, log-frequency spectrogram, silence/near-silence, loudness dynamics, and low/mid/high-band onset novelty. This is actual signal analysis, not transcript analysis.

The model uses those artifacts to build a shot-by-shot breakdown: framing and angle, camera versus subject motion, transition type, composition, lighting/color, sound design, audiovisual sync, and the story function of each beat. Visual classifications remain candidates rather than semantic truth: even high-coverage isolated change may be either a cut or a full-frame flash. Frequency-band onset labels describe where the signal changed, not the exact sound source, without supporting visual or learned semantic evidence.

Use `--analysis standard` to skip the extra pass, or `--analysis cinematic` to force it on a longer selected range.

### Semantic sound understanding

The FFmpeg layer tells us *when and how* the signal changes. The optional learned layer answers *what is probably audible*: footsteps or foley, an impact, whoosh, riser, room tone, environmental ambience, musical texture, a vocalization, and the likely filmmaking function of each event. It returns sub-second events in source time and marks events that coincide with locally measured audio transients or visual-change candidates.

It is intentionally opt-in for upload privacy. Either run it once explicitly:

```bash
python3 scripts/watch.py clip.mp4 --sound-semantics api --sound-provider openai
python3 scripts/watch.py clip.mp4 --sound-semantics api --sound-provider gemini
```

Or enable it for future short clips in `~/.config/watch/.env`:

```dotenv
SOUND_SEMANTICS_PROVIDER=openai  # or gemini
OPENAI_API_KEY=...
# GEMINI_API_KEY=...
```

`auto` is the default but runs the learned layer only when `SOUND_SEMANTICS_PROVIDER` is set. Merely having an OpenAI key for Whisper does not silently opt audio into another request. Only the selected range (maximum 60 seconds) is extracted and uploaded; the video and frames are never sent. The normalized result is cached, while the temporary MP3 is removed after the request. Labels and confidence values remain model judgments, so ambiguous sources should be verified against frames, motion strips, waveform/spectrogram, and transcript.

## Install

| Surface | Install |
|---------|---------|
| **Claude Code** | `/plugin marketplace add bradautomates/claude-video` then `/plugin install watch@claude-video` |
| **claude.ai** (web) | [Download `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **Codex** | `git clone https://github.com/bradautomates/claude-video.git ~/.codex/skills/watch` |
| **Manual / dev** | `git clone https://github.com/bradautomates/claude-video.git ~/.claude/skills/watch` |

### Claude Code

```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Update later with `/plugin update watch@claude-video`.

### claude.ai (web)

1. [Download `watch.skill`](https://github.com/bradautomates/claude-video/releases/latest) from the latest release.
2. Go to Settings → Capabilities → Skills.
3. Click `+` and drop the file in.

Enable "Code execution and file creation" under Capabilities first — the skill shells out to `ffmpeg` and `yt-dlp`, so it won't run without it.

### Codex

```bash
git clone https://github.com/bradautomates/claude-video.git ~/.codex/skills/watch
```

### Manual (developer)

```bash
git clone https://github.com/bradautomates/claude-video.git ~/.claude/skills/watch
```

## First run

On the first `/watch` call, the skill runs `scripts/setup.py --check`. If `ffmpeg` / `yt-dlp` aren't on your PATH, or no Whisper API key is set, it walks you through fixing it:

- **macOS** — auto-runs `brew install ffmpeg yt-dlp`.
- **Linux** — prints the exact `apt` / `dnf` / `pipx` commands.
- **Windows** — prints the `winget` / `pip` commands.
- **API key** — scaffolds `~/.config/watch/.env` (mode `0600`) with placeholders for Whisper and optional semantic sound configuration.

After setup, preflight is silent and `/watch` just works. The check is a sub-100ms lookup, so it doesn't slow you down on subsequent runs.

## Bring your own keys

Captions cover the majority of public videos for free. The Whisper fallback only kicks in when a video genuinely has no caption track — typically local files, TikToks, some Vimeos, and the occasional caption-less YouTube upload.

| Capability | What you need | Cost |
|------------|---------------|------|
| Download + native captions | `yt-dlp` + `ffmpeg` | Free |
| Whisper fallback (preferred) | [Groq API key](https://console.groq.com/keys) — `whisper-large-v3` | Cheap, fast |
| Whisper fallback (alt) | [OpenAI API key](https://platform.openai.com/api-keys) — `whisper-1` | Standard pricing |
| Local cinematic evidence | `ffmpeg` | Free; motion, transitions, waveform/spectrogram, silence, loudness, transients |
| Semantic sound (optional) | OpenAI `gpt-audio-1.5` or Gemini audio model | Provider pricing; selected range only, maximum 60s |
| Disable Whisper entirely | `--no-whisper` | Free; speech text is unavailable, but short-form visual and sound evidence still works |

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

- `--max-frames N` — lower the default 100-frame cap for a tighter token budget.
- `--resolution W` — bump frame width to 1024 px when Claude needs to read on-screen text (slides, terminals, code).
- `--fps F` — force uniform sampling at an explicit rate (capped at 2 fps).
- `--whisper groq|openai` — force a specific Whisper backend.
- `--no-whisper` — disable speech transcription; short-form visual and non-speech sound evidence still works.
- `--sampling scene|uniform` — frame selection strategy (default `scene`; auto-falls-back to uniform when no changes are detected).
- `--analysis auto|standard|cinematic` — short-form motion + non-speech sound evidence; auto enables it through 60 seconds.
- `--sound-semantics auto|api|off` — learned sound-event and story-function analysis. `auto` requires `SOUND_SEMANTICS_PROVIDER`; `api` explicitly enables it for this run.
- `--sound-provider auto|openai|gemini` — choose the semantic audio provider.
- `--sound-vocab "impact, cloth movement, sci-fi ambience"` — optional candidate concepts; hints rather than forced labels.
- `--no-cache` — bypass the persistent cache (`~/.cache/watch`).
- `--vocab "term1, term2, …"` — proper nouns expected in the audio; biases Whisper spelling so product and people names come out right.
- `--out-dir DIR` — keep working files somewhere specific (default: auto-generated tmp dir).

## Limits

- **Best single-pass accuracy: under 10 minutes.** Past that the script prints a sparse-scan warning. Re-run focused with `--start`/`--end`, or use deep analysis mode for complete long-video coverage.
- **Hard caps: 2 fps, 100 frames.** Frame count drives token cost; the script enforces this even when the auto-fps math would imply higher.
- **Semantic labels remain inferences.** Local cinematic evidence finds rhythm, silence, dynamics, frequency texture, motion, and likely change points. The optional audio model can propose event identities and story functions, but its confidence is not a calibrated probability and exact sources can still be ambiguous.
- **Whisper upload limit: 25 MB per request.** Uploads are auto-sized under it (~1 hour per request with Opus on Groq) and longer audio is windowed automatically — no manual splitting needed.
- **No private platforms.** This skill doesn't log into anything. Public URLs and local files only. If yt-dlp can't reach it without auth, neither can `/watch`.

## Structure

```
.
├── SKILL.md                 # skill contract — loaded by all three surfaces
├── scripts/
│   ├── watch.py             # entry point — orchestrates all evidence
│   ├── download.py          # yt-dlp wrapper
│   ├── frames.py            # ffmpeg frame extraction + auto-fps logic
│   ├── media_analysis.py    # motion strips + non-speech sound evidence
│   ├── sound_semantics.py   # optional OpenAI/Gemini semantic sound timeline
│   ├── transcribe.py        # VTT parsing + caption cleanup
│   ├── whisper.py           # Groq / OpenAI clients (pure stdlib)
│   ├── setup.py             # preflight + installer
│   └── build-skill.sh       # build dist/watch.skill for claude.ai upload
├── hooks/                   # SessionStart status hook (Claude Code only)
├── .claude-plugin/          # plugin.json + marketplace.json (Claude Code)
├── .codex-plugin/           # codex packaging
└── .github/workflows/       # release.yml — auto-builds watch.skill on tag push
```

## Develop

```bash
# Build the claude.ai upload bundle:
bash scripts/build-skill.sh      # → dist/watch.skill
```

Releasing: tag `vX.Y.Z`, push the tag. The workflow builds `dist/watch.skill` and attaches it to the GitHub release.

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Open source

MIT license.

Built on `yt-dlp`, `ffmpeg`, and Claude's multimodal `Read` tool. Whisper transcription via [Groq](https://groq.com) or [OpenAI](https://openai.com).

---

[github.com/bradautomates/claude-video](https://github.com/bradautomates/claude-video) · [LICENSE](LICENSE)
