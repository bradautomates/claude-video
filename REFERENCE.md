# /watch — Quick Reference — v1.3

## Basic syntax

```
/watch <url-or-path> [flags] [question]
```

## Where output is saved

Pass `--save-dir` to keep all assets in a named folder. **Never construct the folder name yourself** — let the script do it. The folder name is auto-generated as:

```
UCID YYYY-MM-DD Short Title
```

- **UCID** — auto-allocated unique content ID, e.g. `UCID-0001`. The script scans `--save-dir` for existing `UCID-NNNN*` subfolders, picks the highest `N`, and increments. Override the prefix with `--ucid-prefix XYZ` if you want a different namespace (each prefix has its own counter).
- **YYYY-MM-DD** — the video's publish date from yt-dlp metadata (`upload_date`), falling back to today's date.
- **Short Title** — sanitised and truncated to ~40 chars at a word boundary. Parentheses, ampersands, smart quotes, apostrophes, exclamation marks, shell metacharacters, and any chars unsafe across POSIX/Windows filesystems are stripped. This is deliberate: earlier output folders that contained these characters silently broke the markdown-to-apple PDF/DOCX pipeline.

```
/watch https://youtu.be/abc --save-dir outputs/ summarize this
```

→ creates `outputs/UCID-0014 2026-05-16 Every Claude Code Memory System/`

| raw metadata title | folder name |
|-|-|
| `My Claude Code Can INSTANTLY Watch Any Video (Here's How)` | `UCID-0001 2026-04-29 My Claude Code Can INSTANTLY Watch Any` |
| `Skill Chaining in Claude OS is INSANE (Don't Fall Behind!)` | `UCID-0002 2026-05-14 Skill Chaining in Claude OS is INSANE` |

The full untruncated title is preserved inside `download/video.info.json` and the generated `transcript.md`/`business-article.md`.

### `--save-dir` decision rule

Claude follows this before every run:

1. If the repo has a `CLAUDE.md` naming a specific save dir — use it. Do not ask.
2. If the user explicitly says "just look" / "don't save" / "throwaway" — omit `--save-dir`, use tmp, clean up after.
3. Otherwise — ask once via `AskUserQuestion` with "skip (use a temp dir)" as an option. Reuse the answer for follow-up calls in the same session.

### `--out-dir` vs `--save-dir`

These are mutually exclusive:
- `--save-dir DIR` — saves into `DIR/<UCID YYYY-MM-DD ShortTitle>/` with auto-naming. Use for permanent saves.
- `--out-dir DIR` — uses this exact directory with no naming logic. Use only when you need a specific exact path.

### Folder contents

```
UCID-NNNN YYYY-MM-DD Short Title/
  download/
    video.mp4
    video.info.json
    video.en.vtt        ← captions if available
  frames/
    frame_0001.jpg      ← 512px lo-res, what Claude Reads
    frame_0002.jpg
    …
  hires/
    frame_0001.jpg      ← 1600px publication-grade (same filenames)
    frame_0002.jpg
    …
  audio.mp3             ← only if Whisper was used
  transcript.md         ← clean timestamped transcript, path reported to user
  business-article.REQUIRED   ← sentinel; deleted after Step 4.5 completes
  business assets/
    business-article.md
    business-article.docx
    business-article.pdf
```

## Steps

### Step 0 — Setup preflight

Runs automatically on every `/watch` invocation (`python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --check`). On exit 0: **say nothing to the user** — do not announce "setup is complete." Proceed to Step 1 silently. The only acceptable user-visible output from Step 0 is when remediation is required.

On non-zero exit:

| Exit | Meaning | Action |
|-|-|-|
| `2` | Missing binaries (`ffmpeg`/`ffprobe`/`yt-dlp`) | Run installer |
| `3` | No Whisper API key | Scaffold `.env`, ask user for key via `AskUserQuestion` |
| `4` | Both missing | Run installer + ask for key |

**Structured mode:** `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --json` emits `{status, first_run, missing_binaries, whisper_backend, has_api_key, config_file, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. Use when you need to branch on specifics (e.g. `first_run: true` → run the full wizard).

Within a single session, skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing changes between turns.

### Step 1 — Parse input

Separate the video source (URL or path) from any question and flags.

### Step 2 — Run the watch script

Path resolution depends on how the skill is installed:

```bash
# Local skill install
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "<source>" [flags]

# Claude Code plugin install
python3 "${CLAUDE_PLUGIN_ROOT}/skills/watch/scripts/watch.py" "<source>" [flags]
```

If you need a single invocation that works for both, define a resolver at the top of the run:

```bash
WATCH_SKILL_DIR="${CLAUDE_SKILL_DIR:-${CLAUDE_PLUGIN_ROOT}/skills/watch}"
python3 "$WATCH_SKILL_DIR/scripts/watch.py" "<source>" [flags]
```

**Windows note:** use `python` not `python3` — the `python3` command on Windows is the Microsoft Store stub.

### Step 3 — Read frames

Read all frame paths in a single message (parallel tool calls). Frames are chronological with `t=MM:SS` timestamps. Also tell the user the path to `transcript.md`.

### Step 4 — Answer

Answer using both streams: frames (what's on screen) and transcript (what's said). Transcript header shows source: `captions`, `whisper (groq)`, or `whisper (openai)`.

### Step 4b — Hi-res frames for publication

When the user asks for frames for a newsletter, blog, or any external use — cite `hires/frame_NNNN.jpg`, not the lo-res copy Claude Read. Hi-res frames are 1600px at `-q:v 2` (visually lossless). Remind the user to copy hi-res files out before any cleanup — they live in the same tmp dir and are deleted with it.

### Step 4.5 — Author the business article

Fires automatically when `--save-dir` was passed and the user has not opted out. Skipped when:
- `--save-dir` was **not** used (throwaway/tmp run)
- User said "just save" / "no article" / "don't write it up"
- No transcript available (frames-only run)
- Fewer than 3 frames in `hires/`

When skipping, log one line: `[watch] article generation skipped (<reason>)`

**Produces:** `<UCID-folder>/business assets/business-article.md`, `business-article.docx`, `business-article.pdf`

Workflow:
1. `Read` the MOR-950 style guide at `/Users/jameswatson/Dropbox/MoreResults/delivery/theaiteardown/standards/writing/STYLE_GUIDE_v2_1.md` (use highest version number if multiple versions exist)
2. Draft via `james-watson-copywriter` skill in Mode A (fresh write), applying MOR-950 render order and overrides
3. Run pre-render QA checklist (see below)
4. Write validated `.md` to `business assets/`
5. Run `humanizer` skill — overwrites `.md` in place — mandatory, no opt-out
6. Re-run pre-render QA on humanized file; fix in place if any regressions
7. Invoke renderer (both commands):

```bash
NODE_PATH=$(npm root -g) node ~/.claude/skills/markdown-to-apple-deliverable/scripts/md_to_docx.js \
  "<UCID-folder>/business assets/business-article.md" --out "<UCID-folder>/business assets/"

NODE_PATH=$(npm root -g) node ~/.claude/skills/markdown-to-apple-deliverable/scripts/md_to_pdf.js \
  "<UCID-folder>/business assets/business-article.md" --out "<UCID-folder>/business assets/"
```

8. Verify all three outputs exist; if any missing, surface the renderer error
9. Delete the sentinel:

```bash
rm -f "<UCID-folder>/business-article.REQUIRED"
```

Audit for unfinalised UCID folders across the whole output tree:

```bash
find <save-dir> -name 'business-article.REQUIRED'
```

Empty output = every UCID folder is complete.

**MOR-950 render order** (sections 1–10):

1. Source attribution blockquote — `> Based on: "<Original Title>" by <Creator>`
2. H1 headline — result-led: archetype + outcome + system
3. Source line — italic, *"From <Creator>'s video on <Platform>, published <date>"*
4. Insights — opening reframe paragraph
5. At a Glance — 5 tags: tool, use case, difficulty, time to implement, freshness
6. Why It Matters — 2–3 analytical paragraphs
7. The Teardown — longest section, worked walkthrough, embeds frames inline
8. What to Steal — the reusable artefact
9. Summary — closing action paragraph
10. Watch the Original — one or two lines, link out

**Frame selection heuristics** (pick 3–5 hi-res frames for The Teardown):
- Land at a moment the transcript names something concrete (a tool, a step, an artefact)
- Show visually distinct content from each other — not clustered in the same minute
- Span the video's full time range
- First selected frame is a strong opener (title card, intro shot, hero moment)
- Captions describe what the reader will SEE — one declarative line, not a paraphrase of the speaker

**Pre-render QA checklist** (fix in place before calling renderer; do not proceed until all pass):

- [ ] Source attribution blockquote at top
- [ ] Headline or tease concept carries: specific outcome + recognisable archetype + surprise element + easy entry (easy entry may live in the tease/intro rather than the H1 itself)
- [ ] Opening paragraph reframes the surface story
- [ ] Word count 700–2,000
- [ ] 3–5 hi-res images with one-line frame-describing captions
- [ ] Every specific number anchored to the creator
- [ ] `"X. It is not Y."` reframe count ≤ 4 across the article
- [ ] Zero forbidden words or phrases
- [ ] Final paragraph gives one concrete action
- [ ] British English in paraphrase; original spelling preserved in quotes
- [ ] Tool and feature names match products exactly
- [ ] No invented steps, outcomes, or reordering of the source

**Humanizer preservation constraints** — the humanizer must not alter:
- Every `![caption](path)` image embed — paths, captions, and order unchanged
- Blockquotes (including source-attribution blockquote and direct creator quotes)
- All numbers, tool names, and creator attributions (MOR-938 source fidelity)
- Heading hierarchy (H1/H2/H3) and MOR-950 render order
- MOR-950 voice constraints: third-person, British English in paraphrase, no first person, no exclamation marks, contractions only in headings/quotes. If the humanized output violates any, fix in place before invoking the renderer.

**Article renderer failure modes:**

| failure | action |
|-|-|
| Pre-render QA fails | Fix in place, recheck, do not invoke renderer until clean |
| Humanizer unavailable | Do not invoke renderer. Resolve skill and re-run the pass |
| Humanizer regressed MOR-950 rule | Fix humanized `.md`, re-run QA, then invoke renderer |
| Renderer prints `ERR` | Surface error to user. Leave `.md`. Do not auto-retry |
| Chrome missing | `md_to_pdf.js` fails. Tell user: `CHROME_BIN=/path/to/chrome` |
| `docx` package not installed | `npm install -g docx` |

### Step 5 — Clean up

Check whether the user wants any hi-res frames for publication before deleting. Then `rm -rf <work-dir>` if no follow-ups expected.

### Direct script invocation

Do not treat `scripts/watch.py` as the complete workflow when using `--save-dir`. The script creates the video evidence and the `business-article.REQUIRED` sentinel, but Step 4.5 lives in the Skill. If you run the script directly (e.g. via Bash outside the Skill), audit for unresolved sentinels and run the article step separately:

```bash
find <save-dir> -name 'business-article.REQUIRED'
```

## Frame budgets

### Full-video (default)

| Duration | Frames | Notes |
|-|-|-|
| ≤30 s | ~30 | Dense |
| 30 s – 1 min | ~40 | Dense |
| 1 – 3 min | ~60 | Comfortable |
| 3 – 10 min | ~80 | Sparse |
| > 10 min | 100 | Warning printed — prefer `--start`/`--end` |

### Focused mode (`--start`/`--end`) — denser budgets

| Window | Frame rate | Cap |
|-|-|-|
| ≤5 s | 2 fps | 10 frames |
| 5–15 s | 2 fps | 30 frames |
| 15–30 s | ~2 fps | 60 frames |
| 30–60 s | ~1.3 fps | 80 frames |
| 60–180 s | ~0.6 fps | 100 frames |

Both modes hard-cap at 2 fps and 100 frames.

## Token efficiency

- 80 lo-res frames (512px) ≈ 50–80k image tokens depending on aspect ratio
- Transcript ≈ a few thousand tokens for a 10-minute video
- Hi-res copies are written in the same ffmpeg pass at zero context cost — on disk only, never Read
- Bumping `--resolution` to 1024 roughly quadruples image tokens per frame; use `--hires-resolution` instead to tune publication quality without paying token cost
- If the user asks a follow-up in the same session, do not re-run the script — answer from frames and transcript already in context

## Watching a specific section

```
/watch https://youtu.be/abc --start 2:15 --end 2:45 what's on screen?
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00         # from 1h12m to end
```

Transcript is auto-filtered to the same range. Frame timestamps are absolute (real video timeline).

## Common patterns

| Goal | Command |
|-|-|
| Summarize a video | `/watch <url> summarize this` |
| Analyze a hook / intro | `/watch <url> --end 0:30 break down this hook` |
| Debug a screen recording | `/watch bug.mov when does the UI break?` |
| Extract slides / text | `/watch <url> --resolution 1024 list the slide titles` |
| Save and produce article | `/watch <url> --save-dir outputs/` |
| Save only, skip article | `/watch <url> --save-dir outputs/ just save, no article` |
| Throwaway look, no save | `/watch <url> summarize this` (omit `--save-dir` entirely) |

## All flags

| Flag | Default | Purpose |
|-|-|-|
| `--save-dir DIR` | — | Base dir; auto-names subfolder `UCID YYYY-MM-DD Short Title` |
| `--ucid-prefix XYZ` | `UCID` | Override UCID prefix; each prefix has its own counter |
| `--out-dir DIR` | temp | Use this exact directory (no naming logic). Mutually exclusive with `--save-dir` |
| `--start T` | — | Range start (`SS`, `MM:SS`, or `HH:MM:SS`) |
| `--end T` | — | Range end |
| `--max-frames N` | 80 | Cap frame count (hard max 100) |
| `--resolution W` | 512 | Lo-res frame width in px (what Claude Reads). Use 1024 for on-screen text |
| `--hires-resolution W` | 1600 | Hi-res frame width in px for publication use. Never upscales beyond source |
| `--fps F` | auto | Override fps (capped at 2) |
| `--whisper groq\|openai` | auto | Force a Whisper backend |
| `--no-whisper` | — | Frames only; skip transcription entirely |
| `--macp-register` | — | Register publication assets with MACP after Step 4.5. Requires `--save-dir`. Performed by Skill after article render, not by watch.py directly |
| `--macp-dry-run` | — | Validate MACP payload only; no network calls |
| `--macp-base-url URL` | `$MACP_BASE_URL` | MACP base URL override |
| `--macp-brand-id UUID` | `$MACP_BRAND_ID` | MACP brand ID override |
| `--macp-created-by ID` | `$MACP_CREATED_BY` | MACP user ID override |
| `--macp-editorial-brief-id ID` | `$MACP_EDITORIAL_BRIEF_ID` | Editorial brief ID override |

## Transcript sources

1. **Native captions** — pulled free via yt-dlp (most YouTube, some Vimeo/TikTok)
2. **Whisper (Groq)** — fallback when no captions; `whisper-large-v3`, fast and cheap (preferred)
3. **Whisper (OpenAI)** — second fallback; `whisper-1`

Keys live in `~/.config/watch/.env` (mode `0600`). Also reads `.env` in the current working directory as fallback. Configure via `python3 <skill-dir>/scripts/setup.py`.

## MACP cloud registration

Register a completed UCID folder directly (Step 4.5 must be done first):

```bash
python3 scripts/macp_adapter.py register --folder "<UCID-folder>" --dry-run
python3 scripts/macp_adapter.py register --folder "<UCID-folder>"
```

### Required env vars (by presence; values never printed)

| Var | Purpose |
|-|-|
| `MACP_BASE_URL` | e.g. `https://macp-blond.vercel.app` |
| `MACP_REGISTRATION_TOKEN` | Bearer token for session/commit endpoints |
| `MACP_BRAND_ID` | Target brand UUID |
| `MACP_CREATED_BY` | MACP user ID used as `created_by` |
| `MACP_EDITORIAL_BRIEF_ID` | Required by commit `config_snapshot` |

### Files uploaded

| Role | Source path |
|-|-|
| `transcript_md` | `transcript.md` |
| `article_md` | `business assets/business-article.md` |
| `article_docx` | `business assets/business-article.docx` |
| `article_pdf` | `business assets/business-article.pdf` |
| `frame_hires` | `hires/frame_*.jpg` (1–100) |

**Not uploaded:** `frames/` lo-res images, `download/video.mp4`, audio, captions, logs.

### MACP failure modes

| Condition | Behaviour |
|-|-|
| `business-article.REQUIRED` still present | Fail before session creation |
| Any required file missing | Fail before session creation |
| Text file over 1 MB | Fail before session creation |
| 0 or >100 hi-res frames | Fail before session creation |
| 409 manifest mismatch | Fail clearly; no upload or commit |
| Upload non-2xx | Stop before commit; local files retained |
| Commit 400 missing upload | Report missing role; local files retained |
| Completed idempotent hit | Print existing IDs and admin link; no upload/commit |
| Local file source (no webpage_url) | Skip with diagnostic |

## Security summary

**What goes out:** extracted audio (mono 16 kHz) to Groq or OpenAI Whisper API — only when no native captions and `--no-whisper` not set.

**What stays local:** the video file itself, frames, transcript, all outputs.

**What it does not do:** no login, no platform sessions, no posting, no key sharing between providers, no stdout/stderr logging of keys, no writes outside `~/.config/watch/.env` and the working directory.

## Limits

- Best accuracy under 10 min; past that use `--start`/`--end` on the relevant section
- Whisper upload cap: ~25 MB (≈50 min of mono audio)
- Public URLs and local files only — no login-required platforms
