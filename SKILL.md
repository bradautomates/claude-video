---
name: watch
description: Watch a video (URL or local path). Downloads with yt-dlp, extracts auto-scaled frames with ffmpeg, pulls the transcript from captions (or Whisper API fallback), writes a clean transcript.md, and hands the result to Claude so it can answer questions about what's in the video.
argument-hint: "<video-url-or-path> [question]"
allowed-tools: Bash, Read, AskUserQuestion
homepage: https://github.com/moreresults/claude-video
repository: https://github.com/moreresults/claude-video
author: moreresults
license: MIT
version: 0.4.2
user-invocable: true
---

# /watch — Claude watches a video

You don't have a video input; this skill gives you one. A Python script downloads the video, extracts frames as JPEGs, gets a timestamped transcript (native captions first, then Whisper API as fallback), and prints frame paths. You then `Read` each frame path to see the images and combine them with the transcript to answer the user.

> **Companion docs**
> - `REFERENCE.md` — quick reference: all flags, frame budgets, step summaries, Step 4.5 render order, QA checklist
> - `SKILL_BUILD_INFRA.md` — how to update and distribute this skill across surfaces (Claude Code, claude.ai, Codex)

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

**If an API key is still missing after install:** use `AskUserQuestion` to ask the user whether they have a Groq API key (preferred — cheaper, faster) or an OpenAI key. Then write it into `~/.config/watch/.env` — set the matching `GROQ_API_KEY=...` or `OPENAI_API_KEY=...` line. If they don't want to set up Whisper, proceed with `--no-whisper` and tell them videos without native captions will come back frames-only.

**Structured mode (optional):** `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --json` emits `{status, first_run, missing_binaries, whisper_backend, has_api_key, config_file, platform}` where `status` is one of `ready | needs_install | needs_key | needs_install_and_key`. Use this when you need to branch on specifics (e.g. "is this the user's very first run?" → `first_run: true`).

Within a single session, you can skip Step 0 on follow-up `/watch` calls — once `--check` returned 0, nothing about the environment changes between turns.

## When to use

- User pastes a video URL (YouTube, Vimeo, X, TikTok, Twitch clip, most yt-dlp-supported sites) and asks about it.
- User points at a local video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.) and asks about it.
- User types `/watch <url-or-path> [question]`.

## Recommended limits

- **Best accuracy: videos under 10 minutes.** Frame coverage scales inversely with duration.
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

**Where to save (read this BEFORE running):**

The script writes to a throwaway tmp dir by default. For anything the user wants to keep — articles, research notes, anything they might reference later — you must pass `--save-dir` so the script allocates a sanitised UCID-named folder. **Never construct the folder name yourself by hand** and never `mv` the tmp dir into place under a freehand name; the previous outputs of this project accumulated parentheses, ampersands, smart quotes, and other chars that silently broke the downstream PDF/DOCX pipeline because folder naming was being done by Claude, not by the script.

Decision rule:

1. If the repo has a `CLAUDE.md` (or other project memory) that names a specific save dir — e.g. `outputs/` for this project — pass `--save-dir <that-dir>`. Do not ask first; just use it.
2. If the user explicitly says "don't save" / "just look" / "throwaway", omit `--save-dir` and let it land in tmp. Clean up at Step 5.
3. Otherwise ask the user once (via `AskUserQuestion`) where to save before running, with "skip (use a temp dir)" as one option. Don't ask on follow-up `/watch` calls in the same session — reuse the previous answer.

The folder name the script picks (`UCID-NNNN YYYY-MM-DD Short Title`) is the canonical name. Don't rename, prefix, or "improve" it — other scripts (the migrator, the markdown-to-apple deliverable pipeline) depend on the exact form.

Optional flags:
- `--start T` / `--end T` — focus on a section. Accepts `SS`, `MM:SS`, or `HH:MM:SS`. When either is set, fps auto-scales denser (see "Focusing on a section" below).
- `--max-frames N` — lower the cap for tighter token budget (e.g. `--max-frames 40`)
- `--resolution W` — change frame width in px (default 512; bump to 1024 only if the user needs to read on-screen text)
- `--hires-resolution W` — change hi-res frame width in px (default 1600, never upscales beyond source). Hi-res copies sit on disk for publication use and are never read into Claude's context.
- `--fps F` — override auto-fps (clamped to 2 fps max)
- `--out-dir DIR` — keep working files somewhere specific (default: an auto-generated tmp dir)
- `--save-dir DIR` — save into `DIR/<UCID YYYY-MM-DD ShortTitle>/`, where the UCID is auto-allocated (e.g. `UCID-0001`, `UCID-0002`, …) by scanning existing subfolders of `DIR` for the highest used number and incrementing. The short title is truncated to ~40 chars at a word boundary and stripped of parentheses, ampersands, smart quotes, shell metacharacters, and any other char that has caused tooling failures downstream (see `scripts/safe_name.py`). Use this for permanent saves; use `--out-dir` only when you need a specific exact path. The two flags are mutually exclusive.
- `--ucid-prefix XYZ` — override the UCID prefix (default `UCID`). Only meaningful with `--save-dir`. Each prefix has its own independent counter.
- `--whisper groq|openai` — force a specific Whisper backend (default: prefer Groq if both keys exist)
- `--no-whisper` — disable the Whisper fallback entirely (frames-only if no captions)

### Focusing on a section (higher frame rate)

When the user asks about a specific moment — "what happens at the 2 minute mark?", "zoom into 0:45 to 1:00", "the first 10 seconds" — pass `--start` and/or `--end`. The script switches to focused-mode budgets, which are denser than full-video budgets (still capped at 2 fps):

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
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" video.mp4 --start 50 --end 60

# Zoom into 2:15 → 2:45 at 3 fps (90 frames)
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 2:15 --end 2:45 --fps 3

# From 1h12m to the end of the video
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "$URL" --start 1:12:00
```

**Step 3 — Read every frame path the script lists.** The Read tool renders JPEGs directly as images for you. Read all frames in a single message (parallel tool calls) so you see them together. The frames are in chronological order with a `t=MM:SS` timestamp so you can align them to the transcript.

**Step 4 — answer the user.** You now have two streams of evidence:
- **Frames** — what's on screen at each timestamp
- **Transcript** — what's said at each timestamp. The report's header shows the source (`captions` = yt-dlp pulled native subs; `whisper (groq)` or `whisper (openai)` = transcribed by API).

The script also writes a clean `transcript.md` to the working directory. Tell the user its path so they can open it directly — it contains a header with video metadata followed by the full timestamped transcript as plain text with no VTT tags or duplicate cues.

If the user asked a specific question, answer it directly citing timestamps. If they didn't ask anything, summarize what happens in the video — structure, key moments, notable visuals, spoken content.

### Step 4b — citing hi-res frames for publication

Every frame is written twice: a 512px copy Claude `Read`s and a 1600px copy in
the `hires/` subdir of the working directory, indexed by the same filename
(`frame_0023.jpg` in both dirs is the same moment).

When the user asks for frames for the newsletter, a blog post, an email, or any
use outside this chat — by description ("the laptop reveal"), by timestamp
("the frame at 2:15"), or by curation ("pick the 5 best for the newsletter") —
cite the `hires_path` from the JSON / report, **not** the lo-res `path` you
`Read` for analysis. The hi-res copies are publication-grade JPEGs at `-q:v 2`
(visually lossless); the lo-res copies are preview-grade at `-q:v 4` and will
look soft when embedded.

**Always remind the user to copy the cited hi-res files out of the working
directory before any cleanup step** — they live inside the same tmp dir as the
lo-res frames and the downloaded video, and get deleted with everything else.
A one-liner suffices:

> `cp <hires_path> ~/path/to/issue/assets/`

Hi-res frames cost zero context tokens — they sit on disk untouched. The lo-res
set is the only one in Claude's context window.

## Step 4.5 — author the newsletter article

This step fires automatically whenever `--save-dir` was passed and the user
has not opted out. Skip if:

- The user said "just save" / "no article" / "don't write it up"
- `--save-dir` was not used
- No transcript is available (frames-only run)
- Fewer than 3 frames in `hires/`

When skipping, log one line:
`[watch] article generation skipped (<reason>)`
and proceed to Step 5.

### Produces

Three files in a `business assets/` subfolder inside the UCID folder: `business-article.md`, `business-article.docx`, `business-article.pdf`. The `.md` is **post-humanizer**; the `.docx` and `.pdf` are rendered from it, so all three carry the humanized prose.

Create the subfolder before writing anything:

```bash
mkdir -p "<UCID-folder>/business assets"
```

### Author against the MOR-950 style guide

The operational style guide lives at:

`/Users/jameswatson/Dropbox/MoreResults/delivery/theaiteardown/standards/writing/STYLE_GUIDE_v2_1.md`

If readable, `Read` it first and apply verbatim. Otherwise apply the abbreviated rules below.

Note on versioning: if a higher-numbered `STYLE_GUIDE_v*.md` exists in the same `standards/writing/` directory, prefer the highest version number.

#### Render order (top to bottom)

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

#### Word count

700–2,000. Length follows source complexity. No padding, no compression.

#### Frame embedding

3–5 hi-res frames distributed across The Teardown section (not clustered). Per frame:

```
![<one-line caption describing what is on screen>](<absolute-path-to-hires-frame>)
```

Use absolute paths from the watch report (the `hi-res:` field on each frame line). Per MOR-950 v2.1: "Absolute local paths when the Claude Code render pipeline requires them for docx/PDF output." The renderer's `_parser.js` handles parens, spaces, and other special characters in paths.

Frame selection heuristic — pick frames that:

- Land at a moment the transcript names something concrete (a tool, a step, an artefact)
- Show visually distinct content from each other
- Span the video's time range, not clustered in one minute
- The first selected frame is a strong opener (intro shot, title card, hero moment)

Captions describe what is on screen — one declarative line. Not a paraphrase of the speaker; what the reader will SEE.

#### Voice — non-negotiable

- Third-person analytical observer. British English. No first person. No exclamation marks. No emojis. No engagement-bait questions.
- Contractions only in headings, opening reframe punches, and direct quotes. Expository prose uses "does not" / "is not" / "are not".
- Every analytical body paragraph contains a number, a name, a tool, a source claim, or a worked example. Exempt: short transitions, the Summary close, the Watch the Original line.

#### Four mandatory moves

1. **Open with a reframe.** State the surface narrative, then go underneath it.
2. **Anchor abstractions to named artefacts.** If you cannot name the artefact, the section is too abstract.
3. **Attribute every number to the creator.** *"Puru reports time per request dropped from 20–30 minutes to under 3 minutes."* Never the number alone.
4. **End with one concrete action.** Not a summary. Not a CTA. One thing the reader can do.

#### Source Fidelity (publication-blocking under MOR-938)

- Tool and feature names exact — *Make.com* not *Make*; *ChatGPT* not *GPT*; *Claude Code* not *Claude*
- Creator claims reported, not verified
- Steps match the source — no reordering, no invented steps, no inferred outcomes
- Screenshots correspond to the right step in the source

#### Forbidden words and phrases — strip on sight

Words: delve, leverage (verb), crucial, robust, seamless, holistic, unlock, unleash, supercharge, harness, transformative, paradigm, revolutionary, game-changer, cutting-edge, state-of-the-art, elevate, embark, navigate (metaphor), tapestry, myriad, plethora, journey (metaphor), bespoke (filler), curated (filler), vibrant.

Phrases: "in today's fast-paced world", "dive deep", "delve into", "explore the depths", "at the intersection of", "the future of [X]", "the world of [X]", "in the realm of", "stands as a testament to", "as an AI", "AI-powered" (filler), "harness the power of", "ecosystem of" (buzzword), "unleash your potential".

Constructions: engagement-bait questions ("Are you ready to…?"), empty triads ("faster, smarter, better"), false humility ("here are just a few"), throat-clearing openers ("in this article, we will explore…").

#### Numbers and formatting

- Ranges: en-dash, no spaces — *20–30 minutes*, *60–90%*
- Currency ranges: "to" — *$1,500 to $3,000*, *2 to 3x*
- Currency: symbol-prefixed — *$1*, *€5/month*, *£500*
- Percentages: numeral plus % — *50%*, never *fifty percent*
- Thousands: comma — *6,000*
- Millions: spelled when round (*4 million*), numerals with M when not (*4.2M*)
- Approximate: *around* or *roughly*, never *approximately*
- Quotes: preserve original spelling verbatim including American spelling

#### Headings and lists

- H1 for the title. H2 for sections. H3 only when each named element of a framework needs a full subsection.
- 3+ medium concepts under one H2 → bolded paragraph lead-ins (`**Tool search.** Instead of loading…`).
- Bulleted for unordered, numbered for sequences. Bolded lead-ins when items are named concepts; no bolding when items are full sentences.
- Code formatting for commands, flags, file paths, env vars, CLI snippets. Product/tool names plain.
- Italics: single-word emphasis, inline short quotes. Max one blockquote pull-quote per article.

### Draft the article with james-watson-copywriter

Invoke the `james-watson-copywriter` skill in **Mode A — fresh write** to draft the full article body.

The MOR-950 render order (sections 1–10), word count (700–2,000), frame placement, and structural rules take precedence over any format defaults in the copywriter skill. Apply the skill's voice rules, kill list, and mandatory Final Pass (specificity audit → rate >9 → three final questions) to all prose sections.

Additional constraints that override the copywriter skill's defaults:

- **Third-person only.** MOR-950 voice is third-person analytical observer; the copywriter's "direct second-person / always you" does not apply here.
- **British English in paraphrase.** The copywriter skill defaults to the author's register; here British English is mandatory except inside direct quotes.
- **No exclamation marks.** MOR-950 hard rule; the copywriter's "confrontational closings" must be recast without them.
- **Source fidelity (MOR-938) overrides specificity rules.** Never invent steps, numbers, or outcomes. Every specific must be sourced from the video transcript or metadata — not from the copywriter's preference for sharp claims.

### Pre-render QA — run before invoking the renderer

Run every check. Fix in place if any fails. Do not call the renderer until clean.

- [ ] Source attribution blockquote at top
- [ ] Headline or tease concept carries: specific outcome + recognisable archetype + surprise element + easy entry. Easy entry may live in the tease and intro rather than the H1.
- [ ] Opening paragraph reframes the surface story
- [ ] Word count 700–2,000
- [ ] 3–5 images with one-line frame-describing captions
- [ ] Every specific number anchored to the creator
- [ ] *"X. It is not Y."* reframe count ≤ 4 across the article
- [ ] Zero forbidden words or phrases
- [ ] Final paragraph gives one concrete action
- [ ] British English in paraphrase; original spelling preserved in quotes
- [ ] Tool and feature names match the products exactly
- [ ] No invented steps, outcomes, or reordering of the source

### Write the file

Write the validated article to `<UCID-folder>/business assets/business-article.md`.

### Humanize the prose — mandatory, every run

Invoke the `humanizer` skill on `<UCID-folder>/business assets/business-article.md` and overwrite the file in place with the humanized output. This pass is **not optional** and applies to every run — there is no opt-out, no "skip if short", no "skip if voice already feels human". The renderer must operate on the humanized `.md`, so this step runs **before** `md_to_docx.js` / `md_to_pdf.js`.

The humanizer strips the AI-writing tells the style guide alone does not catch: em-dash overuse, inflated symbolism, vague attributions, rule-of-three, "It is not X. It is Y." reframe spam, AI vocabulary, negative parallelisms, filler phrases.

Constraints when applying:

- Preserve every `![caption](path)` image embed verbatim — paths, captions, and order.
- Preserve every blockquote, including the source-attribution blockquote at the top and any direct quotes from the creator. Direct quotes are source material and must not be reworded.
- Preserve all numbers, tool names, and creator attributions — these are MOR-938 source-fidelity constraints and override the humanizer's stylistic preferences.
- Preserve heading hierarchy (H1/H2/H3) and the MOR-950 render order.
- The MOR-950 voice constraints (third-person, British English, no first person, no exclamation marks, contractions only in headings/quotes) still apply after humanization. If the humanizer's output violates any, re-fix in place.

After the pass, re-run the pre-render QA checklist on the humanized file. The em-dash, reframe-count, and forbidden-word checks are the ones most likely to shift; fix in place if anything regressed.

### Invoke the renderer

Both commands. `npm root -g` resolves the global node_modules path so `require("docx")` works:

```bash
NODE_PATH=$(npm root -g) node ~/.claude/skills/markdown-to-apple-deliverable/scripts/md_to_docx.js \
  "<UCID-folder>/business assets/business-article.md" --out "<UCID-folder>/business assets/"

NODE_PATH=$(npm root -g) node ~/.claude/skills/markdown-to-apple-deliverable/scripts/md_to_pdf.js \
  "<UCID-folder>/business assets/business-article.md" --out "<UCID-folder>/business assets/"
```

Each prints `OK business-article` on success or `ERR business-article — <reason>` on failure.

### Verify outputs and clear the sentinel

```bash
ls -la "<UCID-folder>/business assets/business-article".{md,docx,pdf}
```

If any are missing, surface the renderer error to the user. Leave the `.md` in place — it can be re-rendered manually.

Once all three (`.md`, `.docx`, `.pdf`) exist, **delete the sentinel file** the watch script writes when `--save-dir` is used:

```bash
rm -f "<UCID-folder>/business-article.REQUIRED"
```

The sentinel exists specifically so batch runs that bypass this Skill (e.g. calling `scripts/watch.py` directly via Bash) leave a visible on-disk flag for unfinished Step 4.5 work. To audit the entire output tree for missing articles:

```bash
find <save-dir> -name 'business-article.REQUIRED'
```

Empty output means every UCID folder is complete. Any path printed needs Step 4.5 re-run.

### Step 4.6 — MACP cloud registration (opt-in)

**Only run this step when `--macp-register` was passed to `watch.py`.**

After the sentinel has been deleted (i.e., after `.md`, `.docx`, and `.pdf` all exist), run the adapter:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/macp_adapter.py" register \
  --folder "<UCID-folder>" [--dry-run]
```

If `--macp-dry-run` was also passed, add `--dry-run` — the adapter will validate files and print a payload summary without making any network calls.

If optional overrides were passed (e.g. `--macp-brand-id`), forward them as the corresponding `--brand-id` etc. options. The exact command is printed in the `watch.py` report under the `---` separator when `--macp-register` was used.

**Skip this step if any of the following are true:**
- `--macp-register` was not passed
- `--save-dir` was not used (no persistent UCID folder)
- Article generation was skipped for any reason (e.g. no transcript, user opted out)
- The business-article.REQUIRED sentinel is still present

When skipping: `[watch] MACP registration skipped: <reason>`

**On success**, the adapter prints:

```
[watch] MACP registration complete
session_id: <uuid>
campaign_id: <uuid>
...
admin: https://macp-blond.vercel.app/admin/ingestion/sessions/<session_id>
```

Local files are always left intact — never delete the UCID folder contents after MACP registration.

### Failure modes

| failure | action |
| -- | -- |
| Pre-render QA check fails | Fix in place, recheck, do not invoke renderer until clean |
| Humanizer pass skipped or unavailable | Do not invoke renderer. Resolve the skill availability and re-run the pass — the renderer must operate on humanized prose. |
| Humanizer regressed an MOR-950 voice rule (first person, exclamation mark, American spelling in paraphrase, etc.) | Fix in place on the humanized `.md`, re-run pre-render QA, then invoke renderer. |
| Renderer prints `ERR` | Surface error to user. Leave `.md` in place. Do not retry automatically. |
| Chrome missing at default path | `md_to_pdf.js` fails. Tell user: override with `CHROME_BIN=/path/to/chrome`. |
| `docx` package not installed globally | `md_to_docx.js` fails with module-not-found. Tell user: `npm install -g docx`. |

### Confirm to the user

After successful render:

> Wrote `business-article.md`, `business-article.docx`, and `business-article.pdf` to `<UCID-folder>/business assets/`. Skim the `.md` to check voice and structure; re-run only the renderer if you edit it.

**Step 5 — clean up.** The script prints a working directory at the end. **Before deleting it, check whether the user wants any hi-res frames for publication** — those live at `<work>/hires/frame_NNNN.jpg` and will be lost with the rest of the working dir. If the user has flagged frames for the newsletter, blog, or any external use, prompt them to copy the relevant `hires/*.jpg` files to a persistent location first, or do it for them. Then, if there are no expected follow-ups, delete the working dir with `rm -rf <dir>`. If follow-ups are likely, leave it in place.

## Transcription

The script gets a timestamped transcript in one of two ways:

1. **Native captions (free, preferred).** yt-dlp pulls manual or auto-generated subtitles from the source platform if available.
2. **Whisper API fallback.** If no captions came back (or the source is a local file), the script extracts audio (`ffmpeg -vn -ac 1 -ar 16000 -b:a 64k`, ~0.5 MB/min) and uploads it to whichever Whisper API has a key configured:
   - **Groq** — `whisper-large-v3`. Preferred default: cheaper, faster. Get a key at console.groq.com/keys.
   - **OpenAI** — `whisper-1`. Fallback. Get a key at platform.openai.com/api-keys.

Both keys live in `~/.config/watch/.env`. The script prefers Groq when both are set; override with `--whisper openai` to force OpenAI. Use `--no-whisper` to skip the fallback entirely.

## Failure modes and handling

- **Setup preflight failed** → run `python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py"` (auto-installs ffmpeg/yt-dlp via brew on macOS, scaffolds the `.env`). For API key, ask the user via `AskUserQuestion` and write it to `~/.config/watch/.env`.
- **No transcript available** → captions missing AND (no Whisper key OR Whisper API failed). Script prints a hint pointing to setup. Proceed frames-only and tell the user.
- **Long video warning printed** → acknowledge it in your answer. Offer to re-run focused on a specific section via `--start`/`--end` rather than a sparse full-video scan.
- **Download fails** → yt-dlp's error goes to stderr. If it's a login-required or region-locked video, tell the user plainly; do not keep retrying.
- **Whisper request fails** → the error is printed to stderr (likely: invalid key, rate limit, or 25 MB upload limit on a very long video). The report will say "none available" for transcript. You can retry with `--whisper openai` if Groq failed (or vice versa).

## Token efficiency

This skill burns tokens primarily on frames Claude `Read`s. Order of magnitude:
- 80 lo-res frames at 512px wide is roughly 50-80k image tokens depending on aspect ratio.
- The transcript is cheap (a few thousand tokens at most for a 10-minute video).
- Hi-res copies are written in the same ffmpeg pass at zero context cost — they exist on disk only.
- Bumping `--resolution` (the lo-res Claude reads) to 1024 roughly quadruples the image tokens per frame. Use `--hires-resolution` instead to tune publication quality without paying token cost.

If you already watched a video this session and the user asks a follow-up, do **not** re-run the script — you already have the frames and transcript in context. Just answer from what you have.

## Security & Permissions

**What this skill does:**
- Runs `yt-dlp` locally to download the video and pull native captions when the source supports them (public data; the request goes directly to whatever host the URL points at)
- Runs `ffmpeg` / `ffprobe` locally to extract frames as JPEGs and, when Whisper is needed, a mono 16 kHz audio clip
- Sends the extracted audio clip to Groq's Whisper API (`api.groq.com/openai/v1/audio/transcriptions`) when `GROQ_API_KEY` is set (preferred — cheaper, faster)
- Sends the extracted audio clip to OpenAI's audio transcription API (`api.openai.com/v1/audio/transcriptions`) when `OPENAI_API_KEY` is set and Groq is not, or when `--whisper openai` is forced
- Writes the downloaded video, frames, audio, and an intermediate transcript to a working directory under the system temp dir (or `--out-dir` if specified) so Claude can `Read` them
- Reads / creates `~/.config/watch/.env` (mode `0600`) to store the Whisper API key(s) and a `SETUP_COMPLETE` marker. As a fallback, also reads `.env` in the current working directory

**What this skill does NOT do:**
- Does not upload the video itself to any API — only the extracted audio goes out, and only when native captions are missing AND Whisper is not disabled with `--no-whisper`
- Does not access any platform account (no login, no session cookies, no posting)
- Does not share API keys between providers (Groq key only goes to `api.groq.com`, OpenAI key only goes to `api.openai.com`)
- Does not log, cache, or write API keys to stdout, stderr, or output files
- Does not persist anything outside the working directory and `~/.config/watch/.env` — clean up the working directory when you're done (Step 5)

**Bundled scripts:** `scripts/watch.py` (entry point), `scripts/download.py` (yt-dlp wrapper), `scripts/frames.py` (ffmpeg frame extraction), `scripts/transcribe.py` (caption selection + Whisper orchestration), `scripts/whisper.py` (Groq / OpenAI clients), `scripts/safe_name.py` (folder-name sanitiser for `--save-dir`), `scripts/setup.py` (preflight + installer)

Review scripts before first use to verify behavior.
