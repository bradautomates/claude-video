---
description: Remix a high-performing video for a new topic or offer while preserving the original's hook, pacing, story arc, offer placement, and retention mechanics.
argument-hint: "<video-url-or-path> for <user's topic or offer> [extra notes]"
allowed-tools: [Bash, Read, AskUserQuestion]
---

# /remix — Rewrite a high-performing video for a new offer while preserving the original retention structure

Use this command when the user wants the emotional and structural power of an existing video preserved while the message is rewritten for a new topic or offer.

## Argument parsing

1. Split the user input on the first instance of ` for ` (case-insensitive).
2. The text before `for` is the source video URL or local path.
3. The text after `for` is the user's new topic or offer.
4. If the input is missing, missing a source, missing a target, or the target is vague, ask a clarifying question with `AskUserQuestion`.

### Clarification rules

- If no arguments were provided, ask:
  - "Please give me the source video and the target topic or offer in the format `<source> for <topic or offer>`."
- If the input does not clearly contain `for`, ask:
  - "I need a source video and a target topic/offer. Please give them in the form `<source> for <topic or offer>`."
- If the source side is empty or not a URL/path, ask:
  - "What is the source video URL or file path you want to remix?"
- If the target side is empty or vague, ask:
  - "What is the user offer or topic this remix should be written for?"
- If the user says `serious`, `no jokes`, `no analogy`, or `straight rewrite`, record that as `no-analogy` and do not insert a new analogy.

## Pipeline

1. Run the `/watch` preflight and pipeline on the source to produce frames and a timestamped transcript:

```bash
python3 "${CLAUDE_SKILL_DIR}/scripts/setup.py" --check
python3 "${CLAUDE_SKILL_DIR}/scripts/watch.py" "${SOURCE}"
```

2. If `scripts/watch.py` prints a long-video warning or a sparse-scan warning, use `AskUserQuestion`:
  - "The source video is longer than 10 minutes. Should I remix the full video, or a specific section? If you want a section, reply with `--start` and/or `--end` in `MM:SS` or `HH:MM:SS` format."

3. If the source has no transcript available, proceed with a frames-only remix and clearly flag that the result is built from visual evidence alone.

4. Read every frame path the script lists in parallel with `Read` so you have the visual timeline and captions/transcript data together.

## Analysis and skeleton extraction

Internally construct a structural skeleton of the original video using the frames and transcript. Do not show this table to the user yet.

For each core beat, capture:
- `Hook`
- `Setup`
- `Problem`
- `Insight turn`
- `Mechanism`
- `Proof`
- `Offer`
- `CTA`

For each beat, record:
- timestamp range
- beat function
- what the beat is doing mechanistically
- the dominant shot type or visual mode
- whether it uses on-screen text or kinetic captioning
- the emotional / tonal register

Also capture:
- average shot length / cut frequency by segment
- sentence-length curve (short bursts vs longer narrative lines)
- open-loop structure and any parallel promise/answer cadence
- on-screen text patterns, typography emphasis, and any repeated visual hook motif
- tone register (e.g. playful, urgent, direct, somber, expert, conversational)

Treat this skeleton as the contract. The remix must place every beat at roughly the same percentage of total runtime, with the same functional role and pacing shape.

## Analogy generation

By default, generate a fresh analogy for the problem→insight bridge.

### When to skip analogy

- If the user explicitly asked for `serious`, `no jokes`, `no analogy`, or `straight rewrite`.
- If the original tone is somber or the original problem→insight transition is already tightly mechanism-shaped.
- If `--no-analogy` is present in the user's input.

### Analogy process

1. Extract the selling point's core causal mechanism in one sentence. Example: "narrow targeting beats broad targeting because energy concentrates on the actual problem."
2. Brainstorm 5 candidate analogies from unrelated domains such as cooking, plumbing, travel mishaps, sports, animals, weather, kids, tools, gardening, vehicles, body/medical, sleep, music, construction, fishing, pets.
3. Score each candidate 1-3 on:
   - structural fit
   - sensory vividness
   - mild transgression / humor / surprise
   - brevity
4. Pick the candidate with the highest total score. If tied, choose the one with the best structural fit.
5. Write a 2-4 sentence first-person vignette ending with: `And that's exactly how [the selling point] works — [one-line mapping].`
6. Keep the analogy length to roughly 8-15% of total runtime.
7. Replace the original problem→insight bridge with this analogy; do not append it on top.

If all analogy candidates are forced and no strong analogy emerges, ship the remix without an analogy and explain: "The original problem→insight transition is already mechanism-shaped enough that adding an analogy would dilute its clarity."

## Remix output format

Write the new version as a shooting script, not as prose.

For every beat, include:
- `[t=MM:SS-MM:SS]`
- `VO/ON-CAM:` line
- `ON-SCREEN TEXT:` line (use kinetic caption style if the original had one)
- `VISUAL/B-ROLL:` line (mirror the original shot type and visual energy)
- `CUT:` transition description

Keep the hook shape from the original (question / bold claim / contradiction / reveal). Keep the offer beat at the same percentage of runtime as the original. Make the CTA specific to the user's funnel or offer.

### Structural diff note

At the end, include one short paragraph in plain English that says:
- where the remix mirrors the original beat-for-beat structure,
- where it intentionally departs and why.

### User handoff

When returning the result, provide:
1. The full remix shooting script.
2. The analogy beat separately called out so the user can swap it if desired.
3. A 3-bullet rationale for why this remix should retain audience retention as well or better than the original.
4. A second variant with either a different analogy or a tighter / longer cut.

## Failure handling

- If the source is longer than 10 minutes, ask whether to remix the full video or a section, and use `--start` / `--end` if the user chooses a section.
- If the target topic or offer is vague, ask a clarifying question before proceeding.
- If no transcript is available, proceed frames-only and say so explicitly.
- If the source cannot be downloaded or processed, report the platform error and do not guess.

## Implementation notes

- Use `scripts/watch.py` as the source pipeline; do not duplicate frame extraction or transcription logic.
- Keep the original's retention mechanics: same beat percentages, same hook shape, same transition logic, same offer placement.
- Preserve the original arc and pacing while rewriting the content around the user's offer.
- Use the transcript and frame reads together so the new script is grounded in what actually appeared on screen.
