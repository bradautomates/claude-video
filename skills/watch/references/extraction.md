# Extraction

Use for video URLs or local `.mp4`, `.mov`, `.mkv`, `.webm` files.

```bash
python3 "$SKILL_DIR/scripts/watch.py" "<source>" [flags]
```

Pass the source verbatim under normal shell quoting.

## Detail

- `--detail transcript|efficient|balanced|token-burner`
- `transcript`: captions/Whisper; no ordinary frames
- `efficient`: keyframes; cap 50
- `balanced`: scene-aware; cap 100
- `token-burner`: scene-aware; uncapped; warning past 250
- universal rate cap: 2 fps
- `--max-frames N` overrides mode cap

Full-video targets: ≤30s 12–30; ≤1m 40; ≤3m 60; ≤10m 80; longer sparse to cap. Prefer a focused range for long videos.

## Focus and cue frames

`--start T`, `--end T`: `SS`, `MM:SS`, or `HH:MM:SS`. Focused targets: ≤5s 2fps; 5–15s up to 30; 15–30s up to 60; 30–60s up to 80; 60–180s up to 100. Timestamps remain absolute.

`--timestamps T1,T2,...` pins transcript-cue frames. Use only for real visual cues such as “look here.” Cue frames are additive, reserved before cap allocation, bounded by focus range. `--detail transcript --timestamps ...` produces cue-only frames.

For a second URL pass, reuse the downloaded local file; do not redownload.

## Other flags

- `--resolution W`: default 512; use 1024 only for necessary text detail
- `--fps F`: clamped to 2
- `--out-dir DIR`: retain caller-supplied directory
- `--whisper groq|openai`: force configured backend
- `--no-whisper`: captions or frames only
- `--no-dedup`: preserve subtle sampled changes

Native captions are preferred. Whisper fallback extracts mono 16kHz audio, uses Groq `whisper-large-v3` first when configured, otherwise OpenAI `whisper-1`. External transcription requires applicable approval. Never read `transcript.txt` in the coordinator; trusted reducers own it.
