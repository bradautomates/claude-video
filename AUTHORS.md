# Authors & credits

## Original author

**Bradley Bonanno** — [bradautomates/claude-video](https://github.com/bradautomates/claude-video), [@bradbonanno](https://www.youtube.com/@bradbonanno)

Created `/watch`: the yt-dlp → ffmpeg → captions/Whisper → Claude `Read` pipeline, the `--detail`
modes, perceptual frame dedup, Whisper auto-chunking, `--timestamps`, the SessionStart hook, the
self-contained Agent Skills package and the multi-host install flow. Released under the MIT license,
© 2026 Bradley Bonanno — the [LICENSE](LICENSE) file is his, unchanged.

## This fork — frinsen/claude-video

**frinsen** — [github.com/frinsen](https://github.com/frinsen)

Maintains the fork: triaged the 50 open upstream pull requests and 41 issues, merged or adapted the
fixes below, resolved the issues that had no PR, fixed the test harness, and verified the 0.3.0
release end-to-end. See [CHANGELOG.md](CHANGELOG.md) for the per-item detail.

## Community contributors (upstream pull requests landed in 0.3.0)

Listed by GitHub handle with the upstream PR number. "Merged" means the PR was merged as authored
(conflicts resolved); "adapted" means its idea or parts of its code were re-implemented on top of
other changes, with the author credited in the commit message.

| Contributor | Upstream PRs | Landed as |
|---|---|---|
| [@bagalutenGregor](https://github.com/bagalutenGregor) | #219 ffmpeg `-fps_mode` probe | merged |
| [@IsaiahCalvo](https://github.com/IsaiahCalvo) | #192 Windows UTF-8 stdout | merged |
| [@Jordan-Zhu](https://github.com/Jordan-Zhu) | #206 Windows permissions, #208 test path separators, #210 scene-coverage fallback, #205 time-spaced selection + `WATCH_MAX_FPS` | merged / cherry-picked |
| [@charles98601-sg](https://github.com/charles98601-sg) | #231 test isolation, #232 single `.env` parser, #233 concurrent Whisper chunks, #234 native-language captions | merged / adapted |
| [@OpenClawLinda](https://github.com/OpenClawLinda) | #226 `--fps` spread, #227 yt-dlp staleness, #228 403 diagnosis + transcript-only degrade, #225 rolling captions | merged / adapted |
| [@eltokhy](https://github.com/eltokhy) | #182 rolling-caption dedupe | merged |
| [@oheewono](https://github.com/oheewono) | #223 Whisper hallucination flag, #221 prefer human captions | merged / adapted |
| [@nbkwabi](https://github.com/nbkwabi) | #214 API-key isolation between providers | merged |
| [@jayvee6](https://github.com/jayvee6) | #199 self-hosted OpenAI-compatible Whisper backend | merged |
| [@dsp407](https://github.com/dsp407) | #224 frame candidate counts | merged |
| [@gth-spec](https://github.com/gth-spec) | #236 proper nouns flagged as unverified | merged |
| [@jvdurian-pixel](https://github.com/jvdurian-pixel) | #218 quoted `.env` values with comments | adapted |
| [@AlexLenovo](https://github.com/AlexLenovo) | #212 video's own caption language | adapted |
| [@Rchardd](https://github.com/Rchardd) | #187 `--lang` | adapted |
| [@D0mD0mD0m](https://github.com/D0mD0mD0m) | #207 `--force-whisper` | adapted |
| [@endiaye677](https://github.com/endiaye677) | #184 opt-in cookies | adapted |
| [@drsandeeprana00-bit](https://github.com/drsandeeprana00-bit) | #200 YouTube player-client retry | adapted |
| [@Rasmus257](https://github.com/Rasmus257) | #220 audio-only crash fix | adapted |
| [@mrrobotbuilder](https://github.com/mrrobotbuilder) | #204 probe `OSError` guard | adapted |
| [@sainbayare-net](https://github.com/sainbayare-net) | #97 keyframe-less range falls back to uniform | merged (frames part) |
| [@JMAL1988](https://github.com/JMAL1988) | #147 sidecar `.vtt` for local files | merged |
| [@Daily-AC](https://github.com/Daily-AC) | #175 RGB dedup thumbnails, #176 image-token numbers | merged |
| [@thomaswillner](https://github.com/thomaswillner) | #154 skill description says when to invoke | merged |
| [@stickersfxlab](https://github.com/stickersfxlab) | #119 `.env` in the encodings Windows writes | adapted |
| [@gqbeerman](https://github.com/gqbeerman) | #136 preflight reads `./.env` like whisper.py | adapted |
| [@caleb436](https://github.com/caleb436), [@pornthepp](https://github.com/pornthepp) | #127 #179 `android` client fallback | adapted |

The ffmpeg `-vsync` fix was also independently submitted by @PollxTroy-create (#102), @varunsahni18 (#103), @vgrosetti-maker (#125), @apalm8 (#133), @syrusdigital (#138), @utkarshbindal-wq (#148), @dustymurph (#162), @vakogogu-coder (#166), @ELpistolero21 (#171), @ZiCoreDom (#172), @SVSOnderwijs (#177), @MaCeeeee (#181), @victoropp (#130), @tiff4183 (#132), @sauveteur71 (#168), @thetimlee1 (#139), @rainervianaprocurador (#230),
@blkzera (#213), @SatishGs01 (#211), @tradersc2020-oss (#202), @dd58mk72wv-stack (#198),
@crybbyforreal (#197), @amipcoaching2027 (#188), @endiaye677 (#183), @aromat24 (#194) and
@oheewono (#216); the Windows permission fix by @frankkeil (#201), @ppradeep123-ops (#190),
@vanlieropf-dot (#185) and @Rchardd (#191); the Windows console fix by @tradersc2020-oss (#203)
and @oheewono (#217). Further independent fixes for problems landed above: @fabio-pisoni-hw (#100), @ZiCoreDom (#173), @stickersfxlab (#118), @dungartoriaaa (#146), @khcho98-maker (#159) for Windows; @weekly100million (#104) for subprocess decoding; @redonto-007 (#110) for console encoding; @Maktorin (#105), @drlee91 (#129), @waxandwires (#151), @Ydiouri (#124) for caption parsing; @greekr4 (#106) for test isolation; @tomimoyano15-byte (#113), @nacho-marin (#114), @ai-websites-poland (#116), @Nicopatron (#123), @djhammer20k (#164) for caption language; @Diterex (#112), @androsland (#169) for local transcription; @xiaoqian289-foece (#140) for cookies. Thank you all — the duplicates are how we knew which bugs mattered most. The full disposition is in [UPSTREAM.md](UPSTREAM.md).

## Related project

[taoufik123-collab/claude-watch](https://github.com/taoufik123-collab/claude-watch) by Taoufik is a
separate derivative of the original (from v0.1.3) that adds editorial analysis and Obsidian export.
No code from it is included here.
