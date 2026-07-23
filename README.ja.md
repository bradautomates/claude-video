[English](README.md) | [简体中文](README.zh-CN.md) | **日本語**

# /watch

**Claude があらゆる動画を視聴できるようにします。**

Claude Code（推奨 — マーケットプレイス経由で自動更新）：
```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

Codex、Cursor、Copilot、Gemini CLI、またはその他 50 以上の [Agent Skills](https://agentskills.io) ホスト：
```bash
npx skills add bradautomates/claude-video -g
```
（`-g` はユーザー向けにグローバルインストールし、すべてのプロジェクトで利用できるようにします。外すとプロジェクト単位になります。）

その他のインストール方法（claude.ai ウェブ版、手動）は、下記の[インストール](#インストール)セクションを参照してください。

設定なしで開始できます。`yt-dlp` と `ffmpeg` は初回実行時に macOS では `brew` を通じてインストールされます（Linux/Windows では正確なコマンドを表示します）。ほとんどの公開動画は字幕を使って無料で処理できます。Whisper API キーが必要なのは、動画に字幕がない場合だけです。

---

Claude はウェブページを読み、スクリプトを実行し、リポジトリを閲覧できます。しかし、標準の状態では*動画を見る*ことができません。YouTube のリンクを貼り付けても、タイトルから推測するか、画面上の情報の 90% が欠けた文字起こしを取得するしかありません。

Claude Video `/watch` では、URL またはローカルパスを貼り付けて質問できます。Claude は最初に字幕を取得し、必要なものだけをダウンロードし、フレームを抽出し（シーン認識、または `efficient` 詳細モードの高速キーフレーム）、タイムスタンプ付きの文字起こしを取得し（利用できる場合は無料の字幕、フォールバックとして Whisper API）、各フレームを画像として `Read` します。回答する時点では、動画を*見て*、音声を*聞いた*状態です。

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
```

## 実際の利用例

**他の人のコンテンツを分析する。** `/watch https://youtu.be/<viral-video> what hook did they open with?` Claude は最初のフレームを確認し、冒頭の文字起こしを読み、構成を分析します。広告クリエイティブ、競合のローンチ、ポッドキャストのイントロなど、*何を*伝えるかと同じくらい*どのように*伝えるかが重要なものにも使えます。

**動画からバグを診断する。** 誰かから壊れた動作の画面録画が送られてきたとします。`/watch bug-repro.mov what's going wrong?` Claude は録画を見て、問題が現れるフレームを特定し、画面上の内容を説明します。多くの場合、自分でファイルを開かなくても原因を見つけられます。

**動画を要約する。** `/watch https://youtu.be/<long-thing> summarize this` は当然の処理を行います。構成、重要な瞬間、実際に話された内容と表示された内容を抽出します。2 倍速で見るよりも高速です。

**更新動画から誇張表現を取り除く。** `/watch https://youtu.be/<launch-video> what's actually new — skip the hype` は、「革新的」とうたわれた機能リリースを本当に重要な数点まで絞り込みます。10 分間の導入や過剰な宣伝なしに要点を把握できます。

**再生リストをノートに変える。** `/watch https://youtu.be/<video> summarize this to a note` をシリーズ全体に実行し、動画ごとの要約を保存します。チャンネルやコースが、何時間もかけて視聴するものではなく、検索可能なノート集になります。

## 仕組み

1. **動画と質問を貼り付けます。** URL（yt-dlp が対応するものすべて — YouTube、Loom、TikTok、X、Instagram、およびその他数百のサイト）またはローカルパス（`.mp4`、`.mov`、`.mkv`、`.webm`）を指定できます。
2. **`yt-dlp` が最初に字幕を確認します。** `transcript` 詳細モードでは、字幕付き URL は動画をダウンロードせずに返されます。それ以外の場合、または Whisper が音声を必要とする場合は、その実行に必要なものだけをダウンロードします。
3. **`ffmpeg` が選択された詳細モードでフレームを抽出します。** `efficient` はキーフレームだけをデコードするため、ほぼ瞬時です。`balanced`/`token-burner` はシーン変化フレームを優先し、数が不足する場合は動画の長さを考慮した均等サンプラーにフォールバックします。JPEG はデフォルトで幅 512px、Claude Read との互換性のため高さは 1998px 以下に制限されます。
4. **文字起こしは二つの場所のいずれかから取得します。** 最初の方法では、`yt-dlp` がソースからネイティブ字幕（手動または自動生成）を取得します。無料で即時、精度はおおむね良好です。フォールバックでは、モノラル 16 kHz、64 kbps の mp3 音声クリップ（約 480 kB/分）を抽出し、Whisper に送信します。Groq の `whisper-large-v3`（推奨 — より安価で高速）または OpenAI の `whisper-1` を利用します。
5. **フレームと文字起こしが Claude に渡されます。** スクリプトは `t=MM:SS` マーカー付きのフレームパスと、タイムスタンプ付きの文字起こしを出力します。Claude は各フレームを並列に `Read` します。JPEG はコンテキスト内で画像として直接表示されます。
6. **Claude は画面と音声の実際の内容に基づいて回答します。** 「説明によれば」でも「タイトルによれば」でもありません。フレームを見て、文字起こしを聞き、動画を見た人と同じように回答します。
7. **クリーンアップ。** スクリプトは最後に作業ディレクトリを表示します。追加の質問をしない場合、Claude が削除します。

## フレーム予算 — 重要である理由

トークンコストの大部分はフレームによって決まります。各フレームは画像であり、画像トークンはすぐに積み上がります。スクリプトの自動 fps ロジックは、30 分の動画をまばらに走査するためにコンテキスト予算を浪費しないようにします。そのような質問には、対象の 30 秒間に絞る方が適しています。

| 長さ | デフォルトのフレーム予算 | 得られるもの |
|----------|---------------------|--------------|
| ≤30 秒 | 約 30 フレーム | 高密度 — ほぼすべての重要な瞬間 |
| 30 秒 - 1 分 | 約 40 フレーム | 引き続き高密度 |
| 1 - 3 分 | 約 60 フレーム | 十分な密度 |
| 3 - 10 分 | 約 80 フレーム | まばらですが実用的 |
| > 10 分 | 100 フレーム（上限付きモード） | 「まばらな走査」の警告 — 範囲を絞って再実行するか、`--detail token-burner` で上限なしの完全なカバレッジを使用 |

ユーザーが特定の瞬間（「2:30 あたり」「最後の 30 秒」「0:45 から 1:00」）を指定した場合は、`--start` / `--end` を渡します。フォーカスモードでは 1 秒あたりの予算が高密度になり、上限は 2 fps です。動画全体をまばらに走査するよりはるかに有用です。

## フレームの重複排除

フレーム選択 — キーフレーム（`efficient`）、シーン変化検出（`balanced`/`token-burner`）、または不足時にフォールバックする均等サンプラー — でも、ほぼ同一のフレームが現れることがあります。画面録画で一枚のスライドが 90 秒間表示され続けると、十数枚が生成され、それぞれ別の画像として課金されます。重複排除処理は、フレームが Claude に届く前にそれらを削除します。すべてのフレームモードでデフォルトで動作します（`--no-dedup` で無効化できます）。

1. 一度の `ffmpeg` 呼び出しで、抽出された各 JPEG を 16×16 のグレースケールサムネイルに縮小します。それ以降はすべて Python 標準ライブラリで処理し、画像ライブラリは不要です。
2. 各フレームについて、*直前に保持したフレーム*との**平均絶対差**（ピクセルごとの明るさの変化の平均、0～255）を計算します。
3. 差がしきい値（`2.0`）以下なら、ほぼ重複したフレームとして削除します。それ以外は保持し、新しい参照フレームとします。
4. フレーム予算の上限は重複排除の*後*に適用されるため、予算は異なるフレームに使われます。

直前のフレームではなく、直前に*保持した*フレームと比較することで、フレーム間の変化が一度もしきい値に達しない緩やかなフェードも検出できます。しきい値は意図的に低く、構造ではなく絶対的な明るさを測るため、コードの一行の差分、ターミナルの一行分のスクロール、色の異なる単色スライドはいずれも残ります。

**Frames** 行は、まとめられた内容を `6 selected from 14 candidates (… 8 near-duplicates dropped …)` のように報告します。常に動き続ける映像では何も削除されず、従来と同じコストになります。

## 詳細モード — 実測値

`--detail` ダイヤルは、速度とトークンコストを視覚的な忠実度と引き換えに調整します。以下の数値は、実際の **49:08** の YouTube 動画（1280×720、英語自動字幕）で測定したものです。長く、ほぼ静止した画面録画で、上限に最も厳しいケースです。抽出時間は事前にダウンロードしたコピーをローカル CPU で処理した時間です。一度だけのダウンロードは約 **37 秒** / 76 MB で、三つのフレームモードが共有しました。

| モード | エンジン | フレーム数 | 上限 | 抽出時間 | 時間的カバレッジ | 推定画像トークン |
|------|--------|--------|-----|-----------------|-------------------|-------------------|
| `transcript` | なし（字幕） | 0 | — | **約 4.5 秒**（yt-dlp を一度呼び出し、ダウンロードなし） | 全体（テキスト） | 0（約 26.6k テキストトークン） |
| `efficient` | キーフレーム（`-skip_frame nokey`） | 50 | 50 | **約 0.5 秒** | 0:00 → 49:04（全体） | **約 9.8k** |
| `balanced` | シーン変化 | 100 | 100 | **約 20.9 秒** | 0:00 → 48:38（全体） | **約 19.7k** |
| `token-burner` | シーン変化 | 116 | 上限なし | **約 21.0 秒** | 0:00 → 48:38（全体） | **約 22.8k** |

- **画像トークン**は Anthropic の `(width × height) / 750` を使用します。デフォルト幅 512px では、これらの 720p フレームは 512×288 で、**約 197 トークン/フレーム**です。`--resolution 1024` はおよそ 4 倍になります。字幕付きのすべてのモードで文字起こしも提示され、長い動画では文字起こしの方がコストの大部分を占めることもあります。
- **すべてのフレームモードで一つのサンプリング規則を使用します。** 各モードは全範囲からすべての候補を検出し、その後で上限まで均等にサンプリングします（最初と最後は必ず保持）。モード間の違いは候補の*ソース*（キーフレームまたはシーンカット）と上限だけであり、カバレッジの分散方法は同じです。そのため、最後のフレームは途中ではなく必ず終端に配置されます。
- **`efficient` は速度重視の階層**（約 0.5 秒）です。キーフレームだけを再構築するため、カットを探すために全フレームをデコードするシーンモードより約 40 倍高速です。また、動きの少ない映像では `balanced` より*多く*のフレームを返す場合があります（キーフレームがシーンカットより多いため）。「efficient」は抽出が速いという意味であり、フレームが少ないという意味ではありません。
- **`token-burner` が `balanced` と異なるのは上限を超えた場合だけです。** このクリップには 116 個のカットがあったため、`balanced` は 100 個をサンプリングし、`token-burner` は 116 個すべてを保持しました。数百のカットがある動きの激しい動画では、`token-burner` はすべてを保持し（250 フレーム超のトークン警告を発生させます）、`balanced` は 100 フレームまで間引きます。

未取得の URL からエンドツーエンドで実行する場合、`transcript` が最も低コストです。フレームモードには抽出時間に加え、共通の約 37 秒のダウンロード時間がかかります。

## インストール

| 環境 | インストール |
|---------|---------|
| **Claude Code** | `/plugin marketplace add bradautomates/claude-video`、続いて `/plugin install watch@claude-video` |
| **Codex、Cursor、Copilot、Gemini CLI、その他 50 以上** | `npx skills add bradautomates/claude-video -g` |
| **claude.ai**（ウェブ） | [`watch.skill` をダウンロード](https://github.com/bradautomates/claude-video/releases/latest) → Settings → Capabilities → Skills → `+` |
| **手動 / 開発** | `git clone` して、`skills/watch` をホストの skills ディレクトリへシンボリックリンク（下記参照） |

### Claude Code

```
/plugin marketplace add bradautomates/claude-video
/plugin install watch@claude-video
```

後で `/plugin update watch@claude-video` を使って更新できます。

### Codex、Cursor、Copilot、Gemini CLI、およびその他 50 以上のホスト

[Agent Skills](https://agentskills.io) CLI は、検出した任意の agent に skill をインストールします。

```bash
npx skills add bradautomates/claude-video -g
```

`-g` はユーザー向けにグローバルインストールします（`~/.codex/skills`、`~/.cursor/skills` など）。外すと現在のプロジェクトにインストールされます。便利なフラグ：

- `-a, --agent <names…>` — 特定のホストを対象にします。例：`-a codex -a cursor`
- `-l, --list` — インストールせず、このリポジトリの skill を一覧表示します
- `--copy` — シンボリックリンクではなくファイルをコピーします（シンボリックリンクに対応しないファイルシステム向け）

CLI は `skills/watch/SKILL.md` から skill を検出し、フォルダー全体、つまり `SKILL.md` とその `scripts/` ランタイムを自己完結した単位としてコピーします。`SKILL.md` はインストール先からの相対位置で自身のスクリプトを解決するため、すべてのホストで同じように動作します。

後で `npx skills update watch -g` を使って更新できます。

### claude.ai（ウェブ）

1. 最新リリースから [`watch.skill` をダウンロード](https://github.com/bradautomates/claude-video/releases/latest)します。
2. Settings → Capabilities → Skills に移動します。
3. `+` をクリックしてファイルをドロップします。

最初に Capabilities で「Code execution and file creation」を有効にしてください。この skill は `ffmpeg` と `yt-dlp` を呼び出すため、有効にしないと動作しません。

### 手動（開発者）

リポジトリをクローンし、自己完結した skill フォルダーをホストの skills ディレクトリへシンボリックリンクします。シンボリックリンクにより、作業ツリーを編集するとインストール内容も同期されます。

```bash
git clone https://github.com/bradautomates/claude-video.git
ln -s "$(pwd)/claude-video/skills/watch" ~/.claude/skills/watch   # or ~/.codex/skills/watch
```

claude.ai 向けには、ソースから `.skill` バンドルを構築します。`bash skills/watch/scripts/build-skill.sh` を実行すると `dist/watch.skill` が生成されます。

## 初回実行

最初の `/watch` 呼び出しで、skill は `scripts/setup.py --check` を実行します。`ffmpeg` / `yt-dlp` が PATH にない場合、または Whisper API キーが設定されていない場合は、修正手順を案内します。

- **macOS** — `brew install ffmpeg yt-dlp` を自動実行します。
- **Linux** — 正確な `apt` / `dnf` / `pipx` コマンドを表示します。
- **Windows** — 正確な `winget` / `pip` コマンドを表示します。
- **API キー** — `~/.config/watch/.env` にスキャフォールドを作成し（モード `0600`）、`GROQ_API_KEY`（推奨）と `OPENAI_API_KEY` のコメント付きプレースホルダーを追加します。

セットアップ後、プリフライトは無言になり、`/watch` はそのまま動作します。この確認は 100ms 未満の検索なので、その後の実行を遅くしません。

## 自分のキーを使用する

ほとんどの公開動画は字幕で無料処理できます。Whisper フォールバックが動作するのは、動画に本当に字幕トラックがない場合だけです。一般的にはローカルファイル、TikTok、一部の Vimeo、まれに字幕のない YouTube 動画が該当します。

| 機能 | 必要なもの | コスト |
|------------|---------------|------|
| ダウンロード + ネイティブ字幕 | `yt-dlp` + `ffmpeg` | 無料 |
| Whisper フォールバック（推奨） | [Groq API キー](https://console.groq.com/keys) — `whisper-large-v3` | 安価、高速 |
| Whisper フォールバック（代替） | [OpenAI API キー](https://platform.openai.com/api-keys) — `whisper-1` | 標準料金 |
| Whisper を完全に無効化 | `--no-whisper` | 無料、字幕がない場合はフレームのみ |

## 使い方

```
/watch https://youtu.be/dQw4w9WgXcQ what happens at the 30 second mark?
/watch https://www.tiktok.com/@user/video/123 summarize this
/watch ~/Movies/screen-recording.mp4 when does the UI break?
/watch https://vimeo.com/123 what tools does she mention?
```

特定の区間にフォーカスすると、フレーム予算が高密度になり、トークンコストが下がります。
```
/watch https://youtu.be/abc --start 2:15 --end 2:45
/watch video.mp4 --start 50 --end 60
/watch "$URL" --start 1:12:00            # from 1h12m to end
```

その他の調整項目（`scripts/watch.py` に渡されます）：

- `--detail transcript|efficient|balanced|token-burner` — 忠実度と速度のダイヤルです。`transcript` はフレームを省略し、文字起こしだけを取得します。`efficient` は高速キーフレームを使用します（上限 50）。`balanced` はシーンを認識します（上限 100）。`token-burner` はシーンを認識し、上限がありません。
- `--timestamps T1,T2,…` — 各絶対タイムスタンプ（`SS`/`MM:SS`/`HH:MM:SS`）でフレームを取得します。Claude は最初に文字起こしを読み、次に発表者が示した瞬間（「ここを見て」「ご覧のとおり」）を狙います。詳細モードのフレームに追加されますが、上限の枠は予約されます。フォーカスモードではウィンドウ外の手掛かりが除外されます。`--detail transcript` と組み合わせると、これらが唯一のフレームになります。
- `--max-frames N` — フレーム上限を下げ、トークン予算を抑えます。
- `--resolution W` — Claude が画面上の文字（スライド、ターミナル、コード）を読む必要がある場合、フレーム幅を 1024px に上げます。
- `--fps F` — 自動 fps 計算を上書きします（引き続き 2 fps が上限です）。
- `--whisper groq|openai` — 特定の Whisper バックエンドを強制します。
- `--no-whisper` — 文字起こしを完全に無効化します。字幕がない場合はフレームのみです。
- `--no-dedup` — ほぼ重複したフレームを保持します。デフォルトでは、フレーム差分処理が直前のフレームと視覚的にほぼ同一のフレーム（保持されたスライド、静的な画面録画、一時停止した動画）を削除し、フレーム予算を異なる内容に使います。このフラグはその処理を無効にします。
- `--out-dir DIR` — 作業ファイルを指定した場所に保持します（デフォルトは自動生成された一時ディレクトリ）。

## 制限

- **長い動画の精度は詳細モードに依存します。** 上限付きモード（`efficient`、デフォルトの `balanced`）では、約 10 分を超えるとカバレッジが薄くなります。フレーム上限が動画全体に分散されるため、「まばらな走査」警告が表示され、`--start`/`--end` で範囲を絞って再実行する方が適しています。`token-burner` は上限を解除し、動画全体の*すべての*シーン変化フレームを保持するため、画像トークンを多く消費する代わりに、長い動画でも完全性を保ちます。10 分は上限付きモードの目安であり、厳密な制限ではありません。
- **詳細設定は一つのダイヤルです。** デフォルトはバランス重視で、シーン認識フレーム、最大 2 fps、100 フレーム上限です。高速な 50 フレームのキーフレーム走査には `--detail efficient`、上限なしのシーン候補には `--detail token-burner` を使用します。`WATCH_DETAIL` を `~/.config/watch/.env` で設定するとデフォルトを変更できます。

## 構成

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

## 開発

```bash
# Run the test suite (stdlib + pytest; ffmpeg required for frame tests):
python3 -m pytest -q

# Build the claude.ai upload bundle:
bash skills/watch/scripts/build-skill.sh      # → dist/watch.skill
```

リリースするには `vX.Y.Z` のタグを付けてプッシュします。ワークフローが `dist/watch.skill` をビルドして GitHub Release に添付します。`skills/watch/SKILL.md`、`.claude-plugin/plugin.json`、`.codex-plugin/plugin.json` のバージョンを同期してください。

バージョン履歴は [CHANGELOG.md](CHANGELOG.md) を参照してください。

## オープンソース

MIT ライセンスです。

`yt-dlp`、`ffmpeg`、Claude のマルチモーダル `Read` ツールを基盤としています。Whisper 文字起こしは [Groq](https://groq.com) または [OpenAI](https://openai.com) を利用します。

Brad Bonanno によって構築されました。[YouTube（@bradbonanno）](https://www.youtube.com/@bradbonanno)では AI を使った構築について発信し、[Solaris Automation](https://www.solarisautomation.io/) では企業向け AI オペレーティングシステムを構築しています。`/watch` のおかげで動画を何度もスクラブせずに済んだら、ぜひチャンネルで声をかけてください。

## Star の履歴

<a href="https://www.star-history.com/?repos=bradautomates%2Fclaude-video&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
   <img alt="Star 履歴チャート" src="https://api.star-history.com/chart?repos=bradautomates/claude-video&type=date&legend=top-left" />
 </picture>
</a>

---

[github.com/bradautomates/claude-video](https://github.com/bradautomates/claude-video) · [@bradbonanno](https://www.youtube.com/@bradbonanno) · [Solaris Automation](https://www.solarisautomation.io/) · [ライセンス](LICENSE)
