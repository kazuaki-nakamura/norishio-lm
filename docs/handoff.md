# Norishio-LM 引き継ぎ記録

検証日: 2026-09-10

## 最新の実装状態: Issue #2

### PR #5 出典レビュー対応（2026-09-10）

層全体に出典を指定し候補には指定しない入力で、senses/sememes/conceptsの出典が
tensor化後に失われる問題を再現した。候補出典をunknownで正規化するとgetのfallbackが
使われないことが原因。`ChannelBatch.layer_provenance` に全行の層出典を保持し、
候補固有の `Feature.provenance` と分けた。候補unknownを既知出典へ昇格しない。
特徴が空の行や `.to(device)` 後も層出典を保持する。既存の語彙checkpoint形式は変更なし。

Luna (`gpt-5.6-luna`) が独立回帰テストを作成。修正前の初版検査は15 failed / 1 passed。
親が実装修正とテストレビューを担当し、省略出典の正規化、異なるsource/revision、
3チャネル、空行、層除外、device移動、元入力の変更、出力不変性を補強した。
修正後の回帰検査は20 passed。統合全スイートは **136 passed, 2 subtests passed**、skipなし。
実行は下記と同じ専用venvで `python -m pytest -q -p no:cacheprovider`（Windows Temp用許可付き）。
両デモ終了コード0。encoderのNumPy未導入警告は継続するが、NumPy変換は使用していない。
この節より下の116件・24原本は初回実装時の履歴。

`codex/multichannel-encoder`、基点 `dfddf315168e81e11dde52007e5267ba5de6413e`。
以下の Issue #1 / 初期引き継ぎ節は履歴。この節と [encoder契約](multichannel-encoder.md) を現在の実装範囲として優先する。

実装済み: 10チャネルの独立した固定語彙・tensorizer、特徴ごとの原本パスと出典・候補対応、
PAD/UNKの分離、入力文字列と予約IDの衝突防止、語彙checkpointの往復。
独立Embedding/mean pooling/projectionと入力依存のscalar gateで融合するCPU encoderを追加した。
各チャネルの無効化、欠損時のゼロ寄与、全無効時の有限ゼロ出力、勾配分離を検証する。
字形・字源から語義ラベルを生成しない。語義IDを外した比較にsememe/concept値経由で同IDを混入しない。

隔離worktreeの専用venvへCPU版PyTorchをインストールし、editable import先を確認。
Python 3.11.15 / PyTorch 2.14.0+cpu / pytest 9.1.1。

```powershell
.\.venv\Scripts\python.exe -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,model]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

全テスト **116 passed, 2 subtests passed**（モデル検査のskipなし）。既存のWindows Temp制限のため
全テストは許可付きで実行。CPU forward/backwardのネットワーク禁止注入検査も成功。
既存辞書デモ6例が正常終了。encoderデモはseed=7、6入力でfused `[6,32]`、gate `[6,10]`、
有限値・字形/字源無効化・全無効時ゼロを確認。ランダム初期値の配線検査であり学習成果ではない。
PyTorchの直接import/デモ時にNumPy未導入警告が出るが終了コード0。NumPy変換は使用していない。
知識索引24原本、full=healthy、OKF files=7 / concepts=5 / errors=0 / warnings=0。
MCP実プロセスの検索・原本追跡検査も成功。GitHub Actionsでの検証は未実行。

Luna (`gpt-5.6-luna`) にtensorizer/同テストとencoder回帰テスト・独立レビューを分担。
親が予約文字列衝突などをレビューして修正を統合し、encoder・デモ・文書・最終全テストを担当。
AIレビューのみで、人の研究内容検証印は追加していない。

未実装: 候補選択、順序/グラフを反映するencoder、概念ボトルネック、decoder、学習loss、比較実験。
語彙はtraining splitでfitし、モデルと対応する語彙checkpointを必ず保持する。
外部辞書・性能改善・大規模学習・GPU起動は含まない。次は本PRのレビュー・マージ後にIssue #3の
toy baseline / bypass / concept bottleneck比較へ進む。

## Issue #1 の実装履歴

辞書ベース SemanticCompiler に schema 1.0 を導入した。キー・型・同一表現内の重複語義ID・relation三つ組・JSON重複キーを検証し、位置と原因を持つ SchemaError を返す。
旧辞書の読み込み互換性、レコードJSON保存・復元、層と語義別の由来/source/revision、未選択候補、context/span、元レコードを保持する層除外を実装。
空・空白・未知語も surface を損失なく保持する。否定と願望の区別2例を追加し、元の4例は保持。心生は実験的・詩的な用例として明示。
仕様・移行経路は [スキーマ設計](semantic-schema.md) を参照。以下の初期引き継ぎ節の未実装一覧は履歴であり、現在の状態はこの節と設計書を優先する。

作業ブランチ: `codex/issue-1`。基点: `af25c414341fab91b456080543032bd377880e64`。
隔離 worktree と専用 `.venv` を使用し、import 先が worktree 内であることを確認。

実行コマンド（worktree ルート、Windows）:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

Python 3.11.15 / pytest 9.1.1。環境構築成功。初回は JSON null を空辞書として受け入れる回帰テストが失敗（1 failed / 80 passed）。入力境界を修正し、再実行は **81 passed, 2 subtests passed**、終了コード0。
一時ファイルを使うテストは既知の Windows sandbox 制限を避けて許可付きで実行。ネットワーク禁止を注入したコンパイラ往復・ablation テストも成功。
デモ6例（既存4例と否定・願望比較2例）が正常終了。知識索引17原本、full=healthy、OKF形式検査 errors=0 / warnings=0。MCP実プロセス検証も成功。

未実装: 文脈語義選択、外部辞書連携、出典の内容確認、関係端点のオントロジー検証、テンソル化、学習・生成、比較実験。
学習成果・意味層の有効性は主張しない。大規模学習、有料GPU、外部辞書転載は行っていない。
次の候補: 本PRのレビュー後に adapter / encoder 境界を設計し、CPUベースラインと層別比較へ進む。

## 受け入れた内容と履歴

- ユーザー確認により、現在の作業フォルダを ZIP の展開済み内容として受け入れた。再展開や別コピーによる置換はしていない。
- 既存 Git 履歴を保持。引き継ぎ開始時の `master` は `e89ae540e6050e8d5c89d0a52ce5a323a830e90e`。
- AGENTS.md、README.md、docs/architecture.md、全 Python 実装、テスト、デモ辞書、pyproject.toml を確認した。
- 開始時から存在した demo.py の標準出力 UTF-8 対応を保持し、その変更を含む作業ツリーで検証した。

## 実装済み

- dataclass による `SemanticRecord` と `LexicalSense`。
- 表記、トークン、形態素、文字、字形構造、字源注記、現代語義、語義に付属する sememe / concept、関係の明示的な保持。
- JSON 辞書を読み、入力文字列との完全一致でレコードを組み立てる `SemanticCompiler`。
- 未知語では表記・トークン・文字にフォールバックし、語義を推測して追加しない。
- 「性」「性器」「心生」「ハナレナイ」の手書き辞書と表示デモ。
- 字形と語義の別フィールド保持、多義性、解剖学概念、未知語の基本テスト。

これは SemanticCompiler の初期雛形であり、LLM 本体や学習済みモデルではない。
デモ結果は辞書に記載された情報の出力であり、モデルが学習・推論した成果ではない。
「心生」は実験用の詩的複合語として記載されており、一般語としての意味の実証ではない。

## 未実装と検証上の限界

- 実用的な形態素解析、外部辞書アダプター、文脈による語義選択。
- 入力辞書の厳密なスキーマ検証、関係の三つ組の長さ検証、出典・信頼度の追跡。
- 各層のテンソル化・エンコーダー・融合・概念ボトルネック。
- causal LM、生成器、学習ループ、複合損失、学習済み重み。
- 各意味層を外す ablation harness と、比較実験による有効性評価。
- 日本語の曖昧語、誤誘導する字形、未知複合語、言い換えを網羅するベンチマーク。
- 現在の3テストは限定的な構造確認であり、意味理解の品質や層の有効性を証明しない。
- デモ辞書の参照はソースツリー配置に依存する。通常の wheel 配布での動作は未検証。

## 環境と実行結果

既存 `.venv` を再利用。Python 3.11.15、pytest 9.1.1。
README の仮想環境・editable install 方針に従い、Windows では activate の代わりに仮想環境の Python を直接指定した。

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pip install -e '.[dev]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- 最初の pip install: 終了コード1。一時ディレクトリの build tracker へのアクセス権限エラー。
- 権限を付けて再実行した pip install: 終了コード0。norishio-lm 0.1.0 の editable install 成功。
- デモ: 終了コード0。4例すべての SemanticRecord を表示し、「性」の字形構造と複数の現代語義を別フィールドに保持していることを確認。
- pytest 通常実行: 終了コード0、3 passed, 1 warning。既存 pytest キャッシュへの書き込みが WinError 5 で拒否された警告。
- キャッシュ無効実行: 終了コード0、3 passed、警告なし。テストは省略していない。
- 大規模学習、有料 GPU 起動は実施していない。テストからのネットワークアクセスはない。

## GitHub の状態

- 接続先: https://github.com/kazuaki-nakamura/norishio-lm
- 認証済み個人アカウント: `kazuaki-nakamura`（表示名 `nkmrkzak`）。
- 同名リポジトリは既に Public として存在する。非公開リポジトリの新規作成要件は同名競合のため未達。
- 今回の引き継ぎではリポジトリの上書き、公開範囲変更、push、強制 push を行っていない。
- 引き継ぎ結果はローカルの `master` に記録する。記録コミットは `git log -1 --format=%H` で確認できる。
- 非公開の共同作業拠点を作る場合は、別名の採用または既存リポジトリの扱いについてユーザーの方針が必要。

## 次の実装候補

1. 辞書入力のスキーマ検証と出典情報を追加し、不正な語義・関係・欠損値のテストを整備する。
2. 曖昧語と意図的に誤誘導する字形の回帰テストを増やし、字形情報から現代語義を創作しないことを確認する。
3. 層ごとの有効・無効設定と辞書アダプター境界を設計し、出典を保持してテンソル化する。
4. CPU で動く小さなベースラインから、同一条件で各層を外した比較実験を実装する。

字形・字源は現代語義の根拠と同一視しない。各意味層の採用は比較実験の結果で判断する。

## AI 作業基盤の追加（2026-09-05）

ユーザー指定の ai-project-foundation から、開発用 OKF、読取専用 MCP、索引・整合性検査を取り込んだ。
研究モデル内部の概念表現とは分離している。元コミット、導入範囲、実行コマンド、権限エラーを含む検証結果は [AI 作業基盤](ai-foundation.md) を参照。
導入後はコンパイラと基盤を合わせて19テスト、2 subtests が成功し、デモ4例と MCP 実プロセス検証も成功した。
GitHub への push と MCP クライアント登録は未実施。
