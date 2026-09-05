# Norishio-LM AI 作業基盤

## 由来と取り込み範囲

元: https://github.com/kazuaki-nakamura/ai-project-foundation

取り込み元コミット: `e9bffbfb97441b565e62d2618e4bc26f797f09a6`（2026-09-05 取得）。
ユーザー指定の基盤から選択的に取り込んだ。上流の公開ライセンスは未設定。
自動追従ではなく、更新時にこのコミットとの差分を確認して適用する。

- `codex/okf_mcp/`: 読み取り専用の字句検索・概念取得・根拠追跡・鮮度確認とテスト。
- `codex/tools/`: 索引生成、整合性検査、OKF 形式検査、更新イベント、完了スナップショット公開の部品とテスト。
- 上流の架空 documents / OKF は持ち込まず、既存の原本を参照するプロジェクト固有 OKF を作成。
- トークン収集、費用比較、改善指標の収集スクリプトは今回の初期導入対象外。計測・課金・常駐監視は起動しない。

独自変更: source_files によるルート文書の明示列挙、Python キャッシュ除外、これらの回帰テスト、Norishio-LM の知識で動く MCP live check。
検索基盤は研究用 LM の依存パッケージには追加していない。

## 境界

原本（docs / src / data / tests と明示したルート文書）、OKF（根拠付き要約）、生成索引（キャッシュ）を分ける。
開発用 OKF とモデル内部の sememe・概念層は別物。OKF を自動で学習入力や評価の正解にしない。
索引には本文を複製せず、相対パス・サイズ・更新時刻・SHA-256 を記録する。
`.venv`、`.git`、秘密情報、会話、学習成果物を source_roots に追加しない。

## Windows での検証

リポジトリルートで実行する。既存の Python 3.11+ 環境を使い、基盤自体は標準ライブラリで動作する。

```powershell
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/okf_mcp/server.py --self-check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
.\.venv\Scripts\python.exe -m norishio_lm.demo
```

通常の作業開始時は `check_knowledge.py --mode quick`。索引不一致なら原本と OKF の影響を確認してから再生成する。
quick はパス・サイズ・時刻、full は SHA-256 まで照合する。意味の正しさや外部リンクの内容は保証しない。
OKF 更新時は sources、generated.at、status、stale_after、okf/log.md を更新し、形式検査と full を実行する。
`codex/work_output/` は Git 対象外。新規 checkout では最初に索引を生成する。

## MCP の起動

`codex/okf_mcp/config.toml.example` はプロジェクトルートを起動ディレクトリとした汎用設定例。
別ディレクトリから起動するクライアントでは Python と server.py のパスを checkout に合わせて設定する。
サーバーは stdio で読み取り専用。実際の Codex クライアントへの登録はこの導入には含めない。
プロセスとしての動作とクライアントで利用可能な状態を区別する。

利用順: `okf_search` → `okf_get_concept` → 必要に応じて `okf_trace` → sources の原本確認。
日本語検索は短い語に分ける。ベクトル検索や形態素解析ではない。

## 更新時

上流の変更を比較し、プロジェクト固有の AGENTS、foundation.json、OKF を保持して必要な差分だけ適用する。
公開部品の latest_complete_pointer 自体は完了確認を行わない。将来データ取り込みに使う際は必須ファイル検査後にだけ公開する。
並列 writer や障害復旧は別途設計が必要。今回は取得ジョブや公開ジョブに接続していない。

## 導入時の検証結果（2026-09-05）

- 索引生成: 13原本。Python キャッシュと editable install の egg-info を除外。full 検査: healthy、errors 0。
- OKF: 3概念（index / log を含め5ファイル）、errors 0、warnings 0。
- MCP self-check と実プロセスの JSON-RPC 検証: 成功。6ツールを列挙し、SemanticCompiler 検索、NLM-SEM-001 追跡、概念取得・リソース取得を確認。
- pytest: 19 passed, 2 subtests passed。既存コンパイラ3件と基盤16件を実行。
- 初回 sandbox 内の pytest は一時フォルダ権限で14 failed / 6 errors。権限を付けて同じコマンドを再実行し成功。テスト失敗を省略して成功扱いにはしていない。
- 既存デモ4例は正常終了。学習・GPU 起動はなし。
- MCP クライアントへの登録・接続、GitHub Actions、GitHub への push は未実施。

運用上の改善候補: この Windows sandbox では pip と tempfile 利用テストで一時ディレクトリの権限制限が再発した。
同じコマンドを無条件に再試行せず、テストの保存先と権限を確認してから実行する。
