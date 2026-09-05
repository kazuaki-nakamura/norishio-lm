# Project OKF MCP

プロジェクトのOKF Markdownを検索・段階取得する、読取専用のローカルMCPサーバーです。外部通信と追加Pythonパッケージは不要です。

## 提供機能

- okf_search: 関連度順検索。廃番は既定で除外
- okf_get_concept: 本文、メタデータ、入出リンク取得
- okf_trace: 要件ID、画面、API、テーブル等の横断追跡
- okf_list_open_issues: 未決定、要確認、残課題、リスク一覧
- okf_check_freshness: status、stale_after確認
- okf_list_categories: カテゴリ別件数
- okf:// Resourceの一覧・読取

## 確認コマンド

    python codex/okf_mcp/server.py --self-check
    python -m unittest discover -s codex/okf_mcp/tests -v

サーバーは標準入出力を使います。OKFファイルは更新しません。
