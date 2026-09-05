---
type: Workflow
title: ai-ready Issue 自動開発
status: draft
generated: { by: ai-assisted-implementation, at: 2026-09-05 }
stale_after: 2026-10-05
sources:
  - id: procedure
    resource: docs/issue-automation.md
    title: 起動と対象 / 状態 / 実装と完了
  - id: selection
    resource: codex/tools/issue_queue.py
    title: select_ready
---

# ai-ready Issue 自動開発

AI 投稿を含む open Issue に ai-ready が付いたら、ローカル Codex が定期確認して1件ずつ実装する。
作業中・レビュー待ちを除外し、失敗時は blocked として再試行を待つ。
独立したサブタスクはユーザー指定の gpt-5.6-luna に積極委譲し、親が要件整理・統合・最終検証を担当する。
同時に処理する Issue は1件のまま。自動化や新規タスクの増殖をサブエージェントへ許可しない。
PR は人がレビューし、自動マージは行わない。
PC・Codex・認証・権限・利用枠が必要。未確認: 実 Issue から PR までの最初の本番実行。
