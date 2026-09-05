---
type: Implementation
title: SemanticCompiler の実装範囲
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-05 }
stale_after: 2026-10-05
sources:
  - id: compiler
    resource: src/norishio_lm/semantic_compiler.py
    title: SemanticCompiler.compile / from_json
  - id: schema
    resource: src/norishio_lm/schema.py
    title: SemanticRecord / LexicalSense
  - id: lexicon
    resource: data/demo_lexicon.json
    title: Four hand-authored demonstration entries
  - id: tests
    resource: tests/test_semantic_compiler.py
    title: Three compiler regression tests
---

# SemanticCompiler の実装範囲

SemanticCompiler は入力文字列を手書き JSON 辞書と完全一致で照合し、明示的な dataclass に変換する。
表記・形態素・字形・字源注記・複数語義と付属 sememe / concept・関係を保持する。
未知語は表記、トークン、文字にフォールバックし、語義を自動生成しない。

「性」「性器」「心生」「ハナレナイ」のデモ出力は辞書の転記であり、学習済みモデルの成果ではない。
文脈による語義選択、テンソル化、学習器、生成器は未実装。
3つの既存テストはフィールド分離、解剖学概念、未知語を確認するが、意味理解の性能は実証しない。

[分離原則](../research/semantic-separation.md) と [未解決課題](../research/open-questions.md) を参照。
