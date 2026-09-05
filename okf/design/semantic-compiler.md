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
    title: Six hand-authored demonstration entries, demo-v1
  - id: tests
    resource: tests/test_semantic_compiler.py
    title: Original three compiler regression tests
  - id: schema-tests
    resource: tests/test_schema.py
    title: Validation, roundtrip, provenance, ambiguity and ablation regressions
  - id: schema-contract
    resource: docs/semantic-schema.md
    title: Schema 1.0 and migration contract
---

# SemanticCompiler の実装範囲

SemanticCompiler は入力文字列を手書き JSON 辞書と完全一致で照合し、明示的な dataclass に変換する。
表記・形態素・字形・字源注記・複数語義と付属 sememe / concept・関係を保持する。
未知語は表記、トークン、文字にフォールバックし、語義を自動生成しない。

「性」「性器」「心生」「ハナレナイ」と否定・願望比較2例のデモ出力は手書き辞書の転記であり、学習済みモデルの成果ではない。
schema 1.0 の型・キー・重複語義ID・relation検証、レコードJSON往復、各層・候補別の由来・source・revisionを実装。
旧辞書は互換読み込みし、由来の欠測を unknown にする。候補は未選択で保持する。
context / span は入力位置付きメタデータであり選択器ではない。補助層は元のレコードを保持して除外できる。
「離れない」の否定と「離れたくない」の願望の否定を区別し、心生は experimental_poetic として保持する。
文脈による語義選択、テンソル化、学習器、生成器は未実装。
既存3テストの意図を維持し、入力検証・JSON往復・出典・意味の独立性・層の除外等を追加検証した。意味理解の性能は実証しない。

[分離原則](../research/semantic-separation.md) と [未解決課題](../research/open-questions.md) を参照。
