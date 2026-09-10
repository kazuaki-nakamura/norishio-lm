---
type: ResearchConstraint
title: 意味層の分離と反証実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-10 }
stale_after: 2026-10-10
sources:
  - id: project-rules
    resource: AGENTS.md
    title: Non-negotiable design rules
  - id: architecture
    resource: docs/architecture.md
    title: Separation of evidence channels / Key falsification experiments
---

# 意味層の分離と反証実験

NLM-SEM-001: 字形・字源と現代語義を混同しない。「性 = 忄 + 生」は字形情報であり、現代語義を文字通り合成する根拠ではない。

表記、形態素、字形、字源、現代語義、sememe、概念・関係は別チャネルで保持する。
意味層の有効性は仮説であり、各層を外した ablation と同条件のベースラインで検証する。
v0.x は CPU で基本検証できること。大規模学習や有料 GPU は現段階の前提にしない。

検証状況は [実装範囲](../design/semantic-compiler.md) と [未解決課題](open-questions.md) を参照。
