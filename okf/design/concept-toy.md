---
type: Implementation
title: Concept bottleneck toy実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-10 }
stale_after: 2026-10-10
sources:
  - id: model
    resource: src/norishio_lm/concept_model.py
    title: Named concept heads and strict C decoder
  - id: experiment
    resource: src/norishio_lm/toy_experiment.py
    title: CPU training and measurement harness
  - id: adapter
    resource: src/norishio_lm/toy_adapter.py
    title: Source allowlist and isolated false glyph fixture
  - id: tests
    resource: tests/test_toy_experiment.py
    title: Source swaps, causal histories, input boundaries
  - id: losses
    resource: tests/test_concept_model.py
    title: Masked losses and auxiliary isolation
  - id: corpus
    resource: data/issue3/README.md
    title: Authored data and held-out composition split
  - id: contract
    resource: docs/concept-toy.md
    title: Observation, comparison and limits
  - id: observations
    resource: docs/handoff.md
    title: Issue 3 measured results
---

# Concept bottleneck toy実験

元のcontext/textだけからsource featuresを作る。教師ラベルは補助損失専用。
7概念項目はevent/operators/agent/participant/time/location/repeat_marked。
operatorsは順序付きカテゴリでNOT(WANT)とWANT(NOT)を区別する。
null注釈はmask、UNSPECIFIEDは観測された教材値。語彙fitはtrainのみ。

Aはsource-prefix GRU、Bはencoder latent併用、Cは予測concept確率と過去targetのみ。
C decoderにはsource bytes、encoder latent、sense/sememe出力を渡さない。
LM/sense/sememe/conceptの重みを独立にゼロ化できる。全aux欠損でもLM逆伝播可能。
train/validation/test、各損失、対象数、概念accuracy、チャネル除去と介入を記録する。

単一seed・短いCPU toy学習の観測。テンプレートを共有する未見組合せであり、自然対話、
普遍的意味、自由生成品質を証明しない。soft確率にはargmax以上の情報が残り得る。
入力の語義/sememe/概念/形態素/字形/字源/関係は空。空チャネル除去は有効性比較ではない。
OKF、外部辞書、私的ログは教材に使わない。人による検証の記録ではない。
