---
type: Implementation
title: Concept bottleneck toy実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-11 }
stale_after: 2026-10-11
sources:
  - id: calibrated-metrics
    resource: src/norishio_lm/concept_metrics.py
    title: Class-balanced metrics and separate missing and unseen masks
  - id: checkpoint
    resource: src/norishio_lm/toy_checkpoint.py
    title: CPU weights-only checkpoints with vocabulary integrity validation
  - id: evaluation
    resource: src/norishio_lm/toy_evaluation.py
    title: Fixed 60-step matched controls and validation replay
  - id: diagnosis
    resource: src/norishio_lm/toy_diagnosis.py
    title: Fixed-budget byte and EOS failure diagnosis
  - id: diagnosis-tests
    resource: tests/test_toy_diagnosis.py
    title: Illegal transitions versus incomplete tails and gold EOS positions
  - id: controls
    resource: src/norishio_lm/toy_controls.py
    title: Matched initialization and validation-only content controls
  - id: generation
    resource: src/norishio_lm/toy_generation.py
    title: Reference-free strict C greedy generation
  - id: control-tests
    resource: tests/test_toy_controls.py
    title: Fixed gradients, train-only means and majority, aligned metrics
  - id: generation-tests
    resource: tests/test_toy_generation.py
    title: Incremental causality, EOS, raw invalid UTF-8 retention
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

handoffの外部AIレビュー記録ではvalidation概念accuracyはtrain多数派基準とほぼ同じ。
予測conceptをtrain固定平均に置換しても生成損失はほぼ不変との報告。ローカル再検証と
区別し、内容依存の有効性は未実証とする。

続くローカルtoy_controls追試は同初期state/学習順の定数条件再学習、対応反転、
自由生成を実装。test未使用、validationだけで比較。定数再学習の損失も通常Cと近く、
自由生成4条件は全て150件中EOS終了/有効UTF-8/完全一致0。低いteacher-forced損失を
意味内容の利用や流暢な生成の証明としない。詳細値と出典hashはhandoff参照。

toy_diagnosisの60/600更新比較では600で両条件の有効UTF-8/EOS終了が150/150に回復。
一方全件同じ文で完全一致0のまま。60stepの学習不足がbyte/EOS失敗に寄与する説明を
支持するが、入力依存の意味生成は未実証。モデル・教材・元の既定値は変えずtest未使用。

Issue #7は多数派/クラス別/項目balanced平均/全7項目一致を同一ハーネスで比較する。
欠損と未見のmask・分母を分離し、operators順序を保持。seed17の対応置換について
異なる完全既知フレームのペア数を保存する。soft/hard/zero/train平均/置換と、
同初期値・同学習順の定数再学習を区別する。重みと語彙IDをローカル保存し、
validation150件のlogitsと自由生成を再読込前後で照合する。test評価や予算拡張はしない。
