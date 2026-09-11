---
type: Implementation
title: Concept bottleneck toy実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-11 }
stale_after: 2026-10-11
sources:
  - id: slot-objective-plan
    resource: docs/slot-objective.md
    title: Fixed matched A/B objective and baseline preservation
  - id: slot-objective
    resource: src/norishio_lm/slot_objective.py
    title: Byte-weighted slot CE and train-only matched control
  - id: slot-objective-experiment
    resource: src/norishio_lm/toy_slot_objective.py
    title: Historical baseline replay and oracle-separated validation
  - id: prefix-plan
    resource: docs/prefix-intervention.md
    title: Preregistered oracle prefix boundaries and denominators
  - id: prefix-experiment
    resource: src/norishio_lm/toy_prefix_experiment.py
    title: Frozen replay and validation-only prefix intervention
  - id: prefix-generation
    resource: src/norishio_lm/prefix_generation.py
    title: Prefix-only warmup then self-running generation
  - id: prefix-metrics
    resource: src/norishio_lm/prefix_metrics.py
    title: Post-switch slot eligibility and byte coverage
  - id: slot-retention
    resource: src/norishio_lm/toy_slot_retention.py
    title: Frozen single-slot oracle and reference-history diagnosis
  - id: slot-spans
    resource: src/norishio_lm/toy_spans.py
    title: Authored template UTF-8 byte positions
  - id: span-metrics
    resource: src/norishio_lm/span_metrics.py
    title: Byte-weighted reference-history probability rank and sensitivity
  - id: slot-plan
    resource: docs/slot-retention.md
    title: Precommitted checkpoint replay and oracle boundaries
  - id: step-conditioning
    resource: src/norishio_lm/toy_step_experiment.py
    title: Matched initial-only versus per-step additive comparison
  - id: mechanical-slots
    resource: src/norishio_lm/toy_slots.py
    title: Target-template-only frame scoring with coverage and failure denominators
  - id: step-plan
    resource: docs/step-conditioning.md
    title: Precommitted zero-added-parameter A/B experiment plan
  - id: collapse
    resource: src/norishio_lm/toy_collapse.py
    title: Frozen source probe and explicit gold oracle boundary diagnosis
  - id: probe
    resource: src/norishio_lm/toy_probe.py
    title: Train-only standardized linear seven-head probe
  - id: collapse-statistics
    resource: src/norishio_lm/collapse_metrics.py
    title: Confusion entropy probability variance and pairwise distances
  - id: collapse-plan
    resource: docs/collapse-diagnosis.md
    title: Pre-fixed seed and CPU budgets with diagnostic limits
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

Issue #9のseed7/600更新診断では凍結encoderの線形probeが860/1050、既存head771/1050、
多数派570/1050。encoder/soft conceptは150通り、argmax frame34通りだが生成は1通り。
50通りのgold oracleを渡しても同一文。encoderの情報保持とdecoder出力の多様性は別であり、
headだけの改善で解決するとは言えない。oracleの分布差と単一seedによる限界を保持する。

Issue #12は既存投影ベクトルを各token embeddingへ加えるper-step additiveを追加。
既定のinitial-onlyを保持し、追加parameter0、同初期値・同学習順でA/Bを比較する。
教材target文型への完全一致だけでslotを機械採点し、coverage/全対象分母/条件付き分母を分離。
同じ自己生成履歴で概念だけを介入した位置別logit L1/KL/argmax差を記録する。
seed7/600のper-step通常生成は9種類（旧方式1）、対応置換でLMが悪化し後半の感度も残る。
ただし全slot一致0/150、人物・時点保持は改善せず、文型parse109/150。
条件への依存が増えることと意味の正確さを分け、既定方式は変更しない。

Issue #14は保存済みper-step重みを凍結再利用し、正解履歴byte診断と自己履歴生成を分離。
人物/timeのheadは41/150と43/150。人物のみoracleで自由生成人物20→15、時点のみでは22→22。
正解履歴の高いbyte正解率はslot保持と同一視しない。全条件frame一致0/150。
元結果のJSON表現一致とstate不変を検証。任意prefix介入未実施のため履歴伝播は因果分離せず、
one-hotの分布差も限界として保持する。再学習・test評価・人の検証印は追加しない。

Issue #16は凍結per-stepに6境界×4conceptのprefix介入を比較。元report・通常生成の
厳密再現と重み不変を確認。与え済みslotを加点せず、自己履歴のbyte分母を分離する。
人物直前prefixの人物0/150、時点直前の時点30/150。人物を既に与えた後の文末成功は
人物保持の証拠ではない。履歴修復だけの回復は未確認で、decoder/目的の原因は未分離。
任意counterfactual prefix未実施、長さ/内容効果・one-hot分布差・単一seedの限界が残る。

Issue #18は既存4損失を維持したper-step Aと、人物/time byte CE重み1を加えたBの同初期値/
同schedule対照。構造増分0、Aの過去結果と両checkpoint再読込を厳密再現。
Bはslot byte損失が低下し人物20→26/時点22→25だが、全frame0、全体LM/EOSは悪化。
headも変化するためdecoder単独効果ではなく、単一seed・oracle分布差の限界を保持。
任意の専用slot headは未実装。意味層の有効性や一般日本語性能の証明としない。
