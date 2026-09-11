---
type: Implementation
title: Concept bottleneck toy実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-12 }
stale_after: 2026-10-12
sources:
  - id: pair-head-results
    resource: docs/pair-head-results.md
    title: Joint pair head fails unseen pairs across three fixed seeds
  - id: pair-head-plan
    resource: docs/pair-head.md
    title: Fixed joint marginal route and pair CE protocol
  - id: pair-head-experiment
    resource: src/norishio_lm/toy_pair_head.py
    title: Historical replay and three-seed joint head control
  - id: pair-slot-model
    resource: src/norishio_lm/pair_slot_model.py
    title: Joint distribution marginals with unchanged local gate
  - id: head-following-results
    resource: docs/head-following-results.md
    title: Three-seed ordinary failure and partial oracle following
  - id: head-following-plan
    resource: docs/head-following.md
    title: Fixed calibration, train confidence and three-seed protocol
  - id: head-joint-metrics
    resource: src/norishio_lm/head_joint_metrics.py
    title: Marginal calibration and same-row joint prediction metrics
  - id: head-following-diagnostics
    resource: src/norishio_lm/head_following_diagnostics.py
    title: Selected train subsets and frozen semantic input ablation
  - id: head-following-experiment
    resource: src/norishio_lm/toy_head_following.py
    title: Historical D replay and fixed seeds17 and29
  - id: local-slot-results
    resource: docs/local-slot-results.md
    title: Ordinary joint failure and partial gold-head recovery
  - id: local-slot-plan
    resource: docs/local-slot-injection.md
    title: Preregistered duplicate removal and causal prefix gates
  - id: local-slot-model
    resource: src/norishio_lm/local_slot_model.py
    title: Effective 31-dimensional conditioning and consumed-prefix windows
  - id: local-slot-generation
    resource: src/norishio_lm/local_slot_generation.py
    title: Self-history greedy decoding without gold span access
  - id: local-slot-checkpoint
    resource: src/norishio_lm/local_slot_checkpoint.py
    title: C and D mode-preserving checkpoint metadata
  - id: local-slot-experiment
    resource: src/norishio_lm/toy_local_slots.py
    title: Historical replay and fixed C D comparison
  - id: joint-slot-plan
    resource: docs/joint-slots.md
    title: Algebraically equivalent split projection and fixed factorial protocol
  - id: joint-slot-results
    resource: docs/joint-slot-results.md
    title: Unseen pair support and zero both-slot validation accuracy
  - id: joint-slot-experiment
    resource: src/norishio_lm/toy_joint_slots.py
    title: Historical replay and grouped head intervention diagnosis
  - id: factorized-slot-model
    resource: src/norishio_lm/factorized_slot_model.py
    title: Exact copied additive projection blocks before tanh
  - id: pair-metrics
    resource: src/norishio_lm/pair_metrics.py
    title: Train pair membership and conjunction scoring
  - id: explicit-slot-plan
    resource: docs/explicit-slot-head.md
    title: Preregistered five-class heads and 650 added parameters
  - id: explicit-slot-model
    resource: src/norishio_lm/explicit_slot_model.py
    title: Source-only slot predictions appended to concept probabilities
  - id: explicit-slot-checkpoint
    resource: src/norishio_lm/explicit_slot_checkpoint.py
    title: CPU weights-only state and vocabulary integrity
  - id: explicit-slot-experiment
    resource: src/norishio_lm/toy_explicit_slots.py
    title: Historical replay and separated head interventions
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

Issue #20は人物/time専用5class headを追加し、既存33確率へ10確率を連結。
追加650parameter、head CE各1、slot byte CEなし。A/B過去評価とC保存再読込を厳密再現。
Cは人物36/time25、全frame0/150。head-only goldで59/31へ変わるが全frame0。
単独oracleで人物86/time0、人物5/time72となり、同時保持が未解決。
容量/目的/元concept重複が混ざるため、専用化単独の有効性や一般意味性能は主張しない。

Issue #22はtrain15組/validation5組で全validation150例がunseen pairと確認。
seen0群の率はnull。A歴史Cを再現し、同重みを3線形投影へ分割したBを同schedule学習。
分割前後は数学的に同じ関数クラス、43073parameterで差分0。演算順序/最適化差が残る。
B通常人物40/time25、両slot0/150、人物head goldで72/time1。干渉と同時保持失敗は継続。
8介入・25counterfactual pairの生成/同slotとcross-slot感度、checkpoint完全再現を記録。
seen/unseen性能差と偏りの因果効果はこの分割では推定不可。一般意味性能の証明としない。

Issue #24は旧人物/time群を切るCと、同重みで既生成prefixによる位置窓を使うDを比較。
C/D各42689parameter、A/Bから384減、C/D共通初期重みと同scheduleを固定。
A/Bの歴史評価とC/D保存物再読込を厳密再現。C人物15/time37、D人物57/time43だが通常両slotは0/150。
両head goldではC両slot6/全frame3、D48/16へ部分回復。Dの未見5組中3組のfactorialは0のまま。
補足集計でsource head同時argmax正解C1/D0。head予測とdecoder追随の両方に失敗が残る。
Dの人物介入→先行time logitは構造上0、逆方向はGRU経由で残る。全体parseによるslot採点差を逆因果としない。
文法prior、parameter削減、h0/時刻変更を交絡として保持し、一般意味性能の証明としない。

Issue #26はD固定でseed7を厳密再現、17/29を600更新。同時head正解0/2/0、通常生成両slotは全seed0。
両head goldの両slot48/46/64、平均52.667/150。head誤り相関は負だが因果・soft情報欠如の証明ではない。
confident train correct263例で両slot210、wrong群0で評価null。閾値は両head0.8固定、validation選択なし。
全base入力zeroでは文頭崩壊でgate開始0、parse0。event zero + goldは両slot72へ増えるが全frame12へ低下。
入力group診断と意味層全体の有効性を区別し、n3/共通拡張seed20/文法priorの限界を保持する。

Issue #28はA保存Dの3seedを厳密再現し、825parameter追加の25-way pair head Bを比較。
独立head CEを保ち、joint marginalのみdecoder局所注入。train pair正解446/450・450/450・438/450だが
validation全未見pairは全seed0、gold確率平均0.000495385。B通常両slotも全seed0。
gold両slot61/36/47で部分追随するが平均48はA52.667より低い。factorized目的は未実施。
容量/目的/入力分布の交絡、未見pairへのCE正例不在を保持する。全379テストと保存物再現を確認。
Issue26の追加teacher_forced byte診断は到達不能コードでnullだったため未測定として訂正。修復は残課題。
