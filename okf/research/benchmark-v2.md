---
type: ResearchConstraint
title: Benchmark v2 freeze protocol
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-13 }
stale_after: 2026-10-13
sources:
  - id: protocol
    resource: docs/benchmark-v2.md
    title: Frozen split, metrics, and final-holdout rules
  - id: generator
    resource: data/benchmark_v2/benchmark.py
    title: Deterministic generator and manifest validation
  - id: spec
    resource: data/benchmark_v2/spec.json
    title: Factor vocabulary and authored templates
  - id: metrics
    resource: src/norishio_lm/benchmark_v2_metrics.py
    title: Common free-generation and intervention metrics
  - id: tournament
    resource: data/benchmark_v2/tournament.json
    title: Frozen architecture families and run budget
  - id: tournament-doc
    resource: docs/architecture-tournament-v2.md
    title: Tournament preregistration and final gate
  - id: data-tests
    resource: tests/test_benchmark_v2_data.py
    title: Split, leakage, parser, and export checks
  - id: model
    resource: src/norishio_lm/benchmark_v2_model.py
    title: Frozen arm implementations and gradient boundaries
  - id: runner
    resource: src/norishio_lm/benchmark_v2_runner.py
    title: Deterministic training and development evaluation
  - id: execution
    resource: src/norishio_lm/benchmark_v2_execute.py
    title: Eighteen-run checkpoint and terminal-record driver
  - id: final-gate
    resource: src/norishio_lm/benchmark_v2_final.py
    title: Exclusive one-shot final holdout command
---

# Benchmark v2 freeze protocol

Issue #34 Phase 1は、4因子の固定fixture、split、manifest、共通scorerを学習前に凍結した。
`BENCHMARK_FREEZE_SHA`は`19ce7145495a47b40a178255272050caa615dc79`。
このcommit以前・commit内でv2モデル学習は行っていない。

Phase 2の6 arm（5 familyとAのG1 subarm）は
`TOURNAMENT_FREEZE_SHA = 38e363e1b4e836b890f1c852f6d0a45fe3119296`に固定した。
全armは同じbyte source encoder/causal decoderを使い、trainable parameter差を3%未満、
CPU 600更新、seed 7/17/29、18 terminal runに固定した。このcommit内でも学習は行っていない。

全1152行はparticipant/time/event/operatorの独立直積から機械生成する。各評価splitは
participant-time未見群と、pair既知・participant-time-event未見群を192行ずつ持つ。
trainには全原子値が現れる。source allowlistはcontext/textだけで、row、split、template、
gold、完全表現のcategorical IDをモデル特徴にしない。

主指標はstrict parserによる自由生成frameを採点する。中間head、teacher-forced byte、
interventionは別指標で、parse失敗を分母から除かない。final-holdoutは明示開封とmanifest digestを
必要とし、Phase 2の全arm/seed/checkpoint固定完了後に一度だけ評価する。

これはauthored spec由来の構造fixtureであり、学習成果、一般日本語品質、意味層の有効性を示さない。

freeze後の実装は、source-only encoder、target-only causal decoder、6 arm、Bのprefix-only
FSM、seed由来schedule、再構築可能な初期状態とarchitecture/state/fileを束縛するcheckpoint、
厳格terminal record、18-run development driver、排他的な`--evaluate-final` commandを持つ。
正式実行前の全18 parameter preflightはfreeze値と一致した。development localityは同一評価splitから
非対象3因子が一致する4組を固定順で選び、modelにはsource context/textだけを渡す。

この時点の検証は実行系の配線・再現境界である。保存されていないsmoke trainingやteacher-forced
診断を正式tournament結果、自由生成性能、一般LLM性能として扱わない。
