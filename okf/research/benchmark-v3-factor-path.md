---
type: ResearchConstraint
title: Benchmark v3 factor-path freeze
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-14 }
stale_after: 2026-10-14
sources:
  - id: protocol
    resource: docs/benchmark-v3-factor-path.md
    title: Benchmark v3 preregistration and interpretation limits
  - id: spec
    resource: data/benchmark_v3_factor_path/spec.json
    title: Canonical authored factor vocabulary and templates
  - id: manifest
    resource: data/benchmark_v3_factor_path/expected-manifest.json
    title: Frozen split, payload, and leakage hashes
  - id: generator
    resource: src/norishio_lm/benchmark_v3_data.py
    title: Deterministic generator and final-confirmation access gate
  - id: metrics
    resource: src/norishio_lm/benchmark_v3_metrics.py
    title: Common generation and intervention metrics
  - id: data-tests
    resource: tests/test_benchmark_v3_data.py
    title: Split, non-reuse, leakage, and export checks
  - id: metric-tests
    resource: tests/test_benchmark_v3_metrics.py
    title: Metric and intervention contract checks
  - id: tournament-protocol
    resource: docs/factor-path-tournament-v3.md
    title: Preregistered model comparison and execution gate
  - id: tournament-config
    resource: data/benchmark_v3_factor_path/tournament.json
    title: Canonical arms, parameter budget, training, and checkpoint contract
  - id: tournament-model
    resource: src/norishio_lm/benchmark_v3_model.py
    title: Shared model and factor-path variants
---

# Benchmark v3 factor-path freeze

Issue #36 Phase 0 fixes a new 1,152-row structural benchmark before v3 model
training. The split has 384 rows each for train, diagnostic-validation, and
final-confirmation. Each evaluation split has 192 unseen-pair and 192
pair-known/unseen-triple rows, while train covers every atomic factor value.

Source and target surfaces are globally disjoint, and v3 evaluation surfaces
do not copy benchmark-v2 diagnostic/final records. The expected manifest binds
the spec, generator, split, payload, and leakage hashes. Its content digest is
`930958ca1002b9f566fb13d28e072e99fa8ba5c0493308526586dcc128302d84`.
The learning-free freeze commit is
`ae8ea2727558a57f6fa7c0e23ad28fc767dcc859`.

The shared scorer separates free generation, exact surface recovery,
teacher-forced bytes, canonical intermediate heads, source-side swaps, and
true single-factor probability interventions. This authored fixture is not
learned output and does not test glyph, etymology, lexical sense, sememe, or
general Japanese understanding.

Phase 1 preregisters a 2x2 main comparison of linear versus tanh factor-head
transforms and global versus prefix-local factor injection. D_AUX retains the
four auxiliary heads without decoder factor conditioning; NO_INPUT replaces all
row-specific sources with `[BOS, SEP]`. Counts are 31,448 to 32,120 parameters,
a 2.137% spread under the 3% limit. The frozen configuration digest is
`ff89911b64d8f6e652ed617f971ff0512dbda07e4c71bd585f5e0f23635fd399`.
The learning-free architecture freeze commit is
`bc69ac3da75be75b3f39e7d9f44fd2818f0d214d`.
All arms share three seeds, 600 updates, the sampling schedule, and the four
factor losses. Eighteen terminal records and authenticated checkpoints are
required before the one-shot final-confirmation gate. No v3 model has been
trained and no benchmark result has been observed at this phase.
