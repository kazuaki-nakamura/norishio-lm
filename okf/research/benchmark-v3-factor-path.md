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
