---
type: ResearchConstraint
title: Benchmark v3 factor-path freeze
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-19 }
stale_after: 2026-10-19
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
  - id: tournament-fsm
    resource: src/norishio_lm/benchmark_v3_fsm.py
    title: Causal prefix-only target grammar gates
  - id: tournament-runner
    resource: src/norishio_lm/benchmark_v3_runner.py
    title: Frozen training and diagnostic evaluation paths
  - id: post-freeze-audit
    resource: src/norishio_lm/benchmark_v3_audit.py
    title: Runtime source-target and prior-benchmark isolation audit
  - id: tournament-checkpoint
    resource: src/norishio_lm/benchmark_v3_checkpoint.py
    title: Authenticated exclusive checkpoints
  - id: tournament-final
    resource: src/norishio_lm/benchmark_v3_final.py
    title: All-complete one-shot final-confirmation gate
  - id: development-results
    resource: docs/results/benchmark-v3-development.md
    title: Frozen diagnostic-validation tournament observations
  - id: development-summary
    resource: docs/results/benchmark-v3-development-summary.json
    title: Machine-readable seed and factorial summaries
  - id: final-results
    resource: docs/results/benchmark-v3-final.md
    title: Attested one-shot final-confirmation observations and limits
  - id: final-summary
    resource: docs/results/benchmark-v3-final-summary.json
    title: Machine-readable final selection, seed, support, and intervention summary
  - id: protocol-errata
    resource: docs/results/benchmark-v3-protocol-errata.json
    title: Read-only primary-metric and intervention-target audit
  - id: future-contract
    resource: src/norishio_lm/benchmark_v3_contract.py
    title: Future machine-readable selection and intervention contracts
  - id: future-protocol-freeze
    resource: data/benchmark_v3_factor_path/future-protocol-v1.json
    title: Canonical future-only execution descriptor and digest
  - id: future-protocol-code
    resource: src/norishio_lm/benchmark_v3_future_protocol.py
    title: Future run binding and pre-execution validation gates
  - id: future-intervention-code
    resource: src/norishio_lm/benchmark_v3_future_intervention.py
    title: Future alternate-value intervention helper
  - id: errata-audit-code
    resource: src/norishio_lm/benchmark_v3_errata.py
    title: Saved-JSON-only protocol errata audit
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
`bc69ac3c132c77d9613481ae163b1e8c4342a9eb`.
All arms share three seeds, 600 updates, the sampling schedule, and the four
factor losses. Eighteen terminal records and authenticated checkpoints are
required before the one-shot final-confirmation gate. No v3 model has been
trained and no benchmark result has been observed at this phase.

The post-freeze implementation compiles both target templates into a causal
prefix FSM and provides the fixed runner, both intervention diagnostics,
checkpoint and terminal authentication, seed-level factorial aggregation, and
an exclusive final marker. Final readiness requires all 18 checkpoints to be
complete; recorded failures remain visible but do not authorize final access.
The integrated suite passed 564 tests with one skip. This verifies wiring and
protocol behavior, not trained performance or semantic-layer effectiveness.

The frozen 18-run diagnostic tournament completed with zero failures. On free
generation exact, the three-seed development order is H1L1, H0L1, H1L0, H0L0,
D_AUX, NO_INPUT. The 2x2 descriptive effects are +0.010851 for activation,
+0.035156 for locality, and +0.009549 for their interaction. Unseen-pair
performance is substantially below pair-known/unseen-triple performance for
every source-conditioned arm. The original Issue #36 record counted no target
changes in 12 main-arm probes; Issue #39 supersedes that aggregation with 0/48
fixed class-0 probes. Neither count supplies positive evidence of factor-level
causal control.

After explicit user authorization, the one-shot final-confirmation evaluation
completed from the frozen 18 checkpoints: 384 rows, 18 evaluations, and zero
failed runs. The attested result SHA-256 is
`eb94b2b35d9de0c575b90342d517582e2304c766b3d8bb385ea22f2ac8c56a59`.
The frozen primary ranking is H1L1, H0L1, H1L0, H0L0, D_AUX, NO_INPUT. A
second call was rejected before evaluation by the exclusive marker. The final
intervention result also supplies no positive evidence of factor-level causal
control. Agreement with the development order is descriptive evidence on the
authored fixture, not general semantic or language-model performance.

Issue #39 records a post-merge protocol erratum. The selection implementation
used free-generation exact, while the preregistration named generation-frame
exact. A read-only audit of the saved 18 development JSON files and the attested
final result reconstructed the preregistered field without checkpoint loading,
training, inference, or final access. Both corrected rankings remain H1L1,
H0L1, H1L0, H0L0, D_AUX, NO_INPUT. Development generation-frame effects are
activation +0.011719, locality +0.036024, interaction +0.013021; final effects
are +0.028212, +0.036024, +0.014757. Equality of rankings does not erase the
metric mismatch.

The historical intervention was a fixed class-0 probability intervention. The
scored saved artifacts confirm 0 target changes among 48 main-arm probes in
each split, but omit baseline probability vectors, so baseline-argmax class-0
classification is unavailable. This is not evidence about an alternate-value
intervention. A separately versioned future protocol freezes the primary,
tie-break paths, derived source paths and reducer, aggregation, direction, and
usage phases. Its canonical SHA-256 is
`bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c`.
Future wrappers validate this digest before trainer, evaluator, final marker
publication, and final-row access, and bind it into run metadata and returned
artifacts. A future-only helper selects `(baseline_argmax + 1) % width`, retaining
the class indices and requiring the actual intervened one-hot to select that
exact class while preserving source identity, decoder prefix, and non-target
factor probabilities. Historical Issue #36 execution code and all
existing final artifacts, marker, checkpoints, and attestations remain unchanged.
