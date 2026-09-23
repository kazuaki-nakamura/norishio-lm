---
type: ResearchConstraint
title: Benchmark v3 factor-path freeze
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-24 }
stale_after: 2026-10-24
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
  - id: h0-experiment-descriptor
    resource: data/benchmark_v3_h0_confirmation/experiment-descriptor-v1.json
    title: Issue 44 frozen h0-bypass descriptor
  - id: h0-confirmation-summary
    resource: docs/results/benchmark-v3-h0-confirmation/summary.json
    title: Issue 44 machine-readable aggregate result
  - id: h0-confirmation-report
    resource: docs/results/benchmark-v3-h0-confirmation/README.md
    title: Issue 44 decision record and interpretation limits
  - id: h0-head-factor-audit
    resource: docs/results/benchmark-v3-h0-confirmation/head-factor-audit.json
    title: Saved-raw-only factor head and intervention audit
  - id: head-learning-freeze
    resource: data/benchmark_v3_head_learning/experiment-descriptor-v1.json
    title: Issue 46 pre-training fixture, model, schedule, raw, and decision freeze
  - id: head-learning-result
    resource: docs/results/benchmark-v3-head-learning/README.md
    title: Issue 46 source-to-factor head learnability result and limits
  - id: head-learning-summary
    resource: docs/results/benchmark-v3-head-learning/summary.json
    title: Issue 46 raw-bound factor and class-balanced metrics
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

Issue #44 executes a separate future-only confirmation without reopening any
historical v3 final artifact. A new authored fixture contains 384 train and 96
confirmation rows, split evenly between unseen-pair and pair-known/unseen-triple
support. Its content digest is
`296f32138f9ee448ca8fe975200047f53a6679343af35cbe7423d1e3435e1945`.
The experiment descriptor SHA-256 is
`19c11e5fa7b5ae90dd2bc3b6ddca20366e6d2b9798b98e3ccf1c7c8bcdf10533`.
It freezes H1L1/H1L1_ANCHOR, seeds 7/17/29, six runs, 3,600 updates, one CPU
thread, the run/evaluation order, BOS-start probes, expected and observed gate
traces, donor bindings, denominators, raw fields, and decision thresholds.

All six runs completed. H1L1 versus H1L1_ANCHOR three-seed means were 0.03125
versus 0.06597 for ordinary generation-frame exact and 0.07639 versus 0.08333
for intermediate head-frame exact. The anchor preserves the two metrics under
the frozen drop rule, but neither head is close to the 0.80 strong-head floor.
Alternate-one-hot scheduled nontrivial joint success was 12/24 versus 7/24, so
the anchor fails the preregistered improvement rule. The recorded conclusion is
`decoder_path_inconclusive_due_to_weak_heads`; the h0-bypass hypothesis is not
supported. ANCHOR unseen-pair frame exact was 0.00694 versus 0.125 for
pair-known/unseen-triple, meeting the preregistered fixture-specific
compositional-failure condition. These are authored structural-factor results,
not learned modern semantics, sememes, concepts, or general language quality.

The first summary aggregation failed after all six raw runs had been saved. The
retained raw records prove 3,600 updates and 115.189 seconds of summed training
wall time, but they do not retain the original executor wall time covering
evaluation, interventions, and the failed aggregation. Total-wall budget
compliance is therefore unavailable rather than reconstructed as false; no
training or inference was repeated to fill the missing evidence.

A saved-raw-only post-hoc audit decomposes the weak four-factor joint head result.
Across all 288 confirmation rows per arm, H1L1 atomic rates are 0.69444 event,
0.69792 operator, 0.51736 participant, and 0.28819 time. H1L1_ANCHOR rates are
0.69444, 0.69792, 0.49653, and 0.30556. Weakness is concentrated in participant
and especially time rather than being uniform across all heads. The audit also
retains seed and support-group denominators and factor-specific scheduled-six
alternate-one-hot/donor-soft counts. It is diagnostic only and does not change
the frozen decision, thresholds, or ranking.

Issue #46 froze a new disjoint authored structural-factor fixture before a
six-run JOINT versus FACTOR_ONLY comparison. Both arms use the identical H1L1
module tree, paired initial states, 32,120 trainable parameters, and the same
seeded batches. The freeze descriptor SHA-256 is
`210fdbab6bcd4f8604d8653835ea5f075df3a41a2c79fb0fbc95bc9a07967322`.
All six runs completed exactly 3,600 updates. FACTOR_ONLY training
resubstitution participant/time correct rates were 501/1152 and 488/1152;
all four factor heads were below the preregistered 0.80 class-balanced floor.
Confirmation participant/time were 174/288 and 176/288, with paired deltas
that did not improve both factors in all three seeds. The frozen branch is
`basic_optimization_or_capacity_unresolved`. The comparison does not identify
source encoding, optimization, or capacity as the cause, and it does not alter
Issue #44's decoder-path conclusion. No modern lexical-sense, sememe, concept,
or general language-quality claim follows from this authored fixture.
