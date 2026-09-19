# Benchmark v3 factor-path tournament freeze

This document is the freeze-time snapshot for Issue #36. Current diagnostic
observations are recorded in
[`docs/results/benchmark-v3-development.md`](results/benchmark-v3-development.md);
the pre-training wording below describes the state at the freeze boundary.

Issue #36 Phase 1 fixes the model comparison and execution contract before any
benchmark-v3 training or result observation. The canonical configuration is
[`data/benchmark_v3_factor_path/tournament.json`](../data/benchmark_v3_factor_path/tournament.json),
with SHA-256
`ff89911b64d8f6e652ed617f971ff0512dbda07e4c71bd585f5e0f23635fd399`.
The learning-free architecture freeze commit is
`bc69ac3c132c77d9613481ae163b1e8c4342a9eb`.

## Arms and causal questions

The four main arms form a preregistered 2x2 design. H0 uses a linear factor-head
hidden transform and H1 uses `tanh`; L0 injects all four factor distributions at
every decoder step and L1 injects only the factors licensed by the current target
grammar prefix. All four use identical parameter shapes and the same seeded
initial tensors. The comparison estimates activation and locality main effects
and their interaction on the fixed diagnostic-validation split.

Two controls use the same source encoder, decoder, factor heads, auxiliary losses,
training budget, and seed schedule. `D_AUX` predicts all four factors but does not
feed their distributions to the decoder. `NO_INPUT` replaces every source with the
constant `[BOS, SEP]` token sequence while retaining the same loss structure. It
therefore measures dependence on row-specific source information.

| Arm | Factor transform | Decoder conditioning | Trainable parameters |
| --- | --- | --- | ---: |
| H0L0 | linear | global | 32,120 |
| H0L1 | linear | prefix-local | 32,120 |
| H1L0 | tanh | global | 32,120 |
| H1L1 | tanh | prefix-local | 32,120 |
| D_AUX | linear | latent only | 31,448 |
| NO_INPUT | linear | global, constant source | 32,120 |

The maximum count is 33,000. The observed spread is
`(32120 - 31448) / 31448 = 0.021368608496565758` (2.137%), below the frozen 3%
limit, without dead padding.

## Shared execution contract

Each arm runs seeds 7, 17, and 29 on CPU with one Torch thread and deterministic
algorithms. Every run uses 600 updates, batch size 16, Adam at 0.003, global-norm
gradient clipping at 1.0, and the same seeded sampling schedule. The objective is
one byte-level language-model cross entropy plus unit-weight cross entropy for
participant, time, event, and operator.

The tournament requires all 18 arm/seed runs to reach a visible terminal state.
Completed runs must bind the tournament configuration, benchmark content,
initial state, final state, sampling schedule, and checkpoint file by SHA-256.
The final-confirmation split remains unavailable until this gate passes and may
be invoked only once with the frozen manifest.

For Issue #36 v1, the final gate also requires all 18 terminal records to be
`complete`. The canonical tournament JSON predates an explicit field for this
rule; adding one would change the frozen config digest embedded in existing
checkpoints. The all-complete rule is therefore retained as an explicit
final-gate binding for compatibility.

This document and configuration describe an untrained architecture comparison.
No benchmark-v3 training, diagnostic result, final-confirmation evaluation, or
claim about learned generation, glyphs, etymology, lexical senses, sememes, or
general Japanese understanding has been made at this phase.

## Frozen-contract implementation

The implementation now includes the two-template causal prefix FSM, strict row
preparation, the shared six-arm training runner, source-side swap and true
single-factor probability intervention diagnostics, authenticated checkpoints,
immutable terminal records, seed-level factorial aggregation, and an exclusive
final-confirmation invocation marker. The final gate requires 18 complete
checkpoints; terminal failures stay visible and cannot make the final gate ready.
The final result keeps the historical machine-readable `selection` key for
artifact compatibility; it is a frozen confirmation ranking only and cannot
trigger result-driven architecture or training changes.

Final intervention-pair diagnostics are selected from final target frames after
the final split is opened. They are descriptive probes of the frozen models,
are not model inputs, and are not selection evidence.

Luna implemented the FSM and checkpoint/protocol modules. Parent integration
added the runner, execution and final gates, corrected byte-ID validation and
all-complete final readiness, and verified the combined implementation with
564 passing tests (1 skipped). This is implementation evidence only: formal
training and both result splits remain unobserved at this point.

## Post-merge protocol erratum and future contract

Issue #39 found that the Issue #36 selection implementation used
`all.free_generation_exact.accuracy` although the preregistered primary was
`all.generation_frame_exact.accuracy`. Existing development and final artifacts
remain immutable. Their read-only audit is recorded in
[`docs/results/benchmark-v3-protocol-errata.json`](results/benchmark-v3-protocol-errata.json).

Future tournaments use this exact unweighted three-seed ordering, maximizing
each numeric metric before the final lexical arm-ID tie-break:

1. `all.generation_frame_exact.accuracy`
2. `all.triple_exact.accuracy`
3. `all.pair_exact.accuracy`
4. `all.atomic_balanced_accuracy.mean`
5. `all.exact_target_text.accuracy`
6. `arm_id_ascending`

The machine-readable future-only source is
[`future-protocol-v1.json`](../data/benchmark_v3_factor_path/future-protocol-v1.json),
whose canonical descriptor SHA-256 is
`bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c`.
It binds the exact primary, tie-breakers, derived source paths and reducer,
three-seed aggregation, direction, intervention rule, and protected usage
phases. `benchmark_v3_future_protocol.py` validates that descriptor and digest
before a future trainer, evaluator, final-marker publisher, or final-row
accessor runs, and requires future run metadata and returned artifacts to carry
the same digest. The historical Issue #36 runner, scorer, final gate,
configuration, checkpoints, marker, and attestation remain unchanged.

The balanced mean is explicitly the unweighted arithmetic mean of
`all.atomic_balanced_accuracy.{participant,time,event,operator}`; the contract
lists all four source paths rather than treating `mean` as a stored JSON field.
Future probability interventions choose a deterministic class different from
the baseline argmax using `(baseline_argmax + 1) % width`. They record both
class indices, require the actual intervened one-hot vector to select that exact
class, and retain the existing source-identity, decoder-prefix, and
non-target-probability checks in the separate future-only intervention module.
This future rule does not reinterpret or replace the historical fixed class-0
results.
