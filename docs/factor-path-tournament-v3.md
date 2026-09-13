# Benchmark v3 factor-path tournament freeze

Issue #36 Phase 1 fixes the model comparison and execution contract before any
benchmark-v3 training or result observation. The canonical configuration is
[`data/benchmark_v3_factor_path/tournament.json`](../data/benchmark_v3_factor_path/tournament.json),
with SHA-256
`ff89911b64d8f6e652ed617f971ff0512dbda07e4c71bd585f5e0f23635fd399`.
The learning-free architecture freeze commit is
`bc69ac3da75be75b3f39e7d9f44fd2818f0d214d`.

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

This document and configuration describe an untrained architecture comparison.
No benchmark-v3 training, diagnostic result, final-confirmation evaluation, or
claim about learned generation, glyphs, etymology, lexical senses, sememes, or
general Japanese understanding has been made at this phase.
