# Architecture tournament v2 preregistration

This protocol is frozen after `BENCHMARK_FREEZE_SHA`
`19ce7145495a47b40a178255272050caa615dc79` and before any benchmark-v2 model
training. The canonical machine-readable contract is
`data/benchmark_v2/tournament.json`.

All arms use the same source bytes, 16-dimensional byte embeddings, masked
mean plus a 32-dimensional source latent, and a one-layer 32-dimensional causal
byte GRU decoder. They use CPU, one thread, deterministic algorithms, 600
updates, batch 16, Adam 0.003, global norm clip 1, and seeds 7/17/29. Greedy
generation starts from BOS and uses only its own history. No hyperparameter
search or result-driven rerun is allowed.

| Arm | Frozen factor path | Parameters |
|---|---|---:|
| A_G0 | Four predicted soft factors, global conditioning, end-to-end LM gradient | 29,272 |
| A_G1 | A_G0 with LM gradient stopped only before factor heads; factor CE remains | 29,272 |
| B | Four independent 32-to-8 representations and projections, prefix-driven local injection | 29,848 |
| C | Four hard predicted symbols composed from per-factor embeddings; no frame ID | 29,816 |
| D | Plain latent two-layer adapter, no semantic head | 29,532 |
| E | A_G0 with the manifest's predetermined factor derangements | 29,272 |

The maximum trainable-count difference is below 3%. Counts include every
`requires_grad` tensor and must match before the first optimizer step. There is
no post-result capacity padding. D has no factor loss; the other arms use LM
weight 1 plus four factor CE weights 1. This objective difference is part of
the frozen family definition and must be reported as a limitation.

A, B, C, and E condition generation only on predicted factor values. C uses
factor-wise symbols and embeddings and cannot form a 576-way frame vocabulary.
E changes semantic supervision/conditioning labels while keeping true target
text, so it is a label-semantics control rather than proof that factor
information was destroyed. B compiles every allowed target grammar string with
literal/participant/time/predicate byte tags. After each emitted prefix it
retains all matching grammar candidates. A factor gate is one only when every
remaining candidate tags the next byte with that factor; event and operator
share the predicate tag. Literal, ambiguous, completed, unknown, or malformed
prefixes make the affected gates zero. Target spans, future bytes, gold text,
and hidden template IDs are unavailable.

The development report uses `diagnostic-validation` and reports every seed,
the three-seed arithmetic mean, failures, common free-generation metrics,
intermediate heads, interventions, and teacher-forced bytes as separate
diagnostics. A single favorable seed is not a success claim.

One `--evaluate-final` invocation may evaluate all completed checkpoints only
after all 18 arm/seed records are terminal and every complete checkpoint
manifest matches the benchmark/config/state/schedule/file hashes. Failed runs
remain explicit and are not converted to zero or silently replaced. The final
invocation cannot change architecture, budget, seed, or checkpoint.

`TOURNAMENT_FREEZE_SHA = 38e363e1b4e836b890f1c852f6d0a45fe3119296`.
It is recorded outside the canonical JSON to avoid hashing a commit into its
own configuration. No benchmark-v2 model training occurred before or as part
of that commit.
