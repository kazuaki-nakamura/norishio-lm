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

## Frozen implementation and execution

The implementation uses a target-only causal decoder during training:
`BOS + preceding target bytes` predicts `target bytes + EOS`. Source bytes go
only through the masked source encoder. A, C, and E therefore cannot bypass
their factor bottleneck through the raw source latent. B applies four separate
bias-free probability projections under the prefix-only FSM gates and one
shared bias. D conditions on its plain latent adapter.

Before an optimizer is created, the runner validates the generated corpus
against `expected-manifest.json` and the tournament digest, then checks the
actual trainable parameter count. Checkpoints reconstruct the initial model and
batch schedule from the arm and seed, validate the exact state schema and
tensor bytes, and use exclusive publication. Terminal JSON is strict and is
re-authenticated from its checkpoint at final intake.

Development locality uses four deterministic pairs selected only from the
active evaluation split. Each pair differs in one factor and agrees on the
other three. Gold frames select the comparison pair; the model receives only
the two source `context`/`text` records. The generated before/after frames are
then scored for the requested change and preservation of the other factors.

From a source checkout with the model extra installed, run the complete frozen
development tournament into a new ignored directory:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.benchmark_v2_execute --all --output-root codex/work_output/benchmark-v2-tournament
```

The driver writes every per-seed development report, checkpoint, terminal
record, and a three-seed arithmetic-mean summary. A failed run remains a failed
terminal record and is not converted into a zero score.

Only after all 18 terminal records exist, consume the final holdout once:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.benchmark_v2_final --evaluate-final --root codex/work_output/benchmark-v2-tournament
```

The exclusive `final-invocation` marker is created before final rows are
opened. A callback or write failure after that point is recorded as a consumed
failed invocation and cannot be retried in the same output root.
