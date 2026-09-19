# Plastic memory toy protocol

Issue #37 evaluates a small, pluggable adapter while keeping the core model
fixed. It is a CPU-only associative-memory experiment, separate from the
SemanticCompiler channels and the benchmark-v2/v3 tournament fixtures.

## Boundaries

The experiment uses authored synthetic episode keys and values. It does not use
conversation logs, external user data, OKF records, glyph structure, lexical
senses, sememes, or concept labels. A successful lookup is not evidence of
general language-model quality, human-like memory, lifelong learning, or
semantic understanding.

The core fingerprint is recorded before every episode. Core parameters are
excluded from the optimizer and must remain bitwise unchanged. Only adapter
state or adapter parameters may change.

## Two gradient modes

`frozen_core` disables core parameter updates while retaining the ordinary
activation path into the adapter. `detached_features` additionally detaches the
core feature at the adapter boundary. The report keeps these effects separate:

- trainable parameter count;
- frozen parameter count;
- optimizer-state element count;
- whether a gradient path reaches the source input/core activation;
- an explicit backward-graph proxy.

These diagnostics do not claim exact wall-clock or FLOP reductions. Timing and
hardware-dependent speed are outside the first experiment.

## Episode and replay contract

Every episode has deterministic support keys/values and separate query keys.
Writes are permitted only during the support phase. Query evaluation is
read-only. The seed, ordering, adapter configuration, core fingerprint, episode
IDs, update count, and tensor-state digest are recorded.

An adapter checkpoint contains tensor state and JSON-safe metadata only. Loading
uses CPU weights-only deserialization, validates exact payload keys, dimensions,
finite float32 tensors, state/schema digests, and the expected core fingerprint,
and refuses to overwrite an existing output path.

## Consolidation

At least two fast adapters with the same configuration and core fingerprint are
combined deterministically into one slow adapter. The initial arithmetic-mean
rule is only defined for compatible state. Reports compare:

- no-memory behavior;
- each fast adapter on its own held-out queries;
- the combined fast-adapter behavior;
- the consolidated adapter on both held-out sets;
- absolute error/accuracy and per-episode retention drop.

Configuration or core mismatches are errors rather than implicit projection or
coercion. Interference and shared-key conflicts remain visible and are not
silently averaged away.

## Reproducible entry point

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.plastic_memory_demo --seed 37
```

The JSON output distinguishes fixed protocol metadata, observed metrics, and
interpretation limits. Generated checkpoints and raw reports belong under
ignored `codex/work_output/` paths.
