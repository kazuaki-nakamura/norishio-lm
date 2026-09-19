---
type: Implementation
title: Plastic memory toy prototype
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-19 }
stale_after: 2026-10-19
sources:
  - id: protocol
    resource: docs/plastic-memory.md
    title: Plastic memory toy protocol and claim boundaries
  - id: adapter
    resource: src/norishio_lm/plastic_memory.py
    title: Associative adapter and deterministic consolidation
  - id: checkpoint
    resource: src/norishio_lm/plastic_memory_checkpoint.py
    title: Bound CPU tensor checkpoint format
  - id: experiment
    resource: src/norishio_lm/plastic_memory_experiment.py
    title: Deterministic episode and consolidation evidence
---

# Plastic memory toy prototype

Issue #37 implements a CPU-only associative adapter around a fixed identity toy
core. The authored numeric episode fixture is separate from the SemanticCompiler,
model lexical senses, sememes, concepts, OKF, and external user data.

The adapter is the only mutable component. Reports distinguish trainable and
frozen parameter counts, theoretical optimizer moment counts, actual optimizer
state, and an observed source-gradient proxy. The proxy is not a FLOP or timing
measurement. Checkpoints bind tensor state to the core fingerprint and episode
metadata and refuse replacement of an existing path.

Seed 37 shows an enabled/disabled difference, exact save/load replay, and zero
core drift on two tiny episodes. Arithmetic-mean consolidation preserves argmax
accuracy on this disjoint-key fixture while increasing MSE relative to summing
both fast adapters. These observations do not establish language-model quality,
semantic learning, human-like memory, or practical continual learning.
