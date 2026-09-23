# Issue #46: source-to-factor head learnability protocol

Status: pre-training freeze prepared. Descriptor SHA-256:
`210fdbab6bcd4f8604d8653835ea5f075df3a41a2c79fb0fbc95bc9a07967322`.
Fixture content SHA-256:
`b587c091e3499348c5e9bcfabff68ac3bb3bbd2c1df60edaac62833cf69cbed5`.
This document and the machine-readable descriptor must be committed before
any of the six bounded runs begin. Changes
to the fixture, split, schedule, model, loss, metrics, thresholds, or run order
after that point invalidate the comparison; failures remain in the attempt log.

## Question and claim boundary

The experiment asks whether the source encoder and explicit factor heads can
learn authored structural labels when the causal decoder objective is removed.
It does not test lexical sense, sememe, concept, glyph, etymology, general
Japanese competence, or free generation. Issue #44's completed runs and
decision remain immutable. This experiment reads none of its consumed finals,
checkpoints, markers, or attestations.

## Fixed comparison

`JOINT` and `FACTOR_ONLY` instantiate the same H1L1 module tree with the same
seed, initial state tensors, and trainable parameter count. `JOINT` minimizes
the existing sum of LM and four-factor cross-entropy losses. `FACTOR_ONLY`
minimizes exactly the four-factor loss; its decoder remains present but is not
trained by that objective. Both receive the same new 384-row training fixture,
the same per-seed with-replacement mini-batches, Adam learning rate 0.003,
batch size 16, gradient clipping at 1.0, and exactly 600 optimizer updates.
Each process uses one CPU Torch thread and deterministic algorithms. The seeds
are 7, 17, and 29; run order is seed-major, `JOINT` then `FACTOR_ONLY`. There
are at most six attempts and 3,600 updates total, with no retry or seed search.

The fixture has a disjoint namespace, 6 participant classes, 6 time classes,
4 event classes, and 4 operator classes. Every atom appears in training. The
participant-by-time train graph and held-out pairs, train row IDs, confirmation
row IDs, templates, and fixture digest are frozen in the tracked fixture
specification and descriptor. Confirmation has both unseen participant-time
pairs and seen pairs with unseen higher-order combinations, reported separately.
No row enters training because of a confirmation outcome.

## Raw and aggregate observations

After each run, evaluate all 384 training rows (explicitly resubstitution)
then all 96 confirmation rows, in frozen ID order. Save the four canonical
source-head probability vectors and argmax classes for every row, along with
target class, split, support stratum, arm, seed, and availability. Include the
initial and final state digests, schedule digest, loss trace, optimizer update
count, trainable parameter count, and training wall time. Persist each attempt
before starting the next. Record executor total wall time when available; an
absent timing is `null` with a reason, never zero.

For each arm and seed, report atomic correct/count/rate and class-balanced
accuracy for each factor on train and confirmation, confirmation by both
support strata, per-class denominators, and missing/unavailable counts. Joint
four-head exact match is secondary. Show paired seed deltas as
`FACTOR_ONLY - JOINT` without a composite score. Decoder quality and generation
ranking are out of scope.

## Pre-registered interpretation

The machine-readable descriptor fixes the exact threshold and precedence for
these labels before training. A label is a diagnostic direction, not a proof of
causality. The report must show all underlying factor-level numbers even when
one branch is selected.

- `joint_objective_interference_leading`: FACTOR_ONLY improves both participant
  and time confirmation in the same direction in all three paired seeds,
  participant/time training heads are strong, and event/operator confirmation
  heads have not collapsed.
- `compositional_generalization_leading`: both arms have strong participant
  and time training heads, but both lose accuracy on held-out
  participant-time pairs.
- `basic_optimization_or_capacity_unresolved`: FACTOR_ONLY is weak on training
  participant/time, so the JOINT objective alone cannot explain the problem.
- `narrow_factor_follow_up`: weakness is isolated to one factor.
- `fixture_specific_instability_leading`: arms are similar and the old head
  weakness is absent on this disjoint fixture.
- `mixed_or_inconclusive`: none of the above fits, or evidence is incomplete.

No branch changes the frozen Issue #44 conclusion. Authored fixture
classification does not demonstrate learned modern word meaning.
