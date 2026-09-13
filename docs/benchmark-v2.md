# Benchmark v2 freeze protocol

Benchmark v2 is the fixed evaluation substrate for Issue #34's architecture
tournament. It tests four-factor composition under one common holdout and
scoring contract. It does not establish general language-model quality.

The authored spec, deterministic generator, split seed, factor vocabulary,
surface templates, split rule, support classes, file hashes, and negative
control derangements are committed under `data/benchmark_v2/`. The freeze
commit is recorded as `BENCHMARK_FREEZE_SHA` before any v2 model training.

The main contrasts are:

1. participant-time pairs absent from train;
2. participant-time-event triples absent from train while their
   participant-time pair is present;
3. all four operators on every selected triple.

`diagnostic-validation` is available during the tournament. `final-holdout`
requires an explicit `evaluate_final=True` call bound to the committed content
digest. The Phase-2 evaluator must additionally expose only an
`--evaluate-final` command and verify every arm/seed checkpoint manifest before
opening it. The final split is evaluated once after all three seeds of all
frozen arms are complete.

Every arm reports atomic and balanced accuracy for participant, time, event,
and operator; pair/triple exact accuracy by train support; BOS self-history
slot, full-frame, exact-text, EOS, UTF-8 and output-uniqueness results;
intervention locality; and the 2x2 relationship between intermediate-frame
correctness and generated-frame correctness. Teacher-forced byte results remain
a separate diagnostic and cannot be described as free generation.

The broken-semantics control uses the manifest's predetermined per-factor
derangements. No result-dependent reshuffle is allowed. The source can be
encoded as bytes or characters and can expose preregistered atomic cues, but it
must never use a categorical ID for a complete source surface or complete
frame. Row IDs, template indices, split membership, targets, and gold labels
are unavailable as model features.
