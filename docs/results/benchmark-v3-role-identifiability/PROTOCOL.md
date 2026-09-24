# Issue #48: source-role identifiability protocol

Status: implementation and pre-training audit in progress. No Issue #48 model
training has begun. A machine-readable descriptor, static signature audit,
fixture manifest, schedule hashes, paired initial-state digests, raw schema,
interpretation rules, and a Git freeze commit must exist before the first
attempt. Once frozen, these inputs and rules must not change after outcomes.

## Bounded question

On a new authored structural-factor fixture, does making the four source value
bytes role-distinct help the same H1L1 factor-only head path exceed the
role-aliased training ceiling? This is a paired input-encoding comparison,
not an experiment in decoder control, modern lexical meaning, sememes,
concepts, glyph interpretation, or general Japanese competence. The earlier
Issue #46 observations and frozen decision remain unchanged.

## Fixed arms and initialization

`ROLE_ALIASED` uses shared ASCII digits for participant, time, event, and
operator values. `ROLE_DISJOINT` changes only those four byte positions to
the disjoint alphabets `A..F`, `G..L`, `M..P`, and `Q..T`, respectively. The
new fixture, logical row IDs, split, gold labels, targets, source template
constants, and row order are identical between arms. Model input is only the
canonical JSON of `context` and `text`, surrounded by BOS and SEP.

Both arms instantiate the same 32,120-parameter `BenchmarkV3Model(H1L1)`
module tree and receive the same seed. Before freezing and before every run,
copy each of the 20 disjoint-byte encoder embedding rows from the matching
digit-byte row on **both** models; do not tie their parameters. Verify full
state-dict bitwise equality, identical digest, parameter count, paired
input-embedding sequence equality, and the recorded initial-output tolerance.
The disjoint arm activates 20 value embedding rows instead of the aliased
arm's 6, changing gradient sharing and effective freedom despite equal nominal
parameters and initial tensors. Do not attribute any improvement uniquely to
information loss.

## Fixture and static stop gate

Each arm has 384 train and 96 confirmation rows, with two source/target
variants per logical tuple. Train participant-time offsets are 2, 3, 4, 5;
events per offset are `{E0,E1}`, `{E1,E2}`, `{E2,E3}`, `{E3,E0}` and all four
operators appear. Confirmation has offset-1 unseen pairs with `{E1,E2}` and
`{O0,O1}`, and offset-3 seen pairs with held-out `{E0,E3}` and `{O2,O3}`;
each stratum has 48 rows. Offset 0 is unused. Exact selection, row order,
codebook, namespace, hashes, and class support belong in the frozen fixture
manifest and descriptor.

Before training, audit exact `source_ids` token-count signatures and
normalized-frequency signatures (integer count vectors reduced by their
greatest common divisor) over train, confirmation, each confirmation stratum,
and all 480 rows together. Include BOS, SEP, and JSON wrapper. Report
signature counts, collision groups/rows, conflicting gold signature counts,
and per-factor majority-label ceilings. `ROLE_DISJOINT` must have no
cross-frame collision and every factor ceiling 1.0; `ROLE_ALIASED`
participant and time train ceilings must each be below 0.80. Fail closed
before any run if these preconditions or pairing/coverage checks fail.
These ceilings are theoretical for a deterministic signature-only classifier
within each slice, not achieved model performance or an unconditional
floating-point limit.

## Execution, raw evidence, and limits

Run order is seed-major `ROLE_ALIASED` then `ROLE_DISJOINT` for seeds 7, 17,
29. Each run is exactly 600 updates of factor-only four-head cross-entropy,
batch 16, Adam lr 0.003 with betas (0.9, 0.999), eps 1e-8 and zero weight
decay, gradient clip 1.0, CPU Torch one thread and deterministic algorithms.
The same seed uses the same frozen with-replacement batch schedule in both
arms. Maximum: six attempts, 3,600 optimizer updates, and two hours executor
wall time. No retries, extra seeds, template changes, or outcome-based
extension. Persist each attempt start before execution, completion/failure,
raw hash, updates, initial/final state digest, training wall, and executor
wall. Missing wall evidence remains unavailable, never silently zero.

After each run, evaluate all 384 train rows as resubstitution, then all 96
confirmation rows, once each, with the source heads only. Save every row's
canonical four probability vectors, argmax and predicted frame, gold frame,
arm, seed, row ID, split/stratum, availability, and source/mask/signature
bindings. Validate saved raw and ledger before aggregation. Report atomic
and class-balanced accuracy, class support including zero-support strata,
missing/unavailable counts, paired seed deltas, and secondary four-head exact.
Collision-group probability/argmax differences are descriptive numerical
diagnostics; floating-point differences do not alter the mathematical
signature equivalence.

## Preregistered interpretation

For each factor, `strong_train` requires the disjoint arm's train
class-balanced accuracy to be at least 0.80 at **each** seed. `rescue` also
requires a strictly positive disjoint-minus-aliased train difference at all
three seeds and a disjoint three-seed mean at least 0.05 above the new aliased
arm's static train ceiling. A rescued factor's mean train-minus-stratum gap
of at least 0.15 marks that stratum as a future generalization question.
Both participant/time rescues support useful role-aware training on this
fixture, with the active-embedding caveat above. If disjoint participant/time
remain weak, optimization/effective capacity/conditioning remain unresolved;
do not rerun. A one-factor rescue narrows the next question. Strong train and
both confirmation strata at all seeds support authored factor readout only;
"Event/operator do not collapse" means disjoint-arm class-balanced accuracy
at least 0.50 and at least two distinct predicted classes in each confirmation
stratum at every seed. This is a conservative fixture-specific gate, frozen
before outcomes; zero-support classes are excluded from balanced means.
Decoder intervention would require a separate decision. Any learned score
apparently above its relevant static ceiling triggers a saved-raw audit of
mask, binding, rounding, and scoring before interpretation.

The decision record must state expectation versus observation, competing
explanations, minimum discriminating next test, adopted interpretation and
counterevidence, result-contingent next step, and the semantic-stack boundary.
No merge or Issue closure follows automatically from execution.
