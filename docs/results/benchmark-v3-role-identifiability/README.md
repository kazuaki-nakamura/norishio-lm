# Issue #48: source-role identifiability result

**Frozen decision:** `one_factor_rescued_narrow_follow_up`. Only the event
factor met the preregistered rescue rule. Participant and time did not reach
the 0.80 strong-training floor in any seed; operator also missed it. This is a
small, hand-authored **structural-factor source-head** experiment, not evidence
of learned modern word meaning, sememes, concepts, glyph semantics, decoder
control, or general Japanese generation.

The [pretraining protocol](PROTOCOL.md) and
[`experiment-descriptor-v1.json`](../../../data/benchmark_v3_role_identifiability/experiment-descriptor-v1.json)
were committed at `998ba03e0c4dff7e553bf1b21e0ca97c5d061e0a` before the
first learned attempt. Descriptor SHA-256:
`48388afdde53749620fb20a6c7fc72e4e42af6bf6e6dda02dee84589a0e461d6`.
No Issue #46 or #44 consumed checkpoint, final split, or historical raw run was
used as training or inference input.

## What the comparison fixes and changes

Both arms use the same 480 logical rows per seed (384 train, 96 confirmation),
gold frames, target texts, H1L1 module tree, 32,120 nominal trainable
parameters, initialization, and batch schedule. `ROLE_ALIASED` shares ASCII
digit value bytes across four source roles; `ROLE_DISJOINT` changes only those
four byte positions to role-specific `A..T` bytes. Before training, 20
role-specific embedding rows were copied from the corresponding digit rows on
**both** models, and all 480 paired input embedding sequences and initial
source latents matched exactly at each of three seeds. The rows were not tied:
the disjoint arm activates 20 value embedding rows versus six in the aliased
arm. Gradient sharing and effective freedom therefore differ; any gain cannot
be attributed uniquely to information loss.

The static audit of complete model `source_ids` (including BOS, SEP, and JSON
wrapper) found 140 unique token-count signatures across 384 aliased train
rows, with 340 rows in collision groups. The aliased majority-label train
ceilings were participant 216/384 (0.5625), time 240/384 (0.6250), event
236/384 (0.6146), operator 262/384 (0.6823). The disjoint arm had 384
unique train signatures, 96 unique confirmation signatures, 480 unique union
signatures, no conflicting frames, and ceiling 1.0 for every factor on the
audited slices. The [saved static audit](../../../data/benchmark_v3_role_identifiability/static-audit.json)
also records normalized-frequency signatures, both confirmation strata, and
their per-factor ceilings. A theoretical ceiling is not an achieved model
score.

## Saved-raw observations

Six attempts completed once, in seed-major aliased/disjoint order for seeds
7, 17, 29. Each made exactly 600 factor-only updates with batch 16, Adam
0.003, clipping 1.0, deterministic CPU Torch on one thread. The
[`attempts.jsonl`](attempts.jsonl) ledger has six starts and six completions,
with no failure or retry. All 2,880 scheduled head-output row observations
were available. Saved training-loop wall time sums to **19.033 seconds**;
[`execution-wall.json`](execution-wall.json) records **22.383 seconds** for the
executor, including evaluation and raw writes. Both are below the frozen
two-hour cap. Every [`run-*.json`](.) retains source-binding hashes, gold
frame, four probability vectors, argmax, and predicted frame; the raw-only
aggregator rechecks the fixture, ledger, run hashes, schedule, probabilities,
and frozen commit before producing [`summary.json`](summary.json).
An independent review found that the frozen aggregator does not require the
`final_state_sha256` field, although all six saved runs contain it. A separate
[saved-raw-only post-run audit](postrun-state-digest-audit.json) verified each
final digest is present, is 64 lowercase hex characters, and differs from its
run's initial digest. This is an append-only hardening check, not a revision to
the frozen branch or a model rerun. Requiring the field in a future executor
is an implementation improvement candidate.

| Arm / split | Participant | Time | Event | Operator | Four-head exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aliased train resubstitution | 492/1152 (0.427) | 475/1152 (0.412) | 669/1152 (0.581) | 664/1152 (0.576) | 16/1152 |
| Disjoint train resubstitution | 687/1152 (0.596) | 671/1152 (0.583) | 969/1152 (0.841) | 886/1152 (0.769) | 249/1152 |
| Aliased confirmation | 114/288 (0.396) | 112/288 (0.389) | 170/288 (0.590) | 173/288 (0.601) | 0/288 |
| Disjoint confirmation | 119/288 (0.413) | 113/288 (0.392) | 231/288 (0.802) | 230/288 (0.799) | 32/288 |

These are three-seed aggregated **atomic** rates. The complete `summary.json`
contains class-balanced rates, per-class support (including absent classes),
seed-level paired deltas, both 48-row-per-seed confirmation strata, and
missing/unavailable counts. Full train class support is balanced, so its
atomic and class-balanced rates agree. Confirmation strata contain different
class subsets; their class-balanced values must be read from the saved summary.
In the disjoint arm's unseen-pair confirmation, participant was 39/144
(0.271) and time 29/144 (0.201), versus 80/144 (0.556) and 84/144 (0.583)
on the seen-pair/unseen-triple stratum. This is a fixture-specific gap, not a
generalization claim. Train is resubstitution, not a held-out estimate.

Across seeds 7/17/29, disjoint **train class-balanced** scores were:
participant 0.578/0.607/0.604; time 0.560/0.648/0.539; event
0.859/0.836/0.828; operator 0.799/0.776/0.732. All four disjoint-minus-aliased
train differences were positive at all three seeds, but only event was above
0.80 at each seed and more than 0.05 above its aliased static ceiling in the
three-seed mean. Event and operator passed the frozen noncollapse check in
both confirmation strata at every seed; participant/time did not satisfy the
authored-readout condition. Within colliding aliased signatures, saved raw had
zero argmax disagreements and a maximum probability difference of
2.98e-7, a floating-point diagnostic that does not remove the mathematical
signature collision.

## Decision and next falsification

**Expectation:** separating source roles might lift participant/time above the
aliased representational ceiling. **Observation:** role-distinct values remove
the audited signature collisions and improve all four train heads, but only
event satisfies the preregistered rescue rule; participant/time remain weak
even on training rows. **Adopted interpretation:** the aliased fixture has a
definite role-identifiability defect, while role distinction alone under this
model and training budget does not establish usable participant/time heads.
The event rescue is a narrow result on authored structural labels.

**Competing explanations:** optimization, masked-mean architecture/capacity,
and changed active embedding rows/gradient sharing may contribute to the
remaining errors and observed gains. The minimum next test should be a new,
preregistered CPU control that varies role observability while explicitly
matching or measuring active embedding rows and head capacity, with a fixed
train-strength gate before inspecting confirmation. Do not extend or retry
these six runs. A decoder intervention remains premature until participant
and time heads are strong on training and confirmation. The old Issue #44 and
#46 frozen decisions are unchanged.

Validation: pretraining focused tests **16 passed** and the final full explicit
compiler/foundation suite **753 passed, 1 skipped, 2 subtests passed**.
Post-run raw-only aggregation reproduced the committed summary without model
forward or further training. OKF validation reported 0 errors and 0 warnings;
the full knowledge index checked 308 sources with 0 errors, and MCP self/live
checks passed. `gpt-5.6-luna` handled bounded fixture, model, and read-only
review subtasks; the parent rewrote the fixture, integrated and verified all
work, froze the protocol, executed the runs, and interpreted the raw evidence.
No human verification mark is implied.
