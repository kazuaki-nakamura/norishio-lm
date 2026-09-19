# Next factor-path confirmation experiment (review unapproved)

**Status: planning proposal only.** This document does not authorize dataset
generation, training, checkpoint loading, inference, or final evaluation. Any
execution requires a separate explicit instruction after protocol review.

## Residual question

Benchmark v3 measured a large unseen-pair gap and reported 0/48 main-arm target
changes under a fixed class-0 intervention. Because baseline probability vectors
were not saved, that observation cannot answer whether a decoder follows an
intervention that truly selects a class different from its baseline argmax.

The smallest useful next experiment should distinguish:

- factor-head accuracy from decoder follow-through;
- source-side swaps from internal factor-probability interventions;
- parsed frame retention from exact surface-variant recovery;
- any target change from success at the requested alternate value.

The three working hypotheses are non-exclusive. **H_bypass** says row-specific
source information in decoder h0 competes with or dominates the explicit factor
path. **H_semantic** says the decoder follows the requested canonical factor
class. **H_continuous** says behavior also depends on the soft probability
shape within a class, so one-hot forcing and donor-soft transfer are not
interchangeable evidence.

The corrected selection primary is exactly
`all.generation_frame_exact.accuracy`. Teacher-forced byte accuracy remains a
diagnostic and cannot replace free generation.

## Candidate comparisons

| Candidate | Learned arms | Maximum training | Resolves | Main limitation |
| --- | --- | ---: | --- | --- |
| A | H1L1 only | 3 seeds × 600 = 1,800 updates | True alternate intervention and head-versus-generation measurements | No learned h0-path control |
| **B — preferred** | **H1L1 and H1L1_ANCHOR** | **6 runs × 600 = 3,600 updates** | Tests whether source-dependent decoder h0 is the bottleneck while factor heads, projections, and prefix-local path remain source-conditioned | Does not re-estimate the old 2×2 activation/locality effects |
| C | H1L1 and H1L1_ANCHOR, with historical D_AUX as optional context only | 6 runs × 600 = 3,600 updates | Adds the old latent-only comparison descriptively without a new D_AUX run | Historical D_AUX is a different fixture/result and cannot be new comparative evidence |
| D | Full v3 2×2 plus controls | 18 runs × 600 = 10,800 updates | Repeats the old factorial design | Repeats answered comparisons and is rejected for this plan |

Candidate B is preferred. It compares the ordinary source-dependent H1L1 decoder
initial state with an H1L1 variant named **H1L1_ANCHOR** whose decoder h0 is
source-independent from initialization and throughout training. H1L1_ANCHOR keeps
the same factor heads, factor projections, prefix-local gates, decoder, byte
vocabulary, loss, seeded sampling schedule, and training schedule. D_AUX may
appear only as historical context under candidate C; it must not add a learned
run or exceed the six-run budget. The constant-source inference ablation is
retained only as an OOD diagnostic and is not learned-path evidence or an
information-destruction control.

## Proposed freeze

Before any run, publish a new canonical experiment descriptor and digest under
the future-only protocol. The existing `future-protocol-v1` digest binds only the
inherited selection and intervention contract; it is not a complete freeze for
this proposed 96-row experiment. The old v1/v2/v3 data, checkpoints, markers,
results, and attestations remain immutable and unused.

- New authored fixture: 384 train rows and 96 fixed diagnostic-confirmation
  rows; no separate final split.
- Confirmation support: 48 `unseen_pair` and 48
  `seen_pair/unseen_triple` rows. Every atomic factor value occurs in train.
- New source and target surfaces must be mutually disjoint and disjoint from
  prior v2/v3 evaluation surfaces.
- Model inputs: UTF-8 `context` and `text` only. Row ID, support label, template
  ID, gold frame/text, and intervention class remain evaluator-side.
- Arms: H1L1 and H1L1_ANCHOR; seeds 7, 17, 29; 600 updates; batch 16; Adam 0.003;
  gradient clipping 1.0; one deterministic CPU Torch thread.
- Hard budget: at most six attempted runs and 3,600 scheduled optimizer updates,
  with two CPU hours total across successful and failed attempts. No GPU, paid
  compute, network access, or result-driven tuning.
- Eight source-swap pairs and eight internal-intervention probes per seed are
  fixed before training: four factors × two support groups.

The ordinary H1L1 decoder initial state is `h0(x) = self.h0(x)`, where `x` is
the row's source encoder latent and `self.h0` is the seeded 32→32 linear module.
H1L1_ANCHOR computes `anchor_latent = self.encoder([BOS, SEP])` once per batch,
expands that same fixed-source latent to every row, and uses
`h0(anchor_latent)`. The anchor is source- and row-identity independent from
initialization and throughout training, while the shared encoder, `self.h0`,
decoder, factor heads, factor projections, and local-gate path remain the same
modules with the same seeded initialization and exactly 32,120 trainable
parameters. No adapter, replacement parameter, or extra branch is introduced.
Factor heads and projections still consume each row's real source latent, and
the L1 prefix-local gate schedule remains unchanged. Because the anchor uses the
shared trainable encoder, decoder-h0 gradients still flow through
`encoder([BOS, SEP])` into shared encoder weights; this is the deliberate
gradient-routing/effective-function difference, not an added-capacity control.
The `[BOS, SEP]` **input** is fixed; `anchor_latent` is recomputed from the
current shared encoder and therefore remains trainable rather than being a
permanently frozen vector.

Before running, preregister these decision branches: preserved head accuracy and
ordinary generation frame accuracy together with improved
`requested_value_success` **and** `non_target_preserved` for H1L1_ANCHOR supports
the source-dependent-h0 bypass hypothesis; strong results for both learned arms
weaken that hypothesis; weak ordinary and bypass generation with strong heads
points to decoder follow-through or continuous-distribution limitations; weak
heads prevent a decoder-path conclusion. No branch treats donor-soft success as
proof of a learned joint factor distribution.

Parameter count, initialization family, byte vocabulary, optimizer, schedule,
sampling order, fixture manifest, probe identities, selection paths, and raw
artifact schema must be frozen in the descriptor before training.
The paired H1L1 and H1L1_ANCHOR pre-training `state_dict` tensors must be
bitwise equal and produce the same canonical state digest; only route/mode
metadata may differ. Digest or tensor mismatch stops the run before training.

## Metrics and denominators

The primary for each arm/seed is:

`all.generation_frame_exact.accuracy`

Its denominator is all 96 confirmation rows, including invalid or unparsable
generations. Selection uses the unweighted mean of seeds 7, 17, and 29. Each
support group is also reported with denominator 48 per seed; no subgroup replaces
the primary.

Required secondary fields:

- `all.exact_target_text.accuracy`: exact surface-variant recovery, denominator 96;
- `all.intermediate.row_denominator`: all 96 confirmation rows, and
  `all.intermediate.available_count`: rows with available factor-head predictions;
- `all.intermediate.frame_exact.accuracy`: complete canonical factor-head frame,
  with its denominator equal to `available_count`, not 96;
- `all.head_vs_generation_2x2`: head-correct/generation-correct counts with
  `contingency_row_count=available_count`; unavailable intermediates are reported
  separately and do not enter the 2×2 table;
- per-factor atomic and balanced accuracies using the frozen future paths;
- teacher-forced bytes, parse coverage, EOS, UTF-8 validity, and uniqueness as
  separate diagnostics.

For each arm, source swaps have denominator 24: eight fixed pairs × three seeds,
split as six observations per factor and 12 per support group. Report
`target_changed`, `non_target_preserved`, exact target text, and parse failures
over all 24. Report `target_change_correct` with its separate parseable/comparable
denominator; do not substitute that denominator for 24. A source swap changes
the source identity and is not an internal intervention.

For each arm, every frozen probe identity receives three internal controls, each
with a scheduled denominator of 24, split as six observations per factor and 12
per support group, including parse failures:

1. **Same-class soft-shape control:** let `k` be the baseline argmax and replace
   the target vector with `0.75 * one_hot(k) + 0.25 * uniform(width)`. Its
   argmax remains `k`, while its probability shape is changed. Record the L1
   distance from baseline; if it is at most `1e-6`, mark the control unavailable
   rather than counting an unchanged shape.
2. **Alternate-class one-hot:** replace the target vector with the fixed
   one-hot for `(baseline_argmax + 1) % width`.
3. **Alternate-class donor-soft transfer:** before training, bind each probe and
   factor to a donor table containing one fixed train-row ID for every possible
   requested class. Choose the lexicographically first eligible training row
   with that authored class, excluding all confirmation probes, and freeze the
   table in the descriptor. After each arm/seed is trained, but before any
   intervention generation or outcome scoring, compute the baseline probe maps
   and all pre-bound donor maps once, persist them with digests, and freeze
   them for every control. Then let
   `q = (baseline_argmax + 1) % width` and use only the donor pre-bound for `q`,
   and only when that donor's recorded soft factor vector has argmax `q`.
   Otherwise record `missing`; do not search for or substitute another donor.
   No generated outcome may influence donor selection or map construction.

For all three controls, source identity, decoder prefix, and every non-target
probability vector are held fixed. Record `target_changed` (generated factor
differs from the baseline output), `requested_value_success` (generated factor
equals the selected class/value), and `non_target_preserved` (every other
parsed factor is unchanged). Also report the joint outcome
`requested_value_success AND non_target_preserved`; target change alone is not
requested-value success. For each control type report scheduled, available,
missing, and scored denominators separately; missing or unchanged-shape
controls are unavailable, not failures or zeros. Available controls retain
parse failures in the scored denominator. Save the donor and probe probability
maps in full.

`target_changed` without `requested_value_success` is a wrong response, not
factor-level control. Donor-soft success measures decoder response to an
externally supplied soft vector; it does not prove a learned joint factor
distribution.

## Raw artifacts

Retain every confirmation row with:

- arm, seed, row ID, support group, and source identity;
- evaluator-side target frame and target text;
- all four factor probability vectors, canonical argmax values, and the complete
  intermediate frame;
- generated token IDs/text, EOS, UTF-8 and uniqueness flags;
- parsed frame or parse-failure reason;
- teacher-forced byte counts.

For every source swap retain both source identities, expected frames, baseline
and changed tokens/text/parsed frames, and all scored booleans with their
denominators. For every internal control retain the control type, both prefixes,
the complete baseline map for all four factors, the target replacement map,
all three unchanged non-target maps, the same-class L1 shape distance,
baseline argmax, declared/actual intervention class, canonical requested value,
actual one-hot when applicable, frozen donor table and selected donor
row/factor/map when applicable, availability flag, missing
reason when applicable, both generated outputs/parsed frames, parse failures,
and the three intervention outcomes plus the joint requested-success-and-
non-target-preserved outcome.

## Stop and falsification conditions

Stop before training or scoring on any descriptor/digest mismatch, fixture leak,
source/target overlap, gold-field input leak, parameter or initialization mismatch,
missing seed, duplicate probe, wrong one-hot class, changed non-target vector,
changed intervention source/prefix, missing raw field, or CPU-budget breach. A
failed or incomplete run set is reported as incomplete and produces no ranking.

The proposed factor-path claim is rejected or withheld when any of these hold:

The numeric cutoffs below are proposed review-time definitions and are not
historical results or approved thresholds. `Preserved` means that the H1L1_ANCHOR
three-seed mean is no more than 0.05 below H1L1 for both intermediate head-frame
exactness and ordinary generation frame exactness. `Improved` means an absolute
gain of at least 0.10 in the alternate-one-hot joint
`requested_value_success AND non_target_preserved` rate over the same 24
scheduled probes. A `strong` head rate is at least 0.80; a `weak` joint
follow-through rate is at most 0.05. A joint rate of at least 0.25 is the
proposed positive follow-through floor. These values must enter the new
descriptor before any run.

- If H1L1_ANCHOR fails either the preservation rule or the improvement rule for
  the joint `requested_value_success AND non_target_preserved` outcome, the
  source-dependent-h0 bypass hypothesis is not supported. Per-seed direction is
  descriptive robustness evidence; the comparison is decided by the frozen
  three-seed primary and the preregistered joint intervention outcome, not by a
  new factorial ranking.
- If both learned arms meet the strong-head and positive joint-follow-through
  floors and their ordinary frame means differ by less than 0.05, the h0-bypass
  hypothesis is weakened. If both arms are weak on joint follow-through while
  heads are strong, the result points to decoder follow-through or
  continuous-distribution limitations. A head-frame rate below the fixed
  strong-head floor prevents a decoder-path conclusion for that arm.
- Heads are correct while `requested_value_success` remains zero in the
  alternate-class one-hot control: decoder follow-through is unconfirmed.
- `target_changed > 0` but `requested_value_success = 0`: the intervention causes
  nonspecific changes.
- If the same-class soft-shape control changes parsed target values or lowers
  non-target preservation by at least 0.10 relative to the paired baseline,
  withhold a canonical-class interpretation because probability shape alone is
  behaviorally active. Stable same-class controls plus joint success for both
  alternate one-hot and same-requested-class donor-soft probes support class
  use. One-hot-only success leaves continuous soft-distribution following
  unconfirmed; donor-soft-only success shows shape sensitivity but does not by
  itself establish canonical-class control.
- H1L1's constant-source ablation comes within 0.05 absolute on the three-seed
  mean of `all.generation_frame_exact.accuracy`: withhold source-dependent
  interpretation. This OOD ablation is not learned-path evidence and does not
  authorize adding a trained NO_INPUT arm to this experiment.
- The three-seed mean `unseen_pair` frame exact remains near zero (defined here as
  at most 0.02) while the `seen_pair/unseen_triple` mean is materially higher
  (defined here as an absolute gap of at least 0.10): report continued
  fixture-specific compositional failure.
- Frame exact improves without exact target text: report frame recovery without
  claiming surface-variant reconstruction.

These criteria concern an authored structural-factor fixture. They cannot show
modern lexical meaning, sememe validity, glyph or etymological meaning, or general
Japanese generation.

## Return to the semantic stack

The factor fixture supplies known participant/time/event/operator labels; it does
not test the original surface → morphology → sub-character → lexical sense →
sememe → concept → generation stack. Before returning to that hypothesis, a
separate reviewed protocol must define sourced modern-sense labels, contextual
sense selection, misleading-glyph controls, layer-by-layer ablations, and a
baseline with the same capacity and training budget. It must measure sense and
sememe prediction, concept consistency, controlled paraphrase, rare-compound
generalization, and surface realization separately. Etymology and glyph features
must remain independent of modern lexical labels.

No adapter implementation, external dictionary retrieval, semantic training, or
experiment is authorized by this section.

## Evidence used

- [`docs/results/benchmark-v3-development.md` §§ Support-group split / Protocol erratum](results/benchmark-v3-development.md#support-group-split), result `5be2c9dde1714b5d9826e7e3569493b2540913af`.
- [`docs/results/benchmark-v3-final.md` §§ Frozen selection result / Intervention and interpretation limits / Protocol erratum](results/benchmark-v3-final.md#frozen-selection-result), result `119c6298a49213a156923c435edfab7233d5e9d0`.
- [`docs/factor-path-tournament-v3.md` § Post-merge protocol erratum and future contract](factor-path-tournament-v3.md#post-merge-protocol-erratum-and-future-contract), binding `77d24f1c7e8a0939bec5c689c50141f1067d51c0`, validation `d6e7451bd384e7649c464b2d15c1116df9443a26`.
- [`data/benchmark_v3_factor_path/future-protocol-v1.json`](../data/benchmark_v3_factor_path/future-protocol-v1.json), inherited selection/intervention-contract SHA-256 `bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c`; a future experiment descriptor and digest remain required.
