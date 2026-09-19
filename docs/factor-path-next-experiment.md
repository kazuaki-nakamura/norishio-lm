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

The corrected selection primary is exactly
`all.generation_frame_exact.accuracy`. Teacher-forced byte accuracy remains a
diagnostic and cannot replace free generation.

## Candidate comparisons

| Candidate | Learned arms | Maximum training | Resolves | Main limitation |
| --- | --- | ---: | --- | --- |
| A | H1L1 only | 3 seeds × 600 = 1,800 updates | True alternate intervention and head-versus-generation measurements | No learned negative-control arm |
| **B — preferred** | **H1L1 and D_AUX** | **6 runs × 600 = 3,600 updates** | Tests whether the factor path adds decoder follow-through beyond an otherwise matched latent-only arm | Does not re-estimate the old 2×2 activation/locality effects |
| C | H1L1, D_AUX, and NO_INPUT matched to H1L1 | 9 runs × 600 = 5,400 updates | Adds a trained grammatical-prior control | Larger than the residual question requires |
| D | Full v3 2×2 plus controls | 18 runs × 600 = 10,800 updates | Repeats the old factorial design | Repeats answered comparisons and is rejected for this plan |

Candidate B is preferred. It includes a **constant-source inference ablation**
for H1L1 by replacing the source with `[BOS, SEP]`; this is an out-of-distribution
diagnostic, not a separately trained NO_INPUT arm and not an information-destruction
control. If review requires a learned no-input comparison, use candidate C rather
than silently upgrading the ablation's evidential status.

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
- Arms: H1L1 and D_AUX; seeds 7, 17, 29; 600 updates; batch 16; Adam 0.003;
  gradient clipping 1.0; one deterministic CPU Torch thread.
- Hard budget: at most six attempted runs and 3,600 scheduled optimizer updates,
  with two CPU hours total across successful and failed attempts. No GPU, paid
  compute, network access, or result-driven tuning.
- Eight source-swap pairs and eight internal-intervention probes per seed are
  fixed before training: four factors × two support groups.

Parameter count, initialization family, byte vocabulary, optimizer, schedule,
sampling order, fixture manifest, probe identities, selection paths, and raw
artifact schema must be frozen in the descriptor before training.

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

For each arm, internal interventions also have denominator 24, split as six
observations per factor and 12 per support group, including parse failures. Keep
source identity and decoder prefix fixed, preserve all non-target probability
vectors, and replace one factor with the one-hot for
`(baseline_argmax + 1) % width`. Report separately:

- `target_changed`: generated factor differs from baseline output;
- `requested_value_success`: generated factor equals the selected alternate class;
- `non_target_preserved`: every other parsed factor is unchanged.

`target_changed` without `requested_value_success` is a wrong response, not
factor-level control. D_AUX ignores the injected probabilities by construction
and is the negative control for this diagnostic.

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
and changed tokens/text/parsed frames, and all scored booleans. For every internal
intervention retain both prefixes, both probability maps, baseline argmax,
declared/actual intervention class, canonical requested value, actual one-hot,
both generated outputs/parsed frames, and the three intervention outcomes above.

## Stop and falsification conditions

Stop before training or scoring on any descriptor/digest mismatch, fixture leak,
source/target overlap, gold-field input leak, parameter or initialization mismatch,
missing seed, duplicate probe, wrong one-hot class, changed non-target vector,
changed intervention source/prefix, missing raw field, or CPU-budget breach. A
failed or incomplete run set is reported as incomplete and produces no ranking.

The proposed factor-path claim is rejected or withheld when any of these hold:

The numeric cutoffs below are proposed review-time definitions and are not
historical results or approved thresholds.

- H1L1 does not exceed D_AUX on the frozen three-seed primary mean. The paired
  direction in each seed is reported as descriptive robustness evidence, not as
  a second selection prerequisite.
- Heads are correct while `requested_value_success` remains zero: decoder
  follow-through is unconfirmed.
- `target_changed > 0` but `requested_value_success = 0`: the intervention causes
  nonspecific changes.
- H1L1's constant-source ablation comes within 0.05 absolute on the three-seed
  mean of `all.generation_frame_exact.accuracy`: withhold source-dependent
  interpretation and require the separately trained NO_INPUT candidate before a
  positive claim.
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
