# Concept bottleneck toy experiment

This is a CPU conditional generation experiment, not evidence of LLM performance.
All examples and semantic labels are authored_demo fixtures, not discoveries made
by the model. See [dataset provenance and split](../data/issue3/README.md).

## Observation and comparison

All paths observe the same complete source context/text and predict reference UTF-8
bytes with a one-token shift. PAD and source-prefix positions do not incur LM loss.
The fixed byte vocabulary contains 260 entries; feature and label vocabularies fit
only the 450 training rows. Validation and test each contain 150 rows, grouped by
person/time combination. The 24 diagnostic rows are reported separately.

- A reads the source prefix with a tiny causal GRU.
- B also conditions the GRU on MultiChannelEncoder output.
- C sends only predicted named concept probabilities to its decoder. Its decoder
  reads previous target bytes, never source bytes, source attention state, or the
  encoder latent directly. Gold labels go only to auxiliary loss.

The concept schema preserves event, ordered operators, agent, participant, time,
location and repeat_marked. NOT(WANT(...)) and WANT(NOT(...)) remain distinct.
UNSPECIFIED is an authored value; null is missing supervision. The concept heads
predict one source-level state, after observing the complete source, rather than
giving full-source labels to partial source prefixes.

Inputs are source-only surface/character features. No morphology, dictionary
senses, sememes, concepts, relations, glyph structure or etymology are invented.
The existing encoder's other channels remain available but empty in normal data.
Supervised output heads are separate from those input channels. Empty-channel
ablation cannot establish that a linguistic layer is useful or useless.

## Measurement limits

C uses soft categorical concept probabilities. These can carry information beyond
their argmax labels. Fixed-concept decoder invariance establishes the implemented
boundary, not a perfectly discrete semantic information bottleneck. Hard and zero
concept interventions are diagnostics, not separately trained baselines.

Training budgets and data order are matched, but parameter counts and decoder
sequence lengths differ across A/B/C. Same steps are not equal computational work.
One seed and short training cannot establish superiority. Evaluation uses teacher
forcing and measures reference byte loss, not open-ended generation quality.
Concept accuracy evaluates the author's toy labels, not universal meaning.

Each auxiliary loss can be disabled separately; null labels never become negatives.
Channel removal and concept interventions are recorded separately from retrained
loss ablations. Misleading glyph fixtures are artificial diagnostics, never actual
etymology or normal training input. Their feature IDs may be unknown to the
train-only vocabulary; enabled changes are not evidence of learned glyph use.

The test split is evaluated only after configuration is fixed, without selection
or adjustments based on its score. No external dictionary, paid compute, GPU,
private logs or OKF content is used as training data.

## Run and inspect

Use the optional CPU model environment described in the encoder guide. From the
source checkout root, run:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_experiment --seed 7 --steps 60 --batch-size 16 --out codex/work_output/issue3-seed7.json
```

Choose a new output path on each run. The report includes per-step loss components,
valid-label counts, named concept predictions beside explicitly authored gold,
train/validation/test losses, diagnostic-only measurements, channel removal,
hard/zero concept interventions, parameters and training time. `C_no_*` and
`C_lm_only` are trained afresh with matching budgets. Channel removals are
validation-time interventions, not retrained comparisons.

Public APIs live in `concept_model.py`: `ConceptVocabulary.fit(train_targets)`,
`encode_targets`, `ConceptBottleneck`, `BottleneckOutput.inspect` and
`TinyConceptDecoder.decode_with_concept_intervention`. Reserved categorical ID 0
is not a gold class; unseen or absent categorical supervision is masked. Only
training-defined semantic tags are scored. Whole-source `surface` features are
unknown on held-out combinations. The token and character input channels encode
the same observable characters with separate embeddings, not linguistic analysis.

The initial [measured results](handoff.md) include two identical complete seed=7
runs, excluding wall time, and explicit limitations. There is no saved pretrained
checkpoint or free-running generation quality result at this milestone.

The [external AI review](https://github.com/kazuaki-nakamura/norishio-lm/pull/6#issuecomment-5621273041)
reports that validation concept accuracy nearly matches a train-majority baseline,
and replacing per-input concept predictions with their fixed train mean barely
changes LM loss. These are reviewer-run post-hoc diagnostics, not additional local
verification. Zero interventions also change probability mass: their effect alone
does not prove semantic content is used. Next comparisons should retrain constant
conditioning with matched decoder initialization/budget, permute input-concept
alignment and assess free generation while preserving the v1 data and results.
