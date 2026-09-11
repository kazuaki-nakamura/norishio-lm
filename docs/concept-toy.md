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
checkpoint at this milestone. Initial teacher-forced results and the later failed
free-generation controls are recorded separately.

The [external AI review](https://github.com/kazuaki-nakamura/norishio-lm/pull/6#issuecomment-5621273041)
reports that validation concept accuracy nearly matches a train-majority baseline,
and replacing per-input concept predictions with their fixed train mean barely
changes LM loss. These are reviewer-run post-hoc diagnostics, not additional local
verification. Zero interventions also change probability mass: their effect alone
does not prove semantic content is used. Next comparisons should retrain constant
conditioning with matched decoder initialization/budget, permute input-concept
alignment and assess free generation while preserving the v1 data and results.

## Follow-up controls and free generation

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_controls --seed 7 --steps 60 --max-new-tokens 128 --out codex/work_output/issue3-controls-seed7.json
```

This separate command trains two copies of the identical initial C model with the
same sample schedule and optimizer settings. The constant arm freezes the mean
of the initial model's train-source predictions; it is not gold conditioning or
the final trained model's mean. Both retain the original auxiliary objectives.
The control severs the LM gradient into the encoder while retaining auxiliary
gradients. The shared global clipping recipe is unchanged, not an assertion of
identical per-parameter updates.

Validation compares normal predictions, the final train-prediction mean, globally
reversed concept alignment, and the separately retrained constant arm. Reversal
is over the complete validation order before batching. A majority baseline fits
train labels only, uses smallest vocabulary ID for ties and excludes missing labels.
No test or diagnostic split is scored by this follow-up command.

`toy_generation.greedy_generate(decoder, concept_probs, max_new_tokens=128)`
accepts no references or source rows. It initializes strict C's GRU from concepts,
starts with BOS and feeds back its own output tokens. EOS stops each row separately.
Incremental decoding is checked against full-history decoding. Raw argmax is not
masked to force valid UTF-8 or hide reserved-token emissions. Results retain raw
token IDs, UTF-8 validity, decoded display text, EOS and invalid-special diagnostics.
The report scores exact reference bytes plus EOS only after generation and records
all 150 validation examples for every condition. The cap is fixed independently
of reference length; failure to emit EOS is counted as failure to finish.

The local follow-up found near-identical validation losses, while all four arms
failed free generation (0 exact matches, 0 EOS completions, 0 valid UTF-8 sequences).
These are retained negative results, not repaired or described as fluent outputs.
See handoff for numeric results, two-run reproducibility, hashes and limitations.

## Byte/EOS diagnosis

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_diagnosis --out codex/work_output/issue3-byte-eos-diagnosis.json
```

This separate validation-only diagnosis fixes budgets at 60 and 600 updates,
seed 7, batch 16 and generation cap 128. Both normal and constant C are initialized
from the same model and use prefixes of the same training schedule. Existing
model code, data and default budgets remain unchanged; test is not scored.

Metrics distinguish actual illegal UTF-8 transitions from an incomplete final
codepoint caused by the cap. Teacher-forced byte accuracy excludes EOS and PAD;
EOS accuracy/probability uses only the true reference-end position. A preselected
first validation example records full-history greedy decisions, EOS probability
and rank, and equivalence to incremental generation, without giving that tracing
function a reference.

At 60 updates both arms have actual illegal byte transitions and zero gold-end
EOS argmax accuracy. At 600 updates both produce valid UTF-8 and terminate on all
150 validation inputs, supporting inadequate early training as a contributor to
the byte/EOS failure. Both still generate the same sentence for every input and
have zero exact matches. This does not establish conditional semantic generation.
The 600-update arms were run once, not a multi-seed or cross-environment result.
