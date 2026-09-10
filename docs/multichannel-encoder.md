# Multi-channel CPU encoder

Issue #2 adds tensorization and trainable forward wiring. The dictionary remains
hand-authored, and the demo uses random weights. No semantic-quality improvement
or learned sense selection is claimed.

## Setup and minimal forward

Install a CPU build first to avoid pulling GPU runtimes. Run from an editable
source checkout (Windows commands; elsewhere use the environment's Python):

```powershell
.\.venv\Scripts\python.exe -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,model]'
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

PyTorch is optional for the dictionary compiler. Tensor/encoder tests skip when
PyTorch is absent; such a run does **not** validate Issue #2. All model tests must
run in the environment with the model extra installed. See the upstream
[installation guidance](https://pytorch.org/get-started/locally/).

```python
from norishio_lm import SemanticCompiler
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.encoder import EncoderConfig, MultiChannelEncoder

compiler = SemanticCompiler.from_json('data/demo_lexicon.json')
train_records = [compiler.compile('性'), compiler.compile('性器')]
tensorizer = SemanticTensorizer.fit(train_records)
batch = tensorizer.encode(train_records + [compiler.compile('未知語')])
model = MultiChannelEncoder(EncoderConfig(tensorizer.vocab_sizes))
state = model(batch, ablation={'etymology': False})
```

## Channel contract

The fixed order is `surface`, `tokens`, `morphemes`, `characters`,
`subcharacters`, `etymology`, `senses`, `sememes`, `concepts`, `relations`.
The compiler field `etymology_notes` maps to `etymology` only here.

| Channel | Categorical feature value |
|---|---|
| surface | Whole expression, including empty or whitespace input |
| tokens / morphemes / characters | Each supplied item, with position metadata |
| subcharacters | JSON pair of character and component |
| etymology | JSON pair of key and complete note |
| senses | Candidate sense ID; no candidate is selected |
| sememes / concepts | Each candidate's label; sense association is metadata |
| relations | JSON subject/predicate/object triple |

Each channel stores long `ids[B,L]`, boolean `mask[B,L]`, and feature metadata
for real (non-padding) positions. L is at least one, including missing channels.
Each feature carries its original value, record path, declared provenance and
optional candidate sense ID. Missing provenance stays `unknown`; no confidence
or source is invented. Metadata is retained by `.to(device)` but never fed into
the neural model. Inputs are validated and detached from the source record.

`ChannelBatch.layer_provenance[b]` holds the complete record-level declaration
for that row's channel, including rows with no features. For `senses`, `sememes`
and `concepts`, `Feature.provenance` separately holds the candidate-specific
declaration. A candidate's `unknown` is never promoted to a layer's `sourced`
declaration or merged with it. Other channels have only layer-scoped declarations,
also copied into their features. `etymology` uses the `etymology_notes` declaration.
Both scopes survive `.to(device)`; provenance never affects IDs, masks or gates.
The tensorizer always emits one layer-provenance tuple per row. Manually assembled
legacy `ChannelBatch(ids, mask, features)` instances default this field to `()`
(row-level metadata not supplied). Vocabulary checkpoint format is unchanged.

Fit deterministic, separate vocabularies on the **training split only**. Encoding
never grows them. ID 0 is padding (mask false); ID 1 is an observed out-of-vocabulary
feature (mask true). Missing semantics produce no feature. Save `to_dict()` as
JSON and restore with `from_dict()` alongside the encoder config and `state_dict`.
An equal vocabulary size does not prove equal ID meaning: always use the matching
vocabulary checkpoint. There is no automatic vocabulary/model identity check.
Observed feature keys in the JSON vocabulary are themselves JSON-encoded strings;
this keeps literal input such as `<PAD>` distinct from the reserved special key.

## Encoder and ablation

Every channel has its own embedding, masked mean pooling, linear projection and
tanh. Independent scalar gate heads produce a masked softmax across present,
enabled channels. Outputs are `fused[B,H]`, `channel_states[B,10,H]`,
`gates[B,10]` and `channel_mask[B,10]`. The states are after ablation; they remain
inspectable before fusion. Gate weights are model parameters' outputs, not
semantic confidence or feature provenance.

`ablation={name: False}` disables a channel for the whole batch; omitted entries
are enabled. Missing and disabled states/gates are exactly zero. If every channel
is disabled, fused output and gates are finite zeros. Disabled feature values do
not affect fusion or parameter gradients. Invalid shapes, IDs and channel names
fail explicitly. A batch must contain at least one record.

Channels are independent at the tensorizer and pre-fusion encoder boundary.
Changing glyph/etymology cannot rewrite a modern sense label. The fused latent
can depend on enabled glyph features; this is deliberate and requires future
comparison experiments. Removing `senses` at the encoder does not automatically
remove `sememes` or `concepts`. For full lexical-semantic removal, disable all
three plus `relations`; for orthographic removal also consider surface, tokens,
characters and subcharacters. `SemanticRecord.without_layers` retains its own
documented cascading behavior.

## Limits and next measurements

Pooling is order-insensitive. Candidate association, token order and source paths
are preserved for inspection, but not encoded as structure by this first model.
Relations and etymology notes are atomic categorical features, not a graph encoder
or a text encoder. OOV items in one channel share an embedding. No external
lexicon, contextual WSD, transformer, concept bottleneck, decoder, loss or training
loop is added here. Checkpoint mapping correctness is the caller's responsibility.

Tests of masks, perturbation invariance, gradient flow and CPU forward establish
wiring behavior only. Later work must compare equally trained baselines and
each channel ablation on held-out ambiguous words and misleading glyph examples.
