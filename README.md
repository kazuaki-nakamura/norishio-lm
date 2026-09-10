# norishio-lm

A research prototype for a **hierarchical semantic language model** that connects surface tokens to morphemes, sub-character structure, lexical senses, sememes, and higher-level concepts.

The core hypothesis is simple:

> A written token is not necessarily a semantic atom.

Instead of forcing all linguistic structure into a single opaque hidden state, Norishio-LM makes intermediate semantic structure explicit and testable.

## Target stack

```text
surface text
  -> tokens
  -> morphemes
  -> characters / sub-character structure
  -> lexical senses
  -> sememes
  -> concept graph
  -> concept latent
  -> reasoning / planning
  -> surface realization
```

Historical/etymological information is deliberately kept separate from modern lexical meaning so the model does not confuse attractive folk etymologies with actual semantics.

## v0.1 goal

Build and evaluate a `SemanticCompiler` that can represent a Japanese expression across multiple layers without collapsing them into one embedding.

Initial demo expressions:

- `性`
- `性器`
- `心生`
- `ハナレナイ`

## Quick start

This is a validated **dictionary-based SemanticCompiler**, not a trained language
model. Demo tokens are hand-authored splits, not learned tokenizer output.
The four original examples remain; `離れない` and `離れたくない` are also shown
to distinguish negation from negative desire. `心生` is marked experimental/poetic.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m norishio_lm.demo
pytest
```

On Windows, use `.venv\Scripts\python.exe` directly after creating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Schema 1.0 validates types, required sense fields, duplicate IDs, relation triples
and provenance. Legacy unversioned dictionaries remain readable. Records support
JSON round trips, unselected sense candidates, optional context/span metadata and
non-destructive layer removal. See [schema and API contract](docs/semantic-schema.md).

## Optional CPU encoder

`SemanticTensorizer` and `MultiChannelEncoder` preserve ten independent channels,
feature provenance and candidate associations, then combine channel vectors with
masked learned gates. Every channel can be disabled for ablation. Install the
optional PyTorch model dependencies and run the CPU forward demo using the
[encoder guide](docs/multichannel-encoder.md). The demo is randomly initialized;
it is not a trained language model or evidence of improved semantic understanding.

## Research questions

1. Does explicit semantic decomposition improve rare/novel compound understanding?
2. Does a sememe/concept bottleneck improve paraphrase and compositional generalization?
3. Can sub-character information help without contaminating modern semantics with etymological overreach?
4. Which layers are actually useful? This must be answered by ablation, not aesthetics.
5. Can concept-level planning be decoded into multiple surface forms or languages while preserving meaning?

See [docs/architecture.md](docs/architecture.md).

## AI development foundation

Project knowledge starts at [okf/index.md](okf/index.md). It records verified
implementation facts, research constraints, and open questions with source links.
This development knowledge is separate from the language model's semantic layers.

See [setup and verification](docs/ai-foundation.md) for the metadata index,
read-only local MCP server, maintenance rules, and upstream provenance.

Issues labeled `ai-ready` are handled by the local Codex development automation.
See [Issue automation](docs/issue-automation.md) for the posting contract,
execution requirements, retry labels, and human PR review boundary.
