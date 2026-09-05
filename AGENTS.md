# AGENTS.md

## Project intent

Norishio-LM is a research codebase for testing a vertically integrated semantic representation stack:

`surface -> morphology -> sub-character -> lexical sense -> sememe -> concept -> generation`

The repository must remain experimentally falsifiable. Do not add semantic layers merely because they are elegant.

## Non-negotiable design rules

1. **Do not equate glyph decomposition with modern semantics.**
   - Example: `性 = 忄 + 生` is orthographic/etymological structure.
   - It must not be treated as proof that modern `性` literally means "a heart being born".
2. Keep separate channels for:
   - surface form
   - morphology
   - sub-character/glyph structure
   - etymology
   - modern lexical senses
   - sememes
   - concepts/relations
3. Every added semantic channel must have an ablation path.
4. Prefer interpretable intermediate schemas over magic constants.
5. Unit tests should include ambiguous words and intentionally misleading character decompositions.
6. v0.x should be CPU-runnable. Do not introduce large-model training as a prerequisite for basic tests.

## Initial implementation order

1. Semantic schema and compiler
2. Lexicon adapters
3. Layer encoders
4. Fusion module
5. Explicit concept bottleneck
6. Tiny causal LM baseline
7. Multi-task losses
8. Ablation harness
9. Japanese compositional benchmark

## First benchmark examples

- `性格`, `性質`, `性別`, `性器`, `可能性`, `人間性`
- paraphrase family: `また会いたい`, `明日もいてほしい`, `離れたくない`
- novel/rare compounds where character cues may or may not help

## Coding style

- Python 3.11+
- typed public APIs
- dataclasses / pydantic-like explicit schemas before tensorization
- pytest
- small, inspectable modules
- no hidden network access in tests
