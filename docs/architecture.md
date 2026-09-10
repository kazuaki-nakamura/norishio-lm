# Architecture sketch

## 1. Why another tokenizer?

This project is not primarily about finding better byte boundaries. It tests whether a language model benefits from an explicit hierarchy between text and concept.

A character can be a token without being a semantic atom. A word can be a useful language-model unit without being a concept atom.

## 2. Separation of evidence channels

The design keeps these distinct:

### Surface channel
What was literally written.

### Morphology channel
How an expression decomposes in the contemporary language.

### Sub-character channel
Radicals/components/glyph structure. Useful as a feature, but not automatically a semantic definition.

### Etymology channel
Historical formation and diachronic notes. This is evidence about history, not a direct definition of current meaning.

### Lexical-sense channel
Context-dependent current senses.

### Sememe channel
Smaller reusable semantic primitives attached to each candidate sense. The current compiler does not select a sense.

### Concept channel
Entities, events, attributes, relations, intentions, temporal structure, etc.

## 3. Proposed trainable stack

```text
surface ids ─────────────┐
morpheme features ───────┤
sub-character features ──┤
sense candidates ────────┼─> gated/factorized fusion -> transformer
sememe candidates ───────┤                          |
concept hints ────────────┘                          v
                                             concept bottleneck
                                                    |
                                      concept planner / decoder
                                                    |
                                              surface decoder
```

Do not simply sum every embedding. The model should be able to gate or attend to channels independently.

## 4. Candidate losses

```text
L = L_lm
  + alpha * L_morphology
  + beta  * L_sense
  + gamma * L_sememe
  + delta * L_concept
  + eps   * L_reconstruction
  + zeta  * L_contrastive
```

Each auxiliary objective must be removable for ablation.

## 5. Key falsification experiments

Compare:

- A: normal tokenizer baseline
- B: + morphology
- C: + sub-character
- D: + lexical sense
- E: + sememe
- F: + concept bottleneck
- G: all channels

Evaluate:

- rare/novel compounds
- word-sense disambiguation
- paraphrase equivalence
- compositional generalization
- robustness to misleading glyph cues
- controlled generation from concept structure
- hallucination / semantic drift

If G does not beat simpler systems on targeted tasks, the extra hierarchy is not justified.

## 6. First milestone

The implemented dictionary compiler now has schema validation, provenance,
JSON round trips, candidate-preserving context/span metadata, and a layer-removal
API. See [schema 1.0](semantic-schema.md). These are infrastructure, not evidence
that any added semantic layer improves modeling performance.

Issue #2 implements `SemanticTensorizer` and `MultiChannelEncoder`: ten separate
categorical vocabularies, provenance metadata, masked mean channel encoders and
learned scalar gates. Missing and disabled channels contribute zero; all-disabled
rows are finite zeros. The CPU forward demo uses random weights.
See [tensor shapes, ablation and limitations](multichannel-encoder.md).

This first encoder pools features without order or graph structure. Candidate
association is preserved as metadata, not resolved by contextual WSD. Glyph and
etymology do not rewrite modern labels; enabled glyphs can affect the fused
latent and must be tested by ablation. Issue #3 adds named concept heads, masked
LM/sense/sememe/concept losses, and a tiny conditional GRU decoder. A/B can read
the complete source prefix; strict C reads target history plus predicted concept
probabilities only. Auxiliary sense/sememe predictions never bypass that boundary.
See [the toy experiment](concept-toy.md). Semantic usefulness and general language
generation quality remain unverified; the source encoder still has no sequence
or graph structure.
