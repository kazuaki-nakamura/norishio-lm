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
Smaller reusable semantic primitives attached to a selected sense.

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

Turn `SemanticRecord` into tensors while preserving provenance of every feature. Then train a tiny model that predicts both next token and explicit semantic labels.
