# Benchmark v3 h0-bypass confirmation

Descriptor: `19c11e5fa7b5ae90dd2bc3b6ddca20366e6d2b9798b98e3ccf1c7c8bcdf10533`
Fixture: `296f32138f9ee448ca8fe975200047f53a6679343af35cbe7423d1e3435e1945`

Status: `complete`

This is a bounded authored structural-factor experiment. It is not evidence that a model learned modern lexical semantics, sememes, or concepts.

| Arm | Generation frame exact | Head frame exact | Alternate one-hot nontrivial joint / 24 |
|---|---:|---:|---:|
| H1L1 | 0.0312 | 0.0764 | 0.5000 |
| H1L1_ANCHOR | 0.0660 | 0.0833 | 0.2917 |

Decision: `decoder_path_inconclusive_due_to_weak_heads`.

- Preservation rule: `True`
- Improvement rule: `False`
- Strong-head prerequisite: `False`
- Constant-source closeness requires withholding a source-dependent interpretation: `True`
- Fixture-specific compositional failure condition: `True`
- Frame recovery without exact target text: `True`

The six runs used 3600 optimizer updates. Training took 115.189 CPU seconds with one Torch thread. No GPU, network data, or paid compute was used.

The low intermediate head-frame rates make the decoder-path interpretation inconclusive. The outputs are results on an authored structural-factor fixture and are not learned modern-semantic, sememe, concept, or linguistic-quality evidence.