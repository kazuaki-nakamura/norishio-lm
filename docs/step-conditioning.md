# Issue 12: initial-only versus per-step conditioning

Plan fixed before measurement, 2026-09-11. Use v1 train450/validation150 unchanged;
test is not evaluated. Compare only mandatory A/B; gated C is out of this run.

- A: existing strict C decoder, `h0 = tanh(concept_projection(p))` only.
- B: the same h0 plus the same projected vector added to every token embedding
  before the GRU. Reuse the existing projection, with no added weights, gate or
  hidden-size changes. Both have hidden32 and concept dimension33. Every initial
  parameter key/value is identical. The initial-only default remains unchanged.
- Independently train both arms from that identical state: seed7, 600 updates,
  same batch16 sampling sequence, Adam .003, clip1, all four unit loss weights,
  CPU1thread. No additional seeds or budget selection using results.
- Evaluate predicted, train-source prediction mean, globally permuted (seed17)
  and authored gold one-hot concept conditions. Gold is explicitly oracle, not
  source-only performance. Train only with predicted concepts and normal losses.
- Greedy from BOS uses its own tokens; EOS or cap128 stops. Save all150 sources,
  predictions, interventions, raw generated tokens/bytes, validity, termination
  and references separately. No reference enters normal conditioning/generation.
- Slot scoring is deliberately restricted to complete exact authored target
  grammar matches, with allowed person/time captures. No substring-based meaning
  inference. Reject malformed UTF-8, specials, unterminated outputs, unmatched or
  ambiguous templates. Report coverage and all-row gold-observed denominators;
  unparseable output is not silently dropped or counted correct. Also report
  conditional accuracy among parseable observed fields, distinctly labeled.
- Sensitivity holds each arm's predicted greedy history fixed and swaps only
  concept conditioning to mean/permuted/gold. At every active decode position,
  report mean absolute logit difference across vocabulary, KL(base || intervention)
  and argmax change rate with row count. Prefix is BOS plus baseline self-output;
  EOS-ended rows stop contributing. This is a fixed-history intervention, not the
  different trajectories that would follow each intervention freely.
  A cap128 row contributes decisions0–127 with BOS plus its first127 emitted
  tokens as input; the ungenerated129th decision is not evaluated.
- Save both checkpoints in a new ignored output directory. Require reload equality
  of logits/concepts/greedy/state for all150 validation cases in both arms.

A change in diversity or mechanical slots is a limited authored-corpus observation,
not general semantic understanding. Oracle is a distribution-shifting diagnostic.
Zero improvement is a valid result. Retain old results and the 60-step defaults.
No paid compute, external data, automatic merge, Issue close or worker restart.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_step_experiment --out-dir codex/work_output/issue12-seed7-v1
```

Implementation/results and verification are recorded in `docs/handoff.md`.
