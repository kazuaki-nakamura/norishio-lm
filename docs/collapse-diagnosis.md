# Issue 9: boundary diagnosis plan

Plan fixed before measurement, 2026-09-11. Use unchanged authored v1 data;
train450 and validation150 only. Test is not evaluated or used for selection.
No additional seeds, data, dictionaries or paid compute.

- Train normal strict C from seed7 for 600 updates, batch16, Adam .003, clip1,
  all four loss weights1, CPU1thread. Reuse the existing training implementation.
  This is a diagnostic budget, not a change to the 60-step baseline defaults.
- Freeze the entire resulting model. Extract fused latents from context/text
  only. A separate linear seven-head probe uses train-only mean/std normalization,
  full-batch Adam .01 for 300 steps, seed7. No hidden layers or hyperparameter
  selection. Probe gradients cannot update source latents or encoder weights.
- Score probe and existing concept head against train-only majority, including
  per-field class support, balanced accuracy and complete known-frame matching.
- Report concept confusion (gold rows, prediction columns including unknown0),
  entropy in nats, population probability variance, and seed17 global permutation
  differences. Report raw exact unique rows and pairwise Euclidean distance
  min/mean/max plus fraction of pairs within tolerance1e-6; the tolerance count
  is not a transitive clustering or a semantic equivalence judgment.
- Compare predicted, final train-source mean, seed17 permuted and authored gold
  one-hot concept conditioning on the same decoder. Gold is an explicitly labeled
  oracle intervention, not attainable source-only performance. It may be outside
  the trained soft distribution, so either outcome alone cannot prove a cause.
- Greedy starts at BOS, uses only conditioning and self-produced history, and
  stops at EOS or cap128. Save raw tokens/bytes, UTF-8 validity and termination,
  all150 source/prediction/conditioning/generation/reference fields separately.
- Save normal600 checkpoint and complete JSON report in a new ignored directory.
  Existing outputs must not be overwritten. Record state equality around probe
  training and checkpoint reload checks. No auto-merge or Issue close.

Probe failure can reflect limited probe capacity/optimization, rather than lost
information. A high probe score establishes decodability for these authored
labels, not general meaning. Encoder distances depend on scaling. Low concept
diversity and low output diversity identify candidate boundaries, not causal
proof. Exact reference match is not the sole measure of linguistic correctness.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_collapse --out-dir codex/work_output/issue9-seed7-v1
```

Observed results and verification are recorded in `docs/handoff.md`.
