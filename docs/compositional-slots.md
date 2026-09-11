# Compositional slot diagnosis (Issue #30)

Pre-measurement plan from merge `d844e1667475bf5926b8c503ade7e2997a4db511`.

A: replay saved independent-head D seed7/17/29 from Issue26. B: replay saved
flat25-way pair head seed7/17/29 from Issue28. Authenticate report/checkpoint
SHA256 and require complete recorded evaluations to match. No retraining.

C reuses A's existing independent heads and weights, adds zero parameters and
zero updates. Define q[p,t]=softmax(p_logits)[p]*softmax(t_logits)[t], participant
major Cartesian ordering fit only from train450. Compute decoder marginals
by summing q over the other axis; no second head path. Mathematically these
equal A's probabilities because each distribution sums to one. Float32
product/sums can round: record max absolute marginal/logit differences, exact
equality, allclose(atol=rtol=1e-5), and greedy-token equality rather than claiming
bitwise identity from algebra alone. Keep D's decoder, gate and semantic base.

Do not add arm D: factorized gold-pair NLL is exactly participant CE + time CE.
Adding it at weight1 would double the existing head CE weights, not introduce
a pair-interaction objective. Verify the stable log-softmax loss and gradients
on synthetic logits; do not invent a new loss after observing C.

## Fixed measurements

- A/B historical complete3seed replay; C normal and both-gold generation,
  LM/EOS/UTF8/unique/participant/time/both/fullframe. Oracle is not ordinary
  performance and is only an intervention reference, not a guaranteed bound.
- C marginals: existing accuracy/balanced/Brier/NLL/ECE and joint argmax.
- C and B pair-space on validation150 (all unseen gold pairs): raw gold
  probability/rank, accuracy, NLL/Brier/ECE, top-k gold coverage k=1,3,5,10,
  categorical entropy (natural logs), probability mass on the ten pairs absent
  from train. Rank ties use ascending Cartesian index. Also retain train
  diagnostics separately, with empty-group rates null. No label-based tuning.
- Record every row and each seed plus unweighted arithmetic3seed means.
  n3/common extension seeds, authored grammar and fixed split limit inference.
- Require frozen states unchanged and checkpoint reload; no new checkpoints
  needed for C because it is a deterministic view of A, not a new model.

## Byte diagnostic repair

Move unreachable teacher_forced construction out of the error path. Explicit
opt-in `include_teacher_forced=True` returns version `teacher-forced-byte-v2`;
default legacy output remains null for exact archival replay. This is deliberate
compatibility, not a claim that old byte metrics existed. Compute v2 for seed7
saved D with predicted/participant-gold/time-gold/both-gold conditions. Correct
past history is used only for LM/byte scoring; free generation sees self-history.
Self-baseline span metrics measure byte quality, not intervention sensitivity.
Write a new report directory, preserve Issue26/28 reports without rewriting.
No training, data/split changes, test evaluation, paid GPU, worker restart,
automatic merge/close, or claims of general semantic understanding.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_compositional_slots --baseline-report codex/work_output/issue28-fixed-v1/report.json --out-dir codex/work_output/issue30-fixed-v1
```
