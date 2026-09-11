# Explicit slot-head comparison (Issue #20)

Fixed before measurement from PR #19 merge
`429cd8ea9718579502f8ac047187389b044f6e6c`.

- Reuse saved A/B from Issue #18, verify report SHA256
  `9db04ef46231cfb79439f1dc4f90883567b3f80433a7b8fc295cb2f2fcc1a382`,
  checkpoint hashes from that report, and replay the complete A/B evaluation
  (head statistics, all four conditions, logits-based byte metrics, all outputs).
  Preserve A/B artifacts. No need to retrain already matched historical controls.
- C copies the seed7 common initial model, NOT trained A/B weights. All shared
  parameters and the first33 conditioning columns/bias remain exactly equal.
- C adds participant and time Linear(32,5) heads on encoder fused latent.
  Class order is the existing train-only vocabulary IDs1..5, shifted to0..4;
  no unknown output class and no validation fitting. Keep existing33 concept
  probabilities (including their participant/time groups) and append the two
  five-class softmax distributions: conditioning dimension43.
- Decoder projection becomes Linear(43,32), followed by tanh; use it as h0 and
  additive input at every token exactly as existing per_step_additive. Gold
  values never enter the normal model forward path; no raw source latent enters
  decoder outside concept/slot probabilities.
- In fork_rng with seed20, initialize new heads in participant,time order, then
  expanded projection using default nn.Linear initialization. Copy old33 columns
  and old bias afterward. Caller RNG unchanged. Added parameters:330 head +320
  projection columns =650; total43073 versus historical42423. New column bounds
  follow expanded fan-in43. Existing model/defaults are unchanged.
- Train C from this initial state: seed7 sampling schedule,600 updates,batch16,
  Adam .003,clip1,CPU1thread. Existing LM/concept/sense/sememe weights remain1;
  add mean participant head CE at1 plus mean time head CE at1. NO slot byte CE.
  The added capacity and loss differ from B, so C is not a capacity-matched proof.
- Data v1 train450/validation150; test unused. Fit only train. No seed/weight search.

## Evaluation and interventions

- Predicted C: old concepts + predicted dedicated heads. Participant/time-only
  oracle repairs BOTH the corresponding old concept group and dedicated head.
  Full oracle repairs all old groups and both dedicated heads. Clearly separate
  these oracle diagnoses from normal performance.
- Dedicated-head interventions keep all old concept probabilities predicted:
  predicted, post-training train mean, global seed17 torch.randperm pairing,
  and both-head gold one-hot. Predicted is shared with the normal condition;
  save seven distinct conditions total. No intervention enters training.
- Report five-class head support/confusion/balanced accuracy/entropy/probabilities,
  old concept metrics separately, all-target validation LM, teacher-forced slot
  byte NLL/rank/argmax, and existing BOS greedy EOS/UTF8/unique/slot/frame scores.
  Preserve all-row and parse-only denominators and raw generated tokens.
- Sensitivity: same seed17 permutation, swapping ONLY participant head or ONLY
  time head distributions between rows; old concepts and other head unchanged.
  Hold authored prior history fixed. Record logit L1/KL(base||intervention)/argmax
  change at each slot's start-1 and within each slot, with actual byte positions
  and row/byte denominators. This uses explicit reference history, not free output.
- Save dedicated checkpoint with primitive metadata/tensors, weights_only CPU
  loading, class order/dimensions validated. Reload must reproduce state, old
  concepts, dedicated heads, validation logits and all150 greedy outputs exactly.
- Tests isolate head CE -> heads/encoder and LM -> head distribution gradients;
  future labels/input causality, source allowlist, parameter copying, and reload.

All outputs in a fresh ignored directory; no training artifacts or secrets in Git.
Full-frame0 is still a valid failure result. No test tuning, paidGPU, external
dictionary/private logs, worker restart, automatic merge/close.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_explicit_slots --baseline-report codex/work_output/issue18-seed7-v1/report.json --out-dir codex/work_output/issue20-seed7-v1
```
