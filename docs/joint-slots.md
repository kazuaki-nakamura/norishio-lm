# Joint-slot diagnostic (Issue #22)

Pre-measurement plan from merge `10768943bf9e52afc7929dc8557172a0024d03ca`.

- A is saved Issue20 C. Require report SHA256
  `66e8c507dc5485d5e01825882688aeb8e34f0d6cbbbba36b0ce14d4960761ad8`
  and checkpoint SHA256
  `8bba5fb4915fc0435af474fae9649bf3b3b8e86737968c1f97271c7e77850956`.
  Replay the complete seven-condition evaluation and sensitivity before new work.
- B splits the43-input linear projection into base(33,32,bias), participant(5,32,
  no bias), time(5,32,no bias); sum their outputs BEFORE tanh. Use the sum for h0
  and each token as before. Heads/losses are unchanged. Total43073, increment0.
- IMPORTANT: `W[x;p;t]+b = Wb*x+b+Wp*p+Wt*t`. This split is algebraically the
  same function class, not a mechanism that removes interference. It changes
  parameter organization and floating-point operation order. Report any training
  differences as numerical/optimization observations, not added expressivity.
- Construct common seed7 initial model plus seed20 heads/expanded projection in
  exactly the Issue20 order, then copy its three column blocks into B. No new
  random draws affect weights or caller RNG. Verify common parameters and blocks.
- Train B with the same seed7 schedule600,batch16,Adam .003,clip1,CPU1thread;
  existing four losses1 plus head CE1 each. No slot byte CE or weight/seed search.
- Train450/validation150 only, no test access for fitting/evaluation. Save5x5
  participant-by-time support tables with train-only class order. Define seen
  strictly by positive train pair count. Empty groups get null metrics, not zero.

## Fixed diagnostic matrix

- For BOTH A/B: predicted, participant-head gold, time-head gold, both-head gold,
  participant-head zero/permutation, time-head zero/permutation. Head interventions
  hold all old33 concepts and the other head fixed. Zero input to a bias-free
  branch exactly removes that branch. Seed17 global row permutation, no training
  intervention. Save8 conditions per model.
- Also group A's historical predicted/head_gold/full_oracle outputs by seen pair.
  Full oracle repairs old concepts too; do not conflate it with head-only gold.
- All/seen/unseen groups: LM, dedicated head accuracy/balanced accuracy, reference-
  history slot byte NLL/rank/argmax, free EOS/UTF8/unique/slot/both-slot/frame counts.
  Generation uses BOS and self-history only; authored labels enter scoring afterward.
  Both-slots success requires both parsed slots to equal that row's gold pair.
  Unparseable rows stay in the denominator. Provided gold is explicit oracle only.
- Each intervention: fixed-reference logit L1/KL(base||intervention)/argmax change
  at slot start-1 and inside both slots, with same-slot and cross-slot results.
- Factorial diagnostic for both models: all25 dedicated one-hot participant×time
  assignments per validation source, retaining old predicted concepts. A single
  5x5 matrix provides fixed-participant/vary-time and fixed-time/vary-participant
  series. Save raw generated tokens/parsed slots and assignment-match counts.
  These counterfactual assignments are NOT ordinary gold/reference performance.
- Factorial logit sensitivity holds the ORIGINAL authored prior history fixed:
  compare(p,t) to(p,0) for time changes, and to(0,t) for participant changes.
  Save aggregate start-1/inside effects on BOTH slots (reference-class0 anchors,
  fixed before measurement). No post-result choice of pairs/anchors or split edits.
- Save B weights-only checkpoint; exact state/head/validation logits/greedy replay.
  Tests cover algebraic equivalence, branch-only changes, gradients, future
  causality, pair denominators, both-slot conjunction and checkpoint integrity.

Optional orthogonal/local injection is deferred to another Issue. No paid GPU,
external/private data, test tuning, auto merge/close or worker restart. Single-seed
AI-authored toy observations cannot establish general semantic understanding.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_joint_slots --baseline-report codex/work_output/issue20-seed7-v1/report.json --out-dir codex/work_output/issue22-seed7-v1
```
