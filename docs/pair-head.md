# Joint pair head control (Issue #28)

Pre-measurement plan from PR27 merge `1c34f73c27a9652b657834f42fc2fef50f6e230f`.

## Fixed comparison

A replays the saved D models for seeds7/17/29 and every recorded Issue26
validation head/predicted/both-gold score. Authenticate reports and checkpoints;
do not retrain A. B starts from each original common ToyModel initialization,
with the unchanged D decoder, consumed-prefix gate and retained21 base inputs.
Add Linear(32,25),825 parameters, total43514 versus A42689. Initialize this
head with isolated seed28; D extension seed20 remains shared across all seeds.
Class index is `(participant_train_vocab_id-1)*5 + time_train_vocab_id-1`.
The Cartesian product is train-derived, including ten pairs without CE positives.

B uses softmax25 reshaped5x5, row/column sums for participant5/time5, and passes
only these marginals to the existing local injection. Independent heads stay
for diagnostics and keep their two CE losses, but do not also enter decoder.
Keep LM/concept/sense/sememe losses each1; independent head CE each1; pair CE1.
All components train jointly. Thus capacity, objective and input-distribution
changes are bundled; this is not a pure capacity-matched objective ablation.

Seeds7/17/29:600 updates each,batch16,Adam .003,clip1,CPU1thread, original
seed-specific sampling rule. No repeated seed selection, weights/temperature
search, factorized alternative, data/split changes or test evaluation.
Fit vocabulary/tensorizer and pair support only on train450. Validation150
contains five unseen pairs; absent seen-validation metrics are null.

## Measurements

- Independent heads and joint-head marginals: existing head_joint_metrics,
  accuracy/balanced, raw Brier/NLL/ECE and same-row joint exact.
- Actual25-way pair distribution: accuracy, Brier summed over25 then row mean,
  NLL floor1e-12,10 equal-width confidence bins, gold probability and rank.
  Rank breaks probability ties by ascending Cartesian index,1-based.
  Separate train diagnostics and validation; train-seen/unseen groups retain
  empty denominators as null. No learned calibration.
- Validation BOS free generation with predicted marginals, participant gold,
  time gold and both gold. Report LM/EOS/UTF8/unique/p/t/both/fullframe. Oracle
  labels affect only named interventions; normal inference is source-only.
- All25 factorial assignments, scored against assigned counterfactual slots,
  with train-seen/validation-unseen/neither labels. These are not normal accuracy.
- All seed values and unweighted3seed arithmetic means; no best-seed selection.
- Save immutable checkpoints and verify exact state/pair/marginal/logits/greedy
  reload. Preserve actual failures; n3 and common extension RNGs limit inference.

The Issue26 scorer currently returns null teacher_forced diagnostics; this
experiment preserves historical replay and does not claim those missing byte
diagnostics were measured. LM remains measured using reference past history;
free generation uses only its own past tokens and unchanged causal gate.

Run full pytest, both demos, knowledge/OKF/MCP checks. Keep generated artifacts
ignored. No paid GPU, private data, auto merge/close or worker restart.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_pair_head --baseline-report codex/work_output/issue26-fixed-v1/report.json --out-dir codex/work_output/issue28-fixed-v1
```
