# Head prediction versus decoder following (Issue #26)

Pre-measurement plan from merge `3880c85555f876d05d7ef481a5efde7ac73258d8`.

- Keep Issue24 D architecture, causal consumed-prefix gates,42689 parameters,
  losses (original four1 + dedicated head CE1 each), data and split unchanged.
- Seed7 is saved D, not retrained. Require Issue24 report SHA256
  `eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8`,
  its D checkpoint hash/state, and complete eight-condition/25-factorial replay.
- Additional seeds17,29 control common ToyModel initialization and batch schedule.
  The historical extension RNG seed20 stays fixed for dedicated heads/new slot
  projection columns in all runs. Report this shared initialization component.
  Train each additional seed once:600 updates,batch16,Adam .003,clip1,CPU1thread.
  No gate/architecture/weight/seed selection after results. Test not evaluated.

## Head metrics

Use source-only predicted participant5/time5 distributions, train-fit vocabulary.
Per head: accuracy, balanced accuracy, confusion, multiclass Brier (sum over
five classes then mean over rows), raw NLL (probability floor1e-12),10 equal-width
confidence ECE bins [0,.1),...,[.9,1]. No calibration fitting or temperature tuning.
Joint exact means both argmax labels correct on the same row. For all25 gold
pairs record both correct / participant only / time only / both wrong; absent
cells have counts0 and ratesnull. Error covariance and Pearson/phi correlation
use binary error indicators; zero variance gives null correlation.
Joint gold log-probability is log p_gold + log t_gold; entropy is Hp+Ht;
margin is top1 minus top2 probability of the25-way outer product. This is a
conditional-independence approximation, not a learned joint distribution.
Save per-row probabilities/labels and these quantities for reproducibility.

## Decoder diagnosis (frozen seed7)

- Separate predicted, participant gold, time gold, both gold by the validation
  gold pair; reuse the historical generation results. Report participant/time/
  both/fullframe with all-row denominators including parse failures.
- Label the25 factorial assigned pairs as train-seen, validation-unseen, or
  neither observed in train nor validation. No test rows inspected. Assigned
  pair following is explicitly counterfactual, not ordinary reference accuracy.
- Define confident train subsets ONLY from train-source predictions and train
  labels: BOTH head maxima >=0.8, then jointly correct versus at least one wrong.
  Evaluate each subset's own train rows with predicted distributions and its own
  retained base. These are selected training diagnostics, not validation subsets
  or unbiased/generalization performance. Empty subsets get null metrics; do
  not lower the threshold, replace them with gold, or pick validation examples.

## Semantic base ablation (frozen seed7, inference only)

Compare full retained21 input, all21 zero, event group zero, operators group zero.
Apply each with predicted heads and both-gold heads:8 conditions. Zero semantic
input probabilities without renormalization; constant base projection bias stays.
Old participant/time groups remain disconnected in every case. No weight or
head changes, no retraining, no source latent bypass. These are old-concept
group interventions, not evidence for the whole upstream semantic-layer stack.
Report LM, head/generation separation, byte diagnostics and pair generation
scores; distribution shift from zero inputs remains a limitation.

## Three-seed report and validation

For7/17/29 report uncalibrated head metrics plus predicted/both-gold LM, EOS,
UTF8, unique outputs, participant/time/both/fullframe. Give all individual
values and unweighted arithmetic means (n=3), without selecting a winner.
Save new-seed checkpoints and require exact state/head/logits/greedy reload.
Run leakage/causality and new metric tests, full pytest, both demos, OKF/knowledge
and MCP checks. Preserve failures and immutable ignored outputs. No paid GPU,
external/private data, worker restart, auto merge/close or general meaning claim.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_head_following --baseline-report codex/work_output/issue24-seed7-v1/report.json --out-dir codex/work_output/issue26-fixed-v1
```
