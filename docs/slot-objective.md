# Slot-weighted decoder objective (Issue #18)

Pre-measurement plan from PR #17 merge
`545b81fd035182376f44cdbdc5dad50cae08a180`.

## Fixed comparison

- A: unchanged per_step_additive baseline, NO new slot objective.
  The Issue says "LM-only", but the current baseline already includes the
  existing concept/sense/sememe auxiliary losses. Preserve all four existing
  weights at1; do not silently remove them or relabel A as literally LM-only.
- B: same model/initial parameters and four existing losses, plus
  **1.0 * mean CE over the union of participant/time target byte positions**.
  Both fields have equal per-byte weight, not separate per-example averages.
  Each target slot has6 UTF-8 bytes in v1; batch16 contributes192 selected bytes.
  Use existing authored template spans; reject overlap and non-byte labels.
  EOS/padding/non-slot positions are excluded from the added loss. The ordinary
  LM loss still covers all target bytes and EOS, so these slot bytes receive
  additional supervision. No teacher/reference feature enters the concept input.
- Architecture/parameter increment0. Optional C explicit slot head is deferred
  to keep this comparison about the objective only. No weight search or selection.
- Fit tensorizer and concept vocabulary on train450 only. Evaluate validation150;
  no test evaluation, fitting, selection or tuning. Original v1 and scorers unchanged.
- Seed7,600 updates,batch16,Adam .003,clip1,CPU1thread, deterministic algorithms.
  Same initial state and fixed torch.randint schedule for both. New B starts from
  the shared initial state, NOT the trained A checkpoint. Generation cap128.
- Require initial SHA256
  `f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`
  and sampling SHA256
  `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`.
  A must reproduce the saved Issue #12 per-step state and normal generated
  examples/LM/slots exactly. Preserve existing artifacts; halt if replay differs.

## Fixed evaluation

- For both arms use predicted, participant-only oracle, time-only oracle, and
  full-seven oracle. Oracle conditions are explicit diagnostics, not normal scores.
- Record all-target validation LM, per-slot teacher-forced byte NLL/probability/
  rank/argmax using existing span_metrics. Byte positions j map to label j;
  +4 is a token ID offset only. Mark correct reference history explicitly.
- BOS-started self-running generation uses existing score_condition/score_slots:
  EOS/UTF-8/unique output, each slot/full frame, coverage, all-row denominators,
  and conditional parse-only values. Unparseable rows remain failures.
- Record all concept-head class/balanced metrics, especially participant/time,
  separately from decoder byte and generated-slot metrics.
- Save both checkpoints; require exact state/concepts/validation logits/greedy
  replay on reload. Record hashes, parameter changes, loss traces, training time,
  and selected-byte denominators. All generated artifacts stay ignored locally.
- Tests: slot gradients reach GRU/output head, non-slot logits receive no direct
  slot loss gradient, future inputs cannot affect earlier logits, labels only
  affect loss, strict source-only boundary, zero weight matches old training.

No extra seeds, paid compute, external/private data, modified test sets or
post-result weight changes. No automatic merge/close or worker restart.
Lower teacher-forced loss alone is not semantic retention. Both success and
failure remain toy single-seed observations, not evidence for extra semantic layers.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_slot_objective --baseline-report codex/work_output/issue12-seed7-v1/report.json --baseline-checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --out-dir codex/work_output/issue18-seed7-v1
```
