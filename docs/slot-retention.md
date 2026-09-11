# Issue 14: participant/time retention diagnosis

Plan fixed before measurement, 2026-09-11. Diagnose the exact per_step_additive
checkpoint from Issue12: seed7,600 updates,batch16,Adam .003,clip1,CPU1thread.
Reuse weights without optimizer updates or vocabulary refitting. Require its file
SHA to match the prior report, then reproduce that report's predicted validation
LM, generated examples and concept metrics. No additional seeds or model selection.
The checkpoint and report are inputs, not committed training artifacts.

Use unchanged v1 validation150 only; no test evaluation. Keep the existing
initial-only comparisons and all corpus/grammar/scoring rules unchanged.

- Save all seven concept-head support/confusion/entropy/probabilities/metrics,
  identifying participant/time separately from the other five fields.
- Compare predicted, participant-only gold, time-only gold, both-only gold and
  all-seven gold. Replace only the named probability groups. Gold is always an
  explicitly labeled oracle argument and never an ordinary source feature.
- Derive participant/time spans by rendering the authored target template from
  literal pieces and slot values, accumulating UTF-8 byte lengths. Full rendered
  target must match the reference; do not search for value substrings.
- Teacher-forced diagnostic input is BOS plus prior reference bytes. Logit
  position j predicts raw reference byte j (token ID byte+4); EOS is at byte
  length, outside both half-open spans. Recurrent causality prevents later input
  bytes from influencing an earlier position. Test future-token non-interference.
- For both spans, save each byte's correct probability/NLL/rank/argmax and
  intervention sensitivity logit L1/KL(base||intervention). Aggregate by actual
  bytes with explicit row/byte counts. Rank is 1 plus strictly greater logits
  (ties at the same maximum therefore have rank1).
- Free generation is separately BOS-started using only its own output history.
  Save all raw outputs, EOS/UTF-8/unique counts and conservative mechanical slot
  scores with all-observed-row denominators. Reference appears only in scoring.
- The optional gold-prefix intervention is deferred. The required fixed-reference
  versus self-history comparison alone cannot causally isolate history propagation;
  retain that limitation rather than labeling the failure conclusively.
- Verify unchanged model state through diagnostics and same-checkpoint replay.
  New outputs must go into a new ignored directory and never replace old results.

High teacher-forced byte accuracy may simply exploit correct preceding bytes;
it does not prove that the slot concept is used. Small single-slot oracle changes
can implicate conditioning but are not unique-cause proof. Full target-grammar
parsing is a narrow mechanical score, not a general natural-language evaluator.
No paid compute, external/private data, automatic merge/close or worker restart.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_slot_retention --checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --baseline-report codex/work_output/issue12-seed7-v1/report.json --out-dir codex/work_output/issue14-seed7-v1
```

Measured values, commands, hashes and limitations belong in `docs/handoff.md`.
