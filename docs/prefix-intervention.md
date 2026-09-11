# Frozen gold-prefix diagnostic (Issue #16)

Plan fixed before measurement, based on PR #15 merge
`631bf70e263218ac1531e75c05ae99e2fdffa811`.

- Reuse Issue #12 seed7/600 per-step checkpoint, SHA256
  `0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057`.
  Require Issue #14 report SHA256
  `a4d18bb7fab3d63a94d0d0e5b026b59956aae6db276241eba60b248e03621c92`.
  Replay the complete prior diagnostic (including normal generation and metrics)
  before prefix measurements. Compare exact JSON values. No training/refit/test.
- CPU, one thread, deterministic algorithms, validation150, total output cap128
  tokens including supplied prefix (BOS excluded, EOS included). No sampling.
- Six fixed boundaries: for participant and time separately, `start-1`, `start`,
  `end` in authored UTF-8 bytes. These mean one byte before slot start, immediately
  before slot start, immediately after the slot. Do not round to character edges:
  a prefix may end mid-codepoint; the byte decoder must complete it.
- Cross every boundary with predicted, participant-only gold, time-only gold,
  and full-seven gold concept probabilities. All are explicit prefix-oracle
  diagnostics, never ordinary source-only performance. Other fields untouched.
- Prefix input is exactly BOS + reference byte IDs `[0:boundary]`. Generate after
  the boundary using only emitted history. Generation API has no reference or
  targets argument; evaluation reads gold only after generation. Future gold
  continuation is unavailable to the generation API.
- Save each generated token, prefix length, and post-switch correct-byte
  probability/NLL/rank/argmax. Score both slots at fixed authored positions.
  Full-slot success is eligible only if boundary <= slot start. If partly supplied,
  score remaining bytes separately; if wholly supplied, no slot success credit.
  Include all150 rows, eligibility counts, missing decisions after EOS/cap,
  expected/evaluated byte counts, and exact entire post-prefix suffix+EOS success.
  NLL/probability/rank means only exist for evaluated decisions; missing decisions
  remain in coverage and expected-byte accuracy denominators, never fabricated NLL.
- Keep full reconstructed-text template scores separate: supplied tokens can
  improve those scores, so they are not post-intervention success. The main
  evidence is generated-only metrics, including failures and all-row denominators.
- Optional counterfactual prefixes are deferred: equal byte length alone does not
  match grammar position/content, and some pre-time prefixes are identical. This
  run cannot distinguish content repair from effects of supplied history length.
  No post-result boundary selection, soft interpolation, or extra condition.
- Hash model state before/after and retain prior output artifacts. Use a fresh
  ignored output directory, not an overwrite. Tests cover prefix position/ID
  offset, EOS/cap, self-history switch, and provided-slot exclusion.

Improvement would support a history/conditioning interaction candidate, not a
unique mechanism. One-hot distribution shift, byte overlap, authored templates,
single seed, and conservative fixed-position scoring limit interpretation.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_prefix_experiment --checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --baseline-report codex/work_output/issue12-seed7-v1/report.json --slot-report codex/work_output/issue14-seed7-v2/report.json --out-dir codex/work_output/issue16-seed7-v1
```
