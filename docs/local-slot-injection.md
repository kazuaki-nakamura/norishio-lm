# Duplicate removal and causal local injection (Issue #24)

Pre-measurement plan from PR23 merge `615fc29db1dc0ccef21ab9d4e1c044005e582e16`.

## Arms and initialization

- A: saved Issue20 shared C. Require report SHA256
  `66e8c507dc5485d5e01825882688aeb8e34f0d6cbbbba36b0ce14d4960761ad8`.
- B: saved Issue22 equivalent split projection. Require report SHA256
  `a303e74fab83ccc11b4350579988b2e88e2077ba44abc263f32549890de2be39`.
  Require each report's checkpoint file hash and state digest, replay complete
  Issue22 A/B eight-condition and 25-pair evaluations exactly. A also replays
  Issue20 seven-condition evaluation and sensitivity. No A/B retraining.
- C: old participant/time groups (six dimensions each including UNK) never
  reach decoder conditioning. Keep the other21 old dimensions plus p5/t5 heads.
  The external transport remains43 for diagnostic API compatibility; only31
  selected values reach projection. Old bottleneck outputs/losses remain.
  Base Linear(21,32,bias) plus participant/time Linear(5,32,no bias), summed
  before tanh for h0 and every input token, as in B.
- D: same C weights/parameters; h0 is tanh(base) only. At each token input,
  add tanh(base + participant_gate*participant_projection + time_gate*time_projection).
  Gates follow the already consumed byte prefix, below. Recurrent propagation
  can carry an earlier branch's effect forward; direct zero is not total causal zero.
- Build seed7 common ToyModel and seed20 ExplicitSlotModel in historical order.
  Copy all shared weights, base retained columns and bias, both head columns.
  C/D initial state values must match exactly; no fresh random weight values.
  Each has42689 parameters,384 fewer than A/B43073. Therefore A/B→C changes
  input information and parameter count; C→D keeps capacity but changes timing/h0.

## Causal gate rule fixed before measurement

Known authored output grammar begins with literal `私は、` (9 UTF8 bytes) or
`私が` (6 UTF8 bytes), followed by time (6 bytes), then participant (6 bytes).
These constants are a versioned grammar prior, not read from each reference.
Both prefixes and all five slot values are available in train grammar.

At decoder position j the consumed input is BOS followed by exactly j output
tokens. Recognize a prefix only once all its literal bytes have already been
consumed. Let n be consumed raw bytes after that prefix:

- time_gate=1 for 0<=n<6;
- participant_gate=1 for 6<=n<12;
- otherwise both0 (including incomplete/divergent prefix).

An invalid special token in consumed history disables gates. Never search for
a later prefix or inspect slot byte contents to choose gates. Teacher forcing
uses only the prefix through j, free generation uses the same rule on its own
emitted prefix. No source-dependent position, target length, gold boundaries,
future byte, or reference text enters generation. Failed/misaligned generated
grammar is allowed to fail and is scored normally. No grammar output masking.

## Fixed training and diagnostics

Train450/validation150, all validation unseen pair, no split/test changes.
Seed7 schedule600×16, Adam .003, clip1, CPU1thread; original four losses1 each
plus participant/time dedicated CE1 each, no slot byte CE. No architecture,
weight or seed search. Train C/D once each from common initial weights.

Use Issue22's fixed eight head conditions, all/seen/unseen metrics and25
factorial pair assignments for A/B/C/D. Old concepts stay predicted for new-arm
head intervention. Empty seen group rates null. Report ordinary reference
metrics separately from counterfactual assigned-pair following. Teacher-forced
byte/logit diagnostics use authored past history and are not free generation.
Cross-slot L1/KL/argmax at start-1/inside and all per-position evidence retained.

Checkpoint both C/D with strict weights-only CPU metadata and exact state,
old bottleneck/head probabilities, validation logits and greedy replay.
Regression tests: duplicate groups disconnected, gates zero outside allowed
steps, both prefix forms, malformed prefixes/specials, future/label invariance,
gradient paths, generation consistency, checkpoint mode/rule/metadata checks.

Report failures unchanged. D's grammar prior and C's removed384 parameters are
confounds, not proof of semantic understanding. Generalization outside the
authored grammar, multiple seeds and semantic-layer ablation remain open.
No paid GPU, external/private data, automation restart, auto merge/close.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_local_slots --baseline-report codex/work_output/issue22-seed7-v1/report.json --historical-report codex/work_output/issue20-seed7-v1/report.json --out-dir codex/work_output/issue24-seed7-v1
```
