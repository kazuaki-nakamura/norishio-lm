# Slot-gradient isolation (Issue #32)

Pre-measurement plan from PR31 merge `29533877a11850bb801f54f28296415797d10f80`.

G0 (`end_to_end`) is the current D training path. G1 (`stop_slot_lm`) uses the
same model, parameter count, initial state, schedule, losses and optimizer, but
detaches only the dedicated participant/time softmax distributions on the
decoder input branch. The original logits remain connected to the two slot CE
losses. Base21 LM gradients, decoder updates, encoder sharing and global clip
remain. No head is removed, no gold/zero/hard replacement is used.

Each seed7/17/29 has one independently constructed initial state and schedule;
G0/G1 receive copies. Each runs 600 updates, batch16, Adam .003, global clip1,
CPU one thread, deterministic algorithms. Existing four losses and two slot CE
losses are weight1. Parameter count is fixed42689. Test is not evaluated and
validation labels never select rows or settings.

At updates0/100/300/600, a fixed train-only 16-row probe (Generator seed3200)
records LM-only and per-head CE-only gradient norms, dot and cosine for head
weight+bias. G1 LM gradients to dedicated heads must be zero/None while CE
gradients remain. The probe uses `autograd.grad`, does not update weights,
consume RNG, or leave `.grad` state. Each update records clip pre-norm and
applied coefficient; routing may change the global coefficient and other
updates, so this is not a head-only intervention.

Evaluate both arms with source-only predicted, participant-gold, time-gold and
both-gold named oracle conditions. Record head joint/calibration, LM, EOS,
UTF8, unique, parse and slot/full-frame scores, byte-v2, and seed arithmetic
means. Reproduce historical D G0 where the environment permits and report any
difference without editing old reports. Save new routing metadata checkpoints;
old checkpoints remain compatible as implicit `end_to_end`.

This experiment tests one gradient route only. It cannot establish that the
head is the sole cause of failure or prove general semantic understanding.

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_gradient_routing --baseline-report codex/work_output/issue26-fixed-v1/report.json --historical-report codex/work_output/issue24-seed7-v1/report.json --out-dir codex/work_output/issue32-gradient-v1
```
