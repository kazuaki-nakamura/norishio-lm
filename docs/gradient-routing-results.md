# Issue #32 gradient-routing results

This record compares two training graphs in the same CPU-only toy experiment:
G0 `end_to_end` sends the decoder language-model loss through the dedicated participant/time slot probabilities; G1 `stop_slot_lm` detaches only that decoder path while retaining the slot cross-entropy path. The model, initialization, schedule, losses, optimizer, clipping, data, and probe rows are otherwise shared.

The result is a routing diagnostic, not evidence that the authored toy dictionary was learned. The validation split contains 150 unseen pairs and zero seen pairs; the test split was not evaluated.

## Reproducibility

| seed | updates | batch | parameters | G0 baseline evaluation | G0 final state | G0 seconds | G1 seconds |
|---:|---:|---:|---:|:---:|:---:|---:|---:|
| 7 | 600 | 16 | 42689 | True | True | 45.71 | 45.57 |
| 17 | 600 | 16 | 42689 | True | True | 46.95 | 45.62 |
| 29 | 600 | 16 | 42689 | True | True | 46.69 | 45.81 |

The G0 evaluation and final state digest match the Issue26 fixed baseline for all three seeds. The checked historical Issue24 report was loaded as an input; its SHA256 and version are recorded in the JSON report.

## Generation and slot metrics

Each value below is measured on the 150-row unseen validation set. `predicted` uses the model's own slot probabilities; `both_gold` supplies both participant and time gold distributions as an explicit intervention. `both-slot` is exact joint slot accuracy; `fullframe` is exactness of the complete parsed frame.

| seed | routing | predicted both-slot | both-gold both-slot | both-gold fullframe | both-gold exact text | predicted LM loss | both-gold LM loss |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 7 | `end_to_end` | 0.0% | 32.0% | 10.7% | 10.7% | 0.164826 | 0.104085 |
| 7 | `stop_slot_lm` | 0.0% | 40.0% | 14.7% | 14.7% | 0.164117 | 0.103057 |
| 17 | `end_to_end` | 0.0% | 30.7% | 8.0% | 8.0% | 0.190100 | 0.151850 |
| 17 | `stop_slot_lm` | 0.7% | 33.3% | 10.0% | 10.0% | 0.183370 | 0.145266 |
| 29 | `end_to_end` | 0.0% | 42.7% | 13.3% | 13.3% | 0.181583 | 0.113778 |
| 29 | `stop_slot_lm` | 0.0% | 44.7% | 14.7% | 14.7% | 0.183759 | 0.113354 |

In the ordinary predicted condition, G0 is 0/150 for every seed; G1 is seed7=0/150, seed17=1/150, and seed29=0/150, or 1/450 in aggregate. The isolated success and the small gold-intervention changes do not establish a generalization improvement; the comparison is underpowered and remains confined to this fixed grammar.

## Gradient probes

The probe uses the same 16 train rows at updates 0, 100, 300, and 600. Norms are over each dedicated head's weight and bias. A zero LM norm in G1 is the intended isolation check; CE norms remain nonzero.

| seed | routing | step | participant LM norm | participant CE norm | time LM norm | time CE norm | participant cosine | time cosine |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 7 | `end_to_end` | 0 | 0.000318 | 0.260952 | 0.000383 | 0.353349 | -0.145012 | -0.020841 |
| 7 | `end_to_end` | 100 | 0.000824 | 0.534580 | 0.000717 | 0.762342 | -0.654921 | 0.405286 |
| 7 | `end_to_end` | 300 | 0.000414 | 0.673306 | 0.001435 | 0.599429 | 0.212262 | 0.500732 |
| 7 | `end_to_end` | 600 | 0.002269 | 0.240248 | 0.001209 | 0.207556 | 0.807651 | 0.774701 |
| 7 | `stop_slot_lm` | 0 | 0.000000 | 0.260952 | 0.000000 | 0.353349 | — | — |
| 7 | `stop_slot_lm` | 100 | 0.000000 | 0.534371 | 0.000000 | 0.762534 | — | — |
| 7 | `stop_slot_lm` | 300 | 0.000000 | 0.673208 | 0.000000 | 0.599577 | — | — |
| 7 | `stop_slot_lm` | 600 | 0.000000 | 0.242319 | 0.000000 | 0.209402 | — | — |
| 17 | `end_to_end` | 0 | 0.000189 | 0.273080 | 0.000363 | 0.378711 | 0.312408 | 0.196948 |
| 17 | `end_to_end` | 100 | 0.000350 | 0.430084 | 0.000501 | 0.618984 | 0.151242 | 0.629256 |
| 17 | `end_to_end` | 300 | 0.000721 | 0.524830 | 0.000409 | 0.511094 | 0.413661 | 0.555239 |
| 17 | `end_to_end` | 600 | 0.004319 | 0.171312 | 0.002210 | 0.160413 | 0.775453 | 0.695215 |
| 17 | `stop_slot_lm` | 0 | 0.000000 | 0.273080 | 0.000000 | 0.378711 | — | — |
| 17 | `stop_slot_lm` | 100 | 0.000000 | 0.430042 | 0.000000 | 0.619229 | — | — |
| 17 | `stop_slot_lm` | 300 | 0.000000 | 0.525100 | 0.000000 | 0.511183 | — | — |
| 17 | `stop_slot_lm` | 600 | 0.000000 | 0.172059 | 0.000000 | 0.160549 | — | — |
| 29 | `end_to_end` | 0 | 0.000145 | 0.267023 | 0.000086 | 0.375277 | -0.291628 | -0.236559 |
| 29 | `end_to_end` | 100 | 0.000397 | 0.427845 | 0.000609 | 0.615998 | 0.579752 | -0.535552 |
| 29 | `end_to_end` | 300 | 0.000489 | 0.536722 | 0.005248 | 0.361417 | 0.205978 | 0.355945 |
| 29 | `end_to_end` | 600 | 0.004937 | 0.173323 | 0.000941 | 0.103349 | 0.753280 | 0.198447 |
| 29 | `stop_slot_lm` | 0 | 0.000000 | 0.267023 | 0.000000 | 0.375277 | — | — |
| 29 | `stop_slot_lm` | 100 | 0.000000 | 0.428057 | 0.000000 | 0.615535 | — | — |
| 29 | `stop_slot_lm` | 300 | 0.000000 | 0.536591 | 0.000000 | 0.361676 | — | — |
| 29 | `stop_slot_lm` | 600 | 0.000000 | 0.173460 | 0.000000 | 0.103785 | — | — |

G0 has nonzero LM and CE gradients at all probes. G1 has exact-zero LM gradients for both dedicated heads at all probes, while the CE gradients remain nonzero. The decoder/base-model gradients are covered by the focused routing tests.

## Byte-v2 and checkpoint checks

Teacher-forced byte-v2 diagnostics were run for `predicted`, `participant_gold`, `time_gold`, and `both_gold` for every arm. They are reference-history diagnostics and are not free-generation scores or intervention claims. Every saved G0/G1 checkpoint reloaded with state, old concept distributions, slot heads, and decoder logits equal to the in-memory model.

| routing | condition | mean participant byte accuracy | mean time byte accuracy |
|:---|:---|---:|---:|
| `end_to_end` | `predicted` | 84.0% | 90.0% |
| `end_to_end` | `participant_gold` | 91.3% | 90.0% |
| `end_to_end` | `time_gold` | 83.3% | 99.9% |
| `end_to_end` | `both_gold` | 91.1% | 99.9% |
| `stop_slot_lm` | `predicted` | 84.0% | 89.9% |
| `stop_slot_lm` | `participant_gold` | 91.5% | 89.9% |
| `stop_slot_lm` | `time_gold` | 83.5% | 99.9% |
| `stop_slot_lm` | `both_gold` | 91.1% | 99.9% |

## Limits and next step

- The experiment uses the fixed `norishio-toy-1.0` corpus, three seeds, 600 updates, one CPU thread, and 42,689 parameters.
- It does not evaluate the test split, larger data, a trained language model, or free-generation semantic quality.
- Global gradient clipping is retained; routing changes can therefore alter the shared encoder/base update through the total norm even when the direct LM-to-slot path is detached.
- The next useful experiment is a preregistered larger compositional split with the same routing controls and an explicit ablation of shared-encoder effects.

Report source: `codex/work_output/issue32-gradient-v2/report.json`.
