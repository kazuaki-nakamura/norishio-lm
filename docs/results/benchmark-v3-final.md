# Benchmark v3 factor-path final confirmation

The explicitly authorized one-shot `final-confirmation` evaluation completed
from the 18 frozen checkpoints: 384 authored rows, 18 evaluations, and 0
failed runs. The exclusive invocation marker was consumed once. A second call
was rejected with `FileExistsError` (exit code 1) before final rows could be
evaluated again.

The raw ignored result has SHA-256 `eb94b2b35d9de0c575b90342d517582e2304c766b3d8bb385ea22f2ac8c56a59`. Its tracked attestation
matches that digest and binds benchmark `930958ca1002b9f566fb13d28e072e99fa8ba5c0493308526586dcc128302d84`
and terminal records `f31a87807b5e9859df07ab8ad22a5f40a7f4188fdd492dfb43036d85185d8fac`. The compact
tracked summary is
[`benchmark-v3-final-summary.json`](benchmark-v3-final-summary.json).

## Frozen selection result

| Rank | Arm | Free | Triple | Pair | Balanced | Exact text | TF byte | Parse coverage |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | H1L1 | 0.224826 | 0.226562 | 0.231771 | 0.352431 | 0.224826 | 0.964796 | 0.408854 |
| 2 | H0L1 | 0.190104 | 0.197049 | 0.203993 | 0.324002 | 0.190104 | 0.960579 | 0.388889 |
| 3 | H1L0 | 0.181424 | 0.203993 | 0.217014 | 0.360460 | 0.181424 | 0.960625 | 0.449653 |
| 4 | H0L0 | 0.160590 | 0.174479 | 0.184896 | 0.333550 | 0.160590 | 0.958379 | 0.421007 |
| 5 | D_AUX | 0.006076 | 0.026910 | 0.092882 | 0.170790 | 0.006076 | 0.914329 | 0.496528 |
| 6 | NO_INPUT | 0.001736 | 0.013889 | 0.041667 | 0.208333 | 0.001736 | 0.899500 | 1.000000 |

The frozen primary metric ranks `H1L1 > H0L1 > H1L0 > H0L0 > D_AUX >
NO_INPUT`. No tie-breaker changed the order. `TF byte` is teacher-forced and
is not a free-generation result.

## Preregistered 2x2 effects

| Metric | Activation | Locality | Interaction |
| --- | ---: | ---: | ---: |
| Free generation exact | +0.027778 | +0.036458 | +0.013889 |
| Triple exact | +0.029514 | +0.022569 | +0.000000 |
| Pair exact | +0.029948 | +0.016927 | -0.004340 |
| Mean balanced atomic | +0.027669 | -0.008789 | +0.001519 |
| Exact target text | +0.027778 | +0.036458 | +0.013889 |

These are descriptive effects over three seeds, not significance estimates.

## Support-group split

`seen_pair` below means the participant-time pair was seen in training while
the full triple was unseen. Each group has 192 rows per seed.

| Arm | Group | Free | Triple | Pair | Balanced |
| --- | --- | ---: | ---: | ---: | ---: |
| H0L0 | unseen_pair | 0.053819 | 0.055556 | 0.055556 | 0.291667 |
| H0L0 | seen_pair/unseen_triple | 0.267361 | 0.293403 | 0.314236 | 0.375434 |
| H0L1 | unseen_pair | 0.074653 | 0.076389 | 0.076389 | 0.276042 |
| H0L1 | seen_pair/unseen_triple | 0.305556 | 0.317708 | 0.331597 | 0.371962 |
| H1L0 | unseen_pair | 0.062500 | 0.067708 | 0.067708 | 0.307292 |
| H1L0 | seen_pair/unseen_triple | 0.300347 | 0.340278 | 0.366319 | 0.413628 |
| H1L1 | unseen_pair | 0.112847 | 0.112847 | 0.112847 | 0.308594 |
| H1L1 | seen_pair/unseen_triple | 0.336806 | 0.340278 | 0.350694 | 0.396267 |
| D_AUX | unseen_pair | 0.001736 | 0.010417 | 0.043403 | 0.133247 |
| D_AUX | seen_pair/unseen_triple | 0.010417 | 0.043403 | 0.142361 | 0.208333 |
| NO_INPUT | unseen_pair | 0.001736 | 0.013889 | 0.055556 | 0.208333 |
| NO_INPUT | seen_pair/unseen_triple | 0.001736 | 0.013889 | 0.027778 | 0.208333 |

All source-conditioned arms retain a large gap between unseen pairs and
pair-known/unseen-triple rows. This limits the result to the fixed fixture and
does not demonstrate broad compositional generalization.

## Intervention and interpretation limits

The fixed artificial one-hot intervention changed the decoded target factor in
0/48 main-arm probes (four probes per seed and 12 per arm). It
therefore supplies no positive evidence of factor-level causal control. The
probing sample is deliberately small and its rates are coarse.

The four intervention pairs are chosen after opening the final split by matching
gold target frames. Gold values select diagnostic pairs only and are never model
inputs. These target-dependent probes are descriptive diagnostics and are not
selection evidence.

The development and final rankings agree, but the final values are held-out
observations rather than a new model-selection round. The benchmark vocabulary,
templates, and records are authored fixtures. These results concern a tiny CPU
model under one frozen protocol; they do not establish general language-model
quality, modern lexical meaning, sememe validity, glyph or etymological meaning,
or semantic-layer effectiveness outside this ablation.

Checkpoint authentication binds config, benchmark, schedule, parameter count,
initial/final state, and checkpoint-file hashes. It does not embed a source hash
for the runner or optimizer implementation, so the recorded execution commit is
the external implementation boundary rather than a cryptographic training
transcript.

## Protocol erratum (Issue #39)

The implementation used `all.free_generation_exact.accuracy` for the table and
ranking above. The preregistration instead names
`all.generation_frame_exact.accuracy` as primary. A read-only audit of the
attested result, whose SHA-256 was independently confirmed as
`eb94b2b35d9de0c575b90342d517582e2304c766b3d8bb385ea22f2ac8c56a59`,
reconstructed all 18 arm/seed values without loading checkpoints, opening the
holdout again, or running inference.

| Arm | Generation frame exact |
| --- | ---: |
| H1L1 | 0.226562 |
| H0L1 | 0.190972 |
| H1L0 | 0.183160 |
| H0L0 | 0.162326 |
| D_AUX | 0.006076 |
| NO_INPUT | 0.003472 |

The preregistered ranking remains `H1L1 > H0L1 > H1L0 > H0L0 > D_AUX >
NO_INPUT`; no tie-breaker changes it. The corrected generation-frame effects
are activation `+0.028212`, locality `+0.036024`, and interaction `+0.014757`.
This agreement is a retained-result audit, not a new final evaluation.

The retained scored result confirms that all historical probe vectors selected
class 0 and that the four main arms had 0 target changes in 48 probes. It does
not retain baseline probability vectors, so `baseline argmax == class 0` versus
`!= class 0` cannot be reconstructed. The result must therefore be described as
**0/48 target changes under fixed class-0 probability intervention**. It is not
evidence about the future alternate-value rule. Future code selects
`(baseline_argmax + 1) % width`, preserves source identity, decoder prefix, and
non-target factor vectors, and records the baseline and selected class indices.
The machine-readable audit is
[`benchmark-v3-protocol-errata.json`](benchmark-v3-protocol-errata.json).

## Protocol record

Before authorization, a sandbox-created temporary invocation directory failed
at the filesystem boundary and was removed while the durable marker was absent;
no final rows were accessed. After explicit authorization, one retained formal
invocation completed and produced the attested result above. The verification
call was rejected by the durable marker and did not overwrite the result.
