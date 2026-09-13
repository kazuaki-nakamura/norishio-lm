# Benchmark v2 final tournament

This is the single consumed `final-holdout` evaluation for Issue #34. It uses
the 18 checkpoints selected and frozen before final access. No architecture,
seed, schedule, budget, or checkpoint changed after the development report.
The authored structural fixture and tiny CPU models do not establish general
Japanese or LLM performance.

- implementation commit: `565c2bf1f8b22435f21a74d5833dfeb4821d8174`
- benchmark content digest: `012f57dbefb00f7661ed83aa49047d5736ab03c67d94b043292f40bb780b0134`
- tournament config digest: `a81458f89613215d412bea41ea48d5338a6a9c6c14e681400ab1272a4bcdd174`
- terminal-record digest: `b0fce42488aeb28753a28a963eeb6bbf1304de4fa8ba3676923359ea05d73e29`
- local final result digest: `1d21b4cfa950c3c96c50ca0647e33308bb4d0653c18bcc146560ce1da4266554`
- final invocation: consumed once; 384 rows; 18 evaluations; 0 failed runs
- second invocation check: rejected with `FileExistsError`, exit code 1

## Per-seed observations

`TF byte` is teacher-forced and is not a free-generation result. `Locality` is
the mean non-target preservation rate over four intervention factors. `Parse
failures` gives the number of the four intervention pairs that could not be
strictly parsed.

| Arm | Seed | Free exact | Frame exact | Triple exact | Balanced atomic | TF byte | Locality | Parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_G0 | 7 | 0.036458 | 0.070312 | 0.145833 | 0.274089 | 0.943056 | 0.000000 | 4 |
| A_G0 | 17 | 0.041667 | 0.080729 | 0.104167 | 0.243490 | 0.948545 | 0.000000 | 4 |
| A_G0 | 29 | 0.044271 | 0.062500 | 0.067708 | 0.115885 | 0.950860 | 0.000000 | 4 |
| A_G1 | 7 | 0.036458 | 0.070312 | 0.140625 | 0.271484 | 0.942923 | 0.000000 | 4 |
| A_G1 | 17 | 0.039062 | 0.083333 | 0.104167 | 0.242188 | 0.948413 | 0.000000 | 4 |
| A_G1 | 29 | 0.046875 | 0.065104 | 0.072917 | 0.121094 | 0.950992 | 0.000000 | 4 |
| B | 7 | 0.210938 | 0.421875 | 0.421875 | 0.585938 | 0.968783 | 0.000000 | 3 |
| B | 17 | 0.213542 | 0.419271 | 0.419271 | 0.647786 | 0.965873 | 0.250000 | 0 |
| B | 29 | 0.304688 | 0.611979 | 0.669271 | 0.834635 | 0.975463 | 0.750000 | 0 |
| C | 7 | 0.083333 | 0.153646 | 0.190104 | 0.343099 | 0.945833 | 0.000000 | 3 |
| C | 17 | 0.065104 | 0.130208 | 0.171875 | 0.285807 | 0.943056 | 0.250000 | 3 |
| C | 29 | 0.104167 | 0.190104 | 0.221354 | 0.558594 | 0.956283 | 0.250000 | 0 |
| D | 7 | 0.000000 | 0.000000 | 0.000000 | 0.060547 | 0.915939 | 0.000000 | 4 |
| D | 17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.888294 | 0.000000 | 4 |
| D | 29 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.888294 | 0.000000 | 4 |
| E | 7 | 0.041667 | 0.080729 | 0.169271 | 0.291016 | 0.947024 | 0.000000 | 4 |
| E | 17 | 0.023438 | 0.049479 | 0.138021 | 0.220052 | 0.938426 | 0.000000 | 4 |
| E | 29 | 0.041667 | 0.078125 | 0.138021 | 0.217448 | 0.949405 | 0.000000 | 4 |

## Three-seed arithmetic means

| Arm | Free exact | Frame exact | Triple exact | Balanced atomic | TF byte | Locality |
|---|---:|---:|---:|---:|---:|---:|
| A_G0 | 0.040799 | 0.071181 | 0.105903 | 0.211155 | 0.947487 | 0.000000 |
| A_G1 | 0.040799 | 0.072917 | 0.105903 | 0.211589 | 0.947443 | 0.000000 |
| B | 0.243056 | 0.484375 | 0.503472 | 0.689453 | 0.970040 | 0.333333 |
| C | 0.084201 | 0.157986 | 0.194444 | 0.395833 | 0.948391 | 0.166667 |
| D | 0.000000 | 0.000000 | 0.000000 | 0.020182 | 0.897509 | 0.000000 |
| E | 0.035590 | 0.069444 | 0.148438 | 0.242839 | 0.944951 | 0.000000 |

The frozen final ranking is `B > C > A_G1 > A_G0 > E > D`. A_G0 and
A_G1 tie on the primary free-exact mean; A_G1 wins the first tie-breaker,
generation-frame exact. B remains first on the unseen final split, while seed
29 is materially higher than its other seeds. This supports further testing of
the local independent-factor architecture on this benchmark; it does not prove
that the architecture is generally superior.

E does not outperform A on the final primary metric, while it did exceed A on
several development metrics. That instability does not isolate a general
benefit from the authored factor labels. D remains at zero free/frame exact
despite a high teacher-forced byte score. Locality has many strict parse
failures, so zero preservation cannot be interpreted as a confirmed semantic
change.

The [split audit](benchmark-v2-split-audit.md) exports both 192-row support
groups for every arm and seed from this saved result only. It performs no new
inference or final-holdout evaluation.

## Erratum: E intermediate label space

Arm E was trained against the frozen deranged factor-code labels, but the
original evaluator decoded its raw argmax indices directly with the canonical
factor vocabulary. Therefore the saved E intermediate atomic, pair, triple,
frame, and head-versus-generation 2x2 values are in mismatched label spaces and
must not be compared with other arms. The saved result contains aggregate
counts only; it does not retain raw logits, row-level predicted factor frames,
or a complete confusion matrix, so corrected historical E head values are
unavailable.

The evaluator now applies the inverse frozen derangement before passing E head
predictions to canonical diagnostics, with an integration regression for the
2x2 path. This correction was not used to rerun training, inference, or the
consumed final invocation. It does not affect the saved free-generation,
parsed-generation, teacher-forced, or locality values, and it does not change
the frozen ranking reported above.

## Post-result review hardening

The consumed artifact above was not regenerated or edited after final access.
Read-only replay through the review-added selection aggregator reproduced
`B > C > A_G1 > A_G0 > E > D`. Future invocations now emit machine-readable
three-seed means, ranking, tie-break traces, and a separate result attestation.
The already-consumed artifact predates that attestation, so its result digest
remains externally recorded in this document.

Checkpoint validation binds the frozen config, benchmark, schedule,
architecture metadata, model state, and file bytes. It does not embed the
formal runner/model/FSM source hash or implementation commit in checkpoint
metadata; the implementation commit above remains an external provenance
constraint for this result.

## Execution-set provenance

The reported 18 complete / 0 failed status applies only to the retained formal
run set executed from `565c2bf1f8b22435f21a74d5833dfeb4821d8174`.
Two pre-formal one-run smoke checks (D and E) were unsaved and excluded. An
earlier A_G0/A_G1 six-terminal attempt was interrupted by the unoptimized FSM;
its dedicated output was deleted and was not used in development or final
results. Deleted evidence has not been reconstructed.
