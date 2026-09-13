# Benchmark v2 development tournament

This is the preregistered `diagnostic-validation` result for Issue #34. It is a
small authored structural benchmark, not evidence of general Japanese or LLM
quality. Free generation, parsed frame accuracy, teacher-forced bytes, and
intervention diagnostics are reported separately.

- implementation commit: `565c2bf1f8b22435f21a74d5833dfeb4821d8174`
- benchmark content digest: `012f57dbefb00f7661ed83aa49047d5736ab03c67d94b043292f40bb780b0134`
- tournament config digest: `a81458f89613215d412bea41ea48d5338a6a9c6c14e681400ab1272a4bcdd174`
- local terminal-record digest: `b0fce42488aeb28753a28a963eeb6bbf1304de4fa8ba3676923359ea05d73e29`
- local development-summary file digest: `aa33b8f5d77c6539f20a6e8f1a6754ffb171bdb62ef98538facfa64cf8a9825f`
- terminal status: 18 complete, 0 failed
- updates per run: 600; seeds: 7, 17, 29; CPU threads: 1

## Per-seed observations

| Arm | Seed | Free exact | Frame exact | Triple exact | Balanced atomic | TF byte | Locality |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_G0 | 7 | 0.028646 | 0.052083 | 0.104167 | 0.224514 | 0.939671 | 0.000000 |
| A_G0 | 17 | 0.031250 | 0.062500 | 0.080729 | 0.215686 | 0.936555 | 0.000000 |
| A_G0 | 29 | 0.015625 | 0.031250 | 0.031250 | 0.130390 | 0.944643 | 0.000000 |
| A_G1 | 7 | 0.026042 | 0.052083 | 0.104167 | 0.228479 | 0.939539 | 0.000000 |
| A_G1 | 17 | 0.028646 | 0.057292 | 0.075521 | 0.218732 | 0.936224 | 0.000000 |
| A_G1 | 29 | 0.010417 | 0.020833 | 0.026042 | 0.130600 | 0.943715 | 0.000000 |
| B | 7 | 0.216146 | 0.429688 | 0.429688 | 0.586966 | 0.967780 | 0.500000 |
| B | 17 | 0.161458 | 0.330729 | 0.330729 | 0.602764 | 0.958831 | 0.250000 |
| B | 29 | 0.291667 | 0.598958 | 0.661458 | 0.820522 | 0.973283 | 0.000000 |
| C | 7 | 0.072917 | 0.145833 | 0.177083 | 0.315236 | 0.942522 | 0.000000 |
| C | 17 | 0.023438 | 0.044271 | 0.065104 | 0.282270 | 0.930522 | 0.250000 |
| C | 29 | 0.088542 | 0.179688 | 0.197917 | 0.566183 | 0.948091 | 0.000000 |
| D | 7 | 0.000000 | 0.000000 | 0.000000 | 0.066465 | 0.915076 | 0.000000 |
| D | 17 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.886237 | 0.000000 |
| D | 29 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.886569 | 0.000000 |
| E | 7 | 0.054688 | 0.109375 | 0.184896 | 0.316315 | 0.944113 | 0.000000 |
| E | 17 | 0.013021 | 0.026042 | 0.039062 | 0.178836 | 0.929594 | 0.000000 |
| E | 29 | 0.039062 | 0.080729 | 0.122396 | 0.272668 | 0.944511 | 0.000000 |

## Three-seed arithmetic means

| Arm | Free exact | Frame exact | Triple exact | Balanced atomic | TF byte | Locality |
|---|---:|---:|---:|---:|---:|---:|
| A_G0 | 0.025174 | 0.048611 | 0.072049 | 0.190197 | 0.940290 | 0.000000 |
| A_G1 | 0.021701 | 0.043403 | 0.068576 | 0.192604 | 0.939826 | 0.000000 |
| B | 0.223090 | 0.453125 | 0.473958 | 0.670084 | 0.966631 | 0.250000 |
| C | 0.061632 | 0.123264 | 0.146701 | 0.387896 | 0.940378 | 0.083333 |
| D | 0.000000 | 0.000000 | 0.000000 | 0.022155 | 0.895960 | 0.000000 |
| E | 0.035590 | 0.072049 | 0.115451 | 0.255940 | 0.939406 | 0.000000 |

The frozen development ranking is `B > C > E > A_G0 > A_G1 > D`; no primary
metric tie required a tie-breaker. B is strongest on this fixture, but varies
materially across seeds. A_G1 does not improve on A_G0. E exceeds A on several
development metrics, so this comparison does not support a claim that the
authored factor labels themselves caused the improvement. D produces no exact
free or parsed frames despite high teacher-forced byte match. This gap is why
teacher-forced results cannot be called free-generation success.

Locality is especially limited: A_G0, A_G1, and D have four parse failures in
four intervention pairs, and several other arms also have failures. A zero
preservation value can therefore mean that an output was unparsable rather
than that a non-target factor was demonstrably changed.

The final holdout has not been opened. It requires a separate, one-shot
`--evaluate-final` invocation after explicit authorization.
