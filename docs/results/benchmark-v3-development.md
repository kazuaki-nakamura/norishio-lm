# Benchmark v3 factor-path development tournament

This is the frozen `diagnostic-validation` result for Issue #36. All 18 runs
used execution code `50577091f1aaa5b90045ab55bace25b5f69f5a52`, architecture
freeze `bc69ac3c132c77d9613481ae163b1e8c4342a9eb`, tournament config digest
`ff89911b64d8f6e652ed617f971ff0512dbda07e4c71bd585f5e0f23635fd399`,
and benchmark content digest
`930958ca1002b9f566fb13d28e072e99fa8ba5c0493308526586dcc128302d84`.

The run set is 18 complete / 0 failed. The authenticated terminal-record digest
is `f31a87807b5e9859df07ab8ad22a5f40a7f4188fdd492dfb43036d85185d8fac`.
The tracked machine-readable summary has SHA-256
`3b2a6b73386da45cb12d520e456637fdf247beedb060f2a061cf80e9b6ad40ce`.
The final-confirmation split remains unopened.

## Per-seed observations

`Free` is exact valid free generation. `Balanced` is the mean of the four
atomic balanced accuracies. `TF byte` is teacher-forced and is not a
free-generation result. Each intervention diagnostic has four examples per
run, so its rates are coarse; full values are in the JSON summary.

| Arm | Seed | Free | Triple | Pair | Balanced | TF byte | Parse coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0L0 | 7 | 0.164062 | 0.166667 | 0.182292 | 0.345703 | 0.953452 | 0.468750 |
| H0L0 | 17 | 0.210938 | 0.221354 | 0.221354 | 0.361979 | 0.957302 | 0.447917 |
| H0L0 | 29 | 0.138021 | 0.156250 | 0.166667 | 0.244141 | 0.954345 | 0.302083 |
| H0L1 | 7 | 0.229167 | 0.229167 | 0.239583 | 0.397135 | 0.959983 | 0.513021 |
| H0L1 | 17 | 0.132812 | 0.132812 | 0.132812 | 0.259766 | 0.953108 | 0.317708 |
| H0L1 | 29 | 0.242188 | 0.263021 | 0.268229 | 0.335938 | 0.965003 | 0.382812 |
| H1L0 | 7 | 0.197917 | 0.197917 | 0.213542 | 0.367188 | 0.957233 | 0.473958 |
| H1L0 | 17 | 0.166667 | 0.197917 | 0.197917 | 0.322917 | 0.953864 | 0.419271 |
| H1L0 | 29 | 0.166667 | 0.190104 | 0.208333 | 0.324219 | 0.959090 | 0.406250 |
| H1L1 | 7 | 0.244792 | 0.252604 | 0.263021 | 0.412109 | 0.964521 | 0.510417 |
| H1L1 | 17 | 0.153646 | 0.153646 | 0.153646 | 0.286458 | 0.954620 | 0.346354 |
| H1L1 | 29 | 0.252604 | 0.252604 | 0.257812 | 0.347005 | 0.967478 | 0.388021 |
| D_AUX | 7 | 0.000000 | 0.005208 | 0.049479 | 0.151042 | 0.908210 | 0.528646 |
| D_AUX | 17 | 0.013021 | 0.049479 | 0.208333 | 0.206380 | 0.919004 | 0.494792 |
| D_AUX | 29 | 0.000000 | 0.000000 | 0.033854 | 0.161458 | 0.915292 | 0.442708 |
| NO_INPUT | 7 | 0.000000 | 0.000000 | 0.020833 | 0.208333 | 0.901540 | 1.000000 |
| NO_INPUT | 17 | 0.000000 | 0.000000 | 0.020833 | 0.208333 | 0.900990 | 1.000000 |
| NO_INPUT | 29 | 0.000000 | 0.000000 | 0.000000 | 0.208333 | 0.894802 | 1.000000 |

## Three-seed means

| Arm | Free | Triple | Pair | Balanced | Exact text | TF byte |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1L1 | 0.217014 | 0.219618 | 0.224826 | 0.348524 | 0.217014 | 0.962207 |
| H0L1 | 0.201389 | 0.208333 | 0.213542 | 0.330946 | 0.201389 | 0.959365 |
| H1L0 | 0.177083 | 0.195312 | 0.206597 | 0.338108 | 0.177083 | 0.956729 |
| H0L0 | 0.171007 | 0.181424 | 0.190104 | 0.317274 | 0.171007 | 0.955033 |
| D_AUX | 0.004340 | 0.018229 | 0.097222 | 0.172960 | 0.004340 | 0.914169 |
| NO_INPUT | 0.000000 | 0.000000 | 0.013889 | 0.208333 | 0.000000 | 0.899111 |

The development ordering by the frozen primary metric is
`H1L1 > H0L1 > H1L0 > H0L0 > D_AUX > NO_INPUT`. This ordering is a
diagnostic observation, not the final-confirmation ranking.

## Preregistered 2x2 effects

Effects below are computed from the four main-arm three-seed means. Positive
activation means H1 exceeds H0; positive locality means L1 exceeds L0.

| Metric | Activation | Locality | Interaction |
| --- | ---: | ---: | ---: |
| Free generation exact | +0.010851 | +0.035156 | +0.009549 |
| Triple exact | +0.012587 | +0.025608 | -0.002604 |
| Pair exact | +0.013889 | +0.020833 | -0.005208 |
| Mean balanced atomic | +0.019206 | +0.012044 | -0.003255 |
| Teacher-forced byte | +0.002269 | +0.004905 | +0.001146 |

The full seed-specific effects are preserved in
[`benchmark-v3-development-summary.json`](benchmark-v3-development-summary.json).
With only three seeds and a small authored fixture, these differences are
descriptive and do not establish statistical significance or general model
superiority.

## Support-group split

| Arm | Group | Free | Triple | Pair | Balanced |
| --- | --- | ---: | ---: | ---: | ---: |
| H0L0 | unseen_pair | 0.039931 | 0.041667 | 0.041667 | 0.253906 |
| H0L0 | seen_pair/unseen_triple | 0.302083 | 0.321181 | 0.338542 | 0.380642 |
| H0L1 | unseen_pair | 0.074653 | 0.078125 | 0.079861 | 0.262587 |
| H0L1 | seen_pair/unseen_triple | 0.328125 | 0.338542 | 0.347222 | 0.399306 |
| H1L0 | unseen_pair | 0.036458 | 0.041667 | 0.041667 | 0.279080 |
| H1L0 | seen_pair/unseen_triple | 0.317708 | 0.348958 | 0.371528 | 0.397135 |
| H1L1 | unseen_pair | 0.076389 | 0.076389 | 0.078125 | 0.278212 |
| H1L1 | seen_pair/unseen_triple | 0.357639 | 0.362847 | 0.371528 | 0.418837 |
| D_AUX | unseen_pair | 0.003472 | 0.012153 | 0.046875 | 0.134983 |
| D_AUX | seen_pair/unseen_triple | 0.005208 | 0.024306 | 0.147569 | 0.210938 |
| NO_INPUT | unseen_pair | 0.000000 | 0.000000 | 0.000000 | 0.208333 |
| NO_INPUT | seen_pair/unseen_triple | 0.000000 | 0.000000 | 0.027778 | 0.208333 |

All learned-source arms show a large gap between unseen pairs and pairs seen in
training with unseen triples. This limits the claim to the fixed compositional
fixture and motivates retaining both support groups in final reporting.

The artificial one-hot intermediate intervention changed the target factor in
0/12 main-arm probes across the three seeds. Non-target preservation therefore
does not demonstrate factor control here: the intervention generally failed to
change the decoded target slot. NO_INPUT reached 0 free exact despite perfect
parse coverage, confirming that grammatical output alone is insufficient for
row-level exactness. D_AUX remained far below all four conditioned arms on the
primary metric. These are model observations on authored benchmark data, not
claims about learned modern word meaning, sememes, glyph structure, or general
Japanese understanding.
