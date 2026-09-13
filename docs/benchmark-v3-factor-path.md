# Benchmark v3 factor-path preregistration

Issue #36 starts from PR #35 merge
`20fdff93ae3befd5b2d39ad28259f03d7d0adda5`. Phase 0 freezes a new
confirmation benchmark before any v3 model training. It does not reuse the
consumed benchmark-v2 final for selection or tuning.

The canonical authored specification is
`data/benchmark_v3_factor_path/spec.json`. Generator seed `20260914` and split
seed `90731` produce 1,152 rows: 384 train, 384 diagnostic-validation, and 384
final-confirmation. Each evaluation split contains 192 unseen participant-time
rows and 192 pair-known, unseen participant-time-event rows. Every atomic value
is present in train.

The v3 templates preserve the same small participant/time/event/operator
vocabulary family while changing the surface grammar mechanically. Source and
target text sets are globally disjoint, and v3 evaluation surface records do
not copy benchmark-v2 diagnostic/final records. Model input is restricted to
UTF-8 bytes of `context` and `text`; row, group, split, template, gold, and
whole-surface categorical identifiers remain unavailable.

`data/benchmark_v3_factor_path/expected-manifest.json` binds the canonical
specification, normalized generator source, split plan, counts, support
classes, row payloads, leakage contract, and content digest. The frozen Phase
0 content digest is
`930958ca1002b9f566fb13d28e072e99fa8ba5c0493308526586dcc128302d84`.
The spec digest is
`9011af9f0596a7ef4f6759930c9e7b5ca1c1cf947f8f027538e6450400c6d994`.

The common scorer prioritizes free-generation frame exact, triple exact, pair
exact, atomic balanced accuracy, and exact text in that order. Exact text also
measures surface-variant recovery and is not treated as a pure semantic metric.
Required support groups are `unseen_pair` and `seen_pair`, where `seen_pair`
means pair-known and triple-unseen for this fixture. Parse, EOS, UTF-8, unique
output, teacher-forced bytes, canonical intermediate heads, and the
head-versus-generation 2x2 remain separate diagnostics.

Source-side swaps and true intermediate probability interventions use separate
schemas. A true intervention must keep the source identity and decoder prefix
fixed, supply valid probabilities for all four factors, replace exactly one
factor with a width-correct artificial one-hot vector, and leave all other
factor vectors unchanged. It is not an oracle or a learned-model result.

Final-confirmation rows require an explicit flag and the exact manifest digest.
The architecture contract, checkpoint bindings, and one-shot invocation marker
will be frozen in a later commit before training. Phase 0 contains no training,
model comparison, final-confirmation evaluation, or performance claim.

