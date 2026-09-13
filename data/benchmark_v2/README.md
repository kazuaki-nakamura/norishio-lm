# Norishio benchmark v2 fixture

This directory freezes the authored, machine-generated fixture for Issue #34.
It is benchmark infrastructure, not learned output and not independently
validated Japanese data.

The four independent factors are `participant`, `time`, `event`, and
`operator`. The full grid has 6 x 6 x 4 x 4 frames and two surface variants
per frame. Both variants share a `group_id` and always remain in one split.

The split rule is derived only from `split_seed=3401`:

- one Latin matching of participant-time pairs is held out for
  `diagnostic-validation`;
- a disjoint matching is held out for `final-holdout`;
- for every remaining participant-time pair, one event is held out for each
  evaluation split, leaving two events in train.

Thus each evaluation split contains equal numbers of unseen pairs and of
unseen triples whose participant-time pair was seen in train. Every atomic
factor value occurs in train. All operator values occur for every selected
triple.

`inputs` exposes only `context` and raw `text`. Row, group, split, template,
gold-frame, and provenance fields stay outside the model-input boundary. The
raw text can be byte/character encoded; it must not be converted into one
categorical ID for the whole source surface. The predetermined factor
derangements in the manifest are the only Phase-2 broken-semantics shuffles.

Validate the committed freeze without writing data:

```powershell
.venv\Scripts\python.exe data\benchmark_v2\benchmark.py --check
```

Export deterministic JSONL into a new ignored directory:

```powershell
.venv\Scripts\python.exe data\benchmark_v2\benchmark.py `
  --out codex\work_output\norishio-benchmark-v2
```

Existing exports are never overwritten. Model selection may use
`diagnostic-validation`. Code must call `evaluation_rows(...,
evaluate_final=True, manifest_digest=...)` for `final-holdout`, and only after
all tournament arms and seeds plus checkpoint manifests have been fixed.
