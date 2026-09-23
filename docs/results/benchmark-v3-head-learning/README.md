# Issue #46: source-to-factor head learnability

The preregistered six-run CPU comparison completed. The frozen descriptor is
`210fdbab6bcd4f8604d8653835ea5f075df3a41a2c79fb0fbc95bc9a07967322`,
committed before training at `2bf95812daea5db9c5016782f6068cb0f11627c0`.
The authored fixture digest is
`b587c091e3499348c5e9bcfabff68ac3bb3bbd2c1df60edaac62833cf69cbed5`.
The exact schedule, losses, split/row IDs, initial states, raw schema, and
interpretation rules are in the [frozen protocol](PROTOCOL.md) and
`data/benchmark_v3_head_learning/experiment-descriptor-v1.json`.

The frozen decision is **`basic_optimization_or_capacity_unresolved`**.
FACTOR_ONLY is weak on training resubstitution for all four structural factors
at all three seeds. Removing decoder LM loss therefore does not by itself
make the source heads reliably learn the training labels. JOINT and FACTOR_ONLY
confirmation results are close, and the preregistered all-seed, both-factor
positive direction needed for an interference interpretation is absent.
This result does not isolate optimization, capacity, and source representation
from each other.

| Arm | Split | Participant | Time | Event | Operator | Four-head exact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JOINT | train resubstitution | 491/1152 (0.426) | 484/1152 (0.420) | 665/1152 (0.577) | 658/1152 (0.571) | 19/1152 |
| FACTOR_ONLY | train resubstitution | 501/1152 (0.435) | 488/1152 (0.424) | 665/1152 (0.577) | 655/1152 (0.569) | 21/1152 |
| JOINT | confirmation | 170/288 (0.590) | 177/288 (0.615) | 183/288 (0.635) | 189/288 (0.656) | 36/288 |
| FACTOR_ONLY | confirmation | 174/288 (0.604) | 176/288 (0.611) | 182/288 (0.632) | 188/288 (0.653) | 35/288 |

Each denominator combines three seeds. The fixture is class-balanced in the
full train and confirmation splits, so atomic accuracy and class-balanced
accuracy coincide in this table. Per-class counts, missing/unavailable status,
and every seed are preserved in `summary.json`. Train results are
**resubstitution**, not a generalization estimate. All 2,880 scheduled raw
row observations were available.

| Arm | Confirmation support | Participant | Time | Event | Operator |
| --- | --- | ---: | ---: | ---: | ---: |
| JOINT | unseen pair, 144 rows | 111/144 (0.771) | 120/144 (0.833) | 76/144 (0.528) | 82/144 (0.569) |
| FACTOR_ONLY | unseen pair, 144 rows | 113/144 (0.785) | 120/144 (0.833) | 75/144 (0.521) | 82/144 (0.569) |
| JOINT | seen pair / unseen higher order, 144 rows | 59/144 (0.410) | 57/144 (0.396) | 107/144 (0.743) | 107/144 (0.743) |
| FACTOR_ONLY | seen pair / unseen higher order, 144 rows | 61/144 (0.424) | 56/144 (0.389) | 107/144 (0.743) | 106/144 (0.736) |

The two support strata have different authored combinations. In particular,
the confirmation unseen-pair participant/time scores exceed the seen-pair
stratum and the train resubstitution scores on this fixture. That pattern is
reported as observed; it does not establish compositional generalization or
explain the split difficulty. Event/operator strata contain only some of their
four classes each, so `summary.json` explicitly marks zero-support classes.

The confirmation paired deltas (`FACTOR_ONLY - JOINT`) for participant/time
were `+0.010/0.000` at seed 7, `+0.031/0.000` at seed 17, and
`0.000/-0.010` at seed 29. The four-factor exact metric is secondary; it is
not an arm ranking or a generation metric.

There were exactly six attempts and 3,600 optimizer updates, with no retry.
Summed optimizer-loop wall time was **87.097 seconds**; the saved executor wall
time, including head evaluation and raw writes, was **93.418 seconds**.
`attempts.jsonl` records every start and completion, and each `run-*.json`
retains all 384 train and 96 confirmation rows with canonical source-head
probability vectors and argmaxes. The raw-only aggregator checks attempt hashes,
freeze commit, fixture row metadata, schedules, probabilities, and outputs
before reproducing `summary.json`. No historical consumed final/checkpoint,
marker, or attestation was reopened; no GPU, network data, or paid compute was
used.

The next bounded experiment should first test why even FACTOR_ONLY cannot
reach strong **training** head accuracy, separating source encoding, objective
optimization, and parameter capacity with a newly frozen design. These results
are on hand-authored structural factors. They do not show learned modern
lexical meaning, sememes, concepts, glyph semantics, or general Japanese
language-model quality. Issue #44's frozen decoder-path decision is unchanged.

Validation after integration: `732 passed, 1 skipped, 2 subtests passed` for
the explicit `tests`, `codex/tools/tests`, and `codex/okf_mcp/tests` roots;
`validate_okf.py` reported 0 errors and 0 warnings; `check_knowledge.py
--mode full` checked 277 source entries with 0 errors; MCP self-check and
live check passed; the authored-dictionary demo exited successfully. The
first sandbox-wide pytest attempt encountered Windows temporary-directory ACL
errors, and the first CPU-Torch full run exposed CRLF conversion of older
byte-attested JSON. Explicit test roots and a repository-local temp location
resolved the former; `.gitattributes` preserves the exact historical artifact
bytes across Windows checkouts. These environmental failures are not counted
as passing test attempts.

The fixture, training runner, and metric aggregator were separate bounded
`gpt-5.6-luna` sub-agent tasks. A fourth Luna agent reviewed the pre-training
integration read-only and identified raw-aggregation binding checks; the
parent added those checks, froze the combined implementation, ran all six
attempts, and verified the final suite and GitHub state.

## Post-run source-encoding collision audit (PR #47 R1)

This is an append-only, deterministic audit of the frozen authored fixture and
`source_ids` inputs. It does not load a model or checkpoint, run inference, or
change any of the six raw runs, descriptor, fixture, attempts, thresholds, or
frozen decision. The executable audit and its machine-readable result are
[`benchmark_v3_head_collision_audit.py`](../../../src/norishio_lm/benchmark_v3_head_collision_audit.py)
and [`collision-audit.json`](collision-audit.json).

The H1L1 `SourceByteEncoder` averages embeddings over non-padding source tokens
and then applies a linear projection. The source templates reuse ASCII digits
in the participant, time, and predicate slots. Rows with the same complete
`source_ids` token-count multiset therefore produce the same encoder output,
regardless of token order or the role in which a digit occurs. The signature
includes BOS, SEP, and the canonical JSON wrapper; this is a property of the
actual model input, not of a simplified surface string.

| Frozen fixture slice | Rows | Unique signatures | Collision groups | Rows in collision groups | Participant ceiling | Time ceiling | Event ceiling | Operator ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 384 | 140 | 96 | 340 | 216/384 (0.5625) | 240/384 (0.6250) | 236/384 (0.6146) | 262/384 (0.6823) |
| Confirmation, all | 96 | 70 | 26 | 52 | 94/96 (0.9792) | 94/96 (0.9792) | 70/96 (0.7292) | 70/96 (0.7292) |
| Unseen pair | 48 | 34 | 14 | 28 | 46/48 (0.9583) | 46/48 (0.9583) | 34/48 (0.7083) | 34/48 (0.7083) |
| Seen pair / unseen higher order | 48 | 36 | 12 | 24 | 48/48 (1.0000) | 48/48 (1.0000) | 36/48 (0.7500) | 36/48 (0.7500) |

For each factor, the ceiling sums the majority gold-label count within each
signature and divides by the number of scheduled rows. It is the best possible
**row accuracy for a deterministic signature-only classifier on that slice**,
not an achieved model score. The JSON also reports per-factor counts of
signatures with conflicting gold labels. All four train factors have equal
per-class support (64 rows per participant/time class, 96 per event/operator
class), so row accuracy and class-balanced accuracy coincide there. Thus the
pre-registered 0.80 strong **training** gate is structurally unreachable for
every factor with this frozen encoder/fixture combination. The confirmation
strata have their own label support and ceilings; their high participant/time
ceilings do not repair the train identifiability defect or establish
generalization.

This confirms a definite source-representation information loss, while the
relative contribution of optimization or parameter capacity beyond that loss
remains unmeasured. The frozen result field stays
`basic_optimization_or_capacity_unresolved`; the new audit narrows its
interpretation rather than retroactively selecting another branch. A future,
separately frozen control should compare role-aliased and role-distinct or
order-identifiable inputs under factor-only training before making a joint-loss
or decoder-path claim. No such experiment was run in this audit.

R1 validation: the five collision-audit tests and the integrated CPU suite
passed (`737 passed, 1 skipped, 2 subtests passed`). The audit JSON exactly
matches regeneration from the frozen fixture; OKF validation had 0 errors and
0 warnings, its full index checked 280 sources without error, and MCP
self/live checks passed. The initial unprivileged focused pytest command
encountered the known Windows temp-directory ACL error; the successful run
used an explicit writable temp directory with the required permission.
