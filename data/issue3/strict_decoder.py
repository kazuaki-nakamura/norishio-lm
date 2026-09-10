"""Target-history-only input for the strict concept-bottleneck decoder (path C).

A/B may use toy_corpus.teacher_forcing's source-prefixed sequence. C must NOT:
otherwise raw-source tokens or their attention KV cache bypass the bottleneck.
C's source encoder reads toy_corpus.source_ids(row), predicts explicit concepts,
and the decoder receives only those predicted concepts plus the history below.
The caller must enforce causal masking and must not pass this row to model.forward.
Gold concepts go to the auxiliary loss only. Soft concept scores may still carry
extra information, so hard/soft usage and concept interventions must be reported.
"""
from __future__ import annotations
from typing import Any

BOS, EOS, BYTE_OFFSET = 1, 2, 4


def target_history(row: dict[str, Any]) -> dict[str, list[int]]:
    """Shift the reference by one token; no source prefix enters the decoder."""
    text = row["targets"]["text"]
    if not isinstance(text, str) or not text:
        raise ValueError("a nonempty reference text is required")
    answer = [b + BYTE_OFFSET for b in text.encode("utf-8")]
    return {"input_ids": [BOS, *answer], "labels": [*answer, EOS]}
