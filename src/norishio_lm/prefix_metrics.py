"""Metrics for evaluating generation after an arbitrary supplied prefix.

The scorer deliberately works on token ids and explicit spans.  It does not
parse generated text, which keeps byte offsets unambiguous for multibyte text.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

_FIELDS = ("participant", "time")
_BYTE_MIN, _BYTE_MAX, _VOCAB = 4, 259, 260
_EOS = 2


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _slot_summary(full: list[bool | None], eligible: list[bool], *, total: int) -> dict[str, Any]:
    exact = sum(bool(x) for x in full if x is not None)
    eligible_count = sum(eligible)
    return {
        "exact_rows": exact,
        "exact_rate": exact / eligible_count if eligible_count else None,
        "eligible_count": eligible_count,
        "ineligible_count": total - eligible_count,
        "row_denominator": total,
        "eligible_row_denominator": eligible_count,
    }


def _validate(generations: Sequence[Mapping[str, Any]], references: Sequence[Sequence[int]],
              spans: Sequence[Mapping[str, Mapping[str, int]]]) -> None:
    if not isinstance(generations, Sequence) or isinstance(generations, (str, bytes)):
        raise ValueError("generations must be a sequence")
    if len(generations) != len(references) or len(generations) != len(spans):
        raise ValueError("generations, references, and spans must have equal lengths")
    for i, (generation, reference, row_spans) in enumerate(zip(generations, references, spans)):
        if not isinstance(generation, Mapping):
            raise ValueError(f"generations[{i}] must be a mapping")
        for key in ("token_ids", "generated_token_ids", "prefix_length", "decision_logits", "ended_eos"):
            if key not in generation:
                raise ValueError(f"generations[{i}] missing {key!r}")
        prefix = generation["prefix_length"]
        if not isinstance(prefix, int) or isinstance(prefix, bool) or prefix < 0:
            raise ValueError("prefix_length must be a nonnegative integer")
        if not isinstance(reference, Sequence) or isinstance(reference, (str, bytes)):
            raise ValueError("references must contain integer sequences")
        if any(not isinstance(x, int) or isinstance(x, bool) or not _BYTE_MIN <= x <= _BYTE_MAX for x in reference):
            raise ValueError("references contain a non-byte token")
        token_ids = generation["token_ids"]
        generated_ids = generation["generated_token_ids"]
        if (not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)) or
                not isinstance(generated_ids, Sequence) or isinstance(generated_ids, (str, bytes))):
            raise ValueError("token_ids and generated_token_ids must be sequences")
        if len(token_ids) < prefix or list(token_ids[prefix:]) != list(generated_ids):
            raise ValueError("token_ids must concatenate the supplied prefix and generated_token_ids")
        logits = generation["decision_logits"]
        if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[1] != _VOCAB:
            raise ValueError("decision_logits must have shape [generated_steps, 260]")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("decision_logits must be finite")
        if len(generated_ids) != logits.shape[0]:
            raise ValueError("decision_logits rows must equal generated_token_ids length")
        if not isinstance(row_spans, Mapping) or set(row_spans) != set(_FIELDS):
            raise ValueError(f"spans[{i}] must contain participant and time")
        for field in _FIELDS:
            span = row_spans[field]
            if not isinstance(span, Mapping) or set(span) != {"start", "end"}:
                raise ValueError("each span must contain start and end")
            start, end = span["start"], span["end"]
            if (not isinstance(start, int) or isinstance(start, bool) or
                    not isinstance(end, int) or isinstance(end, bool) or start < 0 or end <= start or
                    end > len(reference)):
                raise ValueError("spans must be nonempty ranges within the reference")


def score_prefix_rows(
    generations: list[dict[str, Any]],
    references: list[list[int]],
    spans: list[dict[str, dict[str, int]]],
) -> dict[str, Any]:
    """Score byte slots and continuations with supplied prefixes excluded.

    Span offsets are absolute byte-token offsets and half-open.  A decision
    logit row ``j`` predicts absolute position ``prefix_length + j``.  Thus
    only decisions at or after ``prefix_length`` can contribute metrics.
    Missing decisions (including premature EOS) remain in the expected-byte
    denominator, while probability/rank/NLL are reported only for evaluated
    decisions.
    """
    _validate(generations, references, spans)
    total = len(generations)
    field_data: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {f: [] for f in _FIELDS}
    for field in _FIELDS:
        full: list[bool | None] = []
        eligible: list[bool] = []
        suffix_expected = suffix_matched = 0
        suffix_exact_rows = 0
        full_expected = full_matched = 0
        byte_items: list[dict[str, Any]] = []
        for row, (generation, reference, row_spans) in enumerate(zip(generations, references, spans)):
            start, end = row_spans[field]["start"], row_spans[field]["end"]
            prefix = generation["prefix_length"]
            ids = [int(x) for x in generation["generated_token_ids"]]
            eligible_row = prefix <= start and end <= len(reference)
            eligible.append(eligible_row)
            slot_start = start - prefix
            slot_end = end - prefix
            full.append(bool(ids[slot_start:slot_end] == reference[start:end]) if eligible_row else None)
            if eligible_row:
                expected_full = reference[start:end]
                full_expected += len(expected_full)
                full_matched += sum(a == b for a, b in zip(ids[slot_start:slot_end], expected_full))
            first = max(start, prefix)
            expected = list(reference[first:end]) if first < len(reference) else []
            suffix_expected += len(expected)
            matched = sum(1 for offset, target in enumerate(expected)
                          if first - prefix + offset < len(ids) and
                          ids[first - prefix + offset] == target)
            suffix_matched += matched
            suffix_exact_rows += int(bool(expected) and matched == len(expected) and
                                     first - prefix + len(expected) <= len(ids))
            row_items: list[dict[str, Any]] = []
            logits = generation["decision_logits"].detach().to(device="cpu", dtype=torch.float64)
            for absolute in range(first, min(end, len(reference))):
                step = absolute - prefix
                # A row that stopped early has no self-history decision for
                # later positions, even if a caller supplied padded logits.
                if step < 0 or step >= logits.shape[0] or step >= len(ids):
                    continue
                target = int(reference[absolute])
                line = logits[step]
                logp = torch.log_softmax(line, dim=-1)
                probability = float(logp[target].exp())
                rank = 1 + int((line > line[target]).sum())
                item = {"position": absolute, "decision_step": step, "target_id": target,
                        "probability": probability, "nll": float(-logp[target]),
                        "rank": rank, "argmax_correct": bool(int(line.argmax()) == target)}
                row_items.append(item)
                byte_items.append(item)
            examples[field].append({"row": row, "fullslot": full[-1], "eligible": eligible_row,
                                    "expected_bytes": len(expected), "matched_bytes": matched,
                                    "evaluated_bytes": len(row_items), "bytes": row_items})
        summary = _slot_summary(full, eligible, total=total)
        evaluated = len(byte_items)
        byte = {
            "expected_bytes": suffix_expected, "matched_bytes": suffix_matched,
            "match_rate": suffix_matched / suffix_expected if suffix_expected else None,
            "evaluated_bytes": evaluated,
            "coverage": evaluated / suffix_expected if suffix_expected else None,
            "mean_probability": _mean([x["probability"] for x in byte_items]),
            "mean_nll": _mean([x["nll"] for x in byte_items]),
            "mean_rank": _mean([float(x["rank"]) for x in byte_items]),
            "argmax_correct": sum(x["argmax_correct"] for x in byte_items),
            "argmax_accuracy": sum(x["argmax_correct"] for x in byte_items) / evaluated if evaluated else None,
            "exact_rows": suffix_exact_rows,
            "row_denominator": total,
        }
        summary["expected_token_count"] = full_expected
        summary["exact_token_count"] = full_matched
        summary["exact_all_row_rate"] = summary["exact_rows"] / total if total else None
        summary["exact_eligible_rate"] = summary["exact_rows"] / summary["eligible_count"] if summary["eligible_count"] else None
        field_data[field] = {"fullslot": summary, "suffix": byte, "byte": byte}

    all_success = 0
    slot_success: dict[str, int] = {field: 0 for field in _FIELDS}
    slot_eligible: dict[str, int] = {field: 0 for field in _FIELDS}
    slot_continuations: dict[str, dict[str, Any]] = {}
    for generation, reference, row_spans in zip(generations, references, spans):
        prefix = generation["prefix_length"]
        ids = [int(x) for x in generation["generated_token_ids"]]
        expected = reference[prefix:] + [_EOS]
        all_success += int(ids == expected)
        for field in _FIELDS:
            span = row_spans[field]
            if prefix <= span["start"]:
                slot_eligible[field] += 1
                slot_begin = span["start"] - prefix
                slot_end = span["end"] - prefix
                slot_ok = ids[slot_begin:slot_end] == reference[span["start"]:span["end"]]
                # Supplied bytes before the slot are intentionally irrelevant.
                slot_success[field] += int(slot_ok and ids[slot_end:] == reference[span["end"]:] + [_EOS])
    continuation = {"success_rows": all_success, "row_denominator": total,
                    "success_rate": all_success / total if total else None,
                    "definition": "generated bytes after prefix exactly equal reference suffix followed by EOS"}
    for field in _FIELDS:
        slot_continuations[field] = {
            "success_rows": slot_success[field], "row_denominator": total,
            "eligible_count": slot_eligible[field],
            "success_rate": slot_success[field] / total if total else None,
            "eligible_success_rate": slot_success[field] / slot_eligible[field] if slot_eligible[field] else None,
            "definition": f"full {field} slot exact for eligible rows, then remaining suffix and EOS exact; supplied bytes before the slot are ignored",
        }
    return {"version": "prefix-metrics-1", "row_count": total, "fields": field_data,
            "aggregate": {"continuation": continuation, "slot_start_continuation": slot_continuations},
            "examples": examples}


__all__ = ["score_prefix_rows"]
