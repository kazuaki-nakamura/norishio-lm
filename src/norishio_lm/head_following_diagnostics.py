"""Small, inference-only diagnostics for dedicated slot heads (Issue 26)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary
from .local_slot_generation import greedy_generate_local
from .toy_controls import conditional_loss, generation_metrics
from .toy_slots import score_slots
from .toy_experiment import pad
from .toy_spans import authored_slot_spans
from .span_metrics import span_metrics


_SLOT_FIELDS = ("participant", "time")


def _field_slices(vocabulary: ConceptVocabulary, width: int) -> dict[str, slice]:
    offset = 0
    result: dict[str, slice] = {}
    for field in CONCEPT_FIELDS:
        end = offset + len(vocabulary.fields[field]) + 1
        result[field] = slice(offset, end)
        offset = end
    if offset != width:
        raise ValueError(f"old concept width {width} does not match vocabulary ({offset})")
    return result


def _validate_three(old: Tensor, heads: Tensor, gold: Tensor) -> None:
    if not all(isinstance(value, Tensor) and value.ndim == 2 for value in (old, heads, gold)):
        raise ValueError("old, heads, and gold must be rank-2 tensors")
    if old.shape[0] != heads.shape[0] or heads.shape != gold.shape:
        raise ValueError("old, heads, and gold must have aligned batch dimensions")
    if heads.shape[1] != 10:
        raise ValueError("heads and gold must have ten values (five per slot)")
    if not bool(torch.isfinite(heads).all()) or not bool(torch.isfinite(gold).all()):
        raise ValueError("heads and gold must be finite")
    if bool((heads < 0).any()) or bool((gold < 0).any()):
        raise ValueError("heads and gold must be nonnegative probabilities")
    ones = torch.ones(len(heads), device=heads.device)
    if not bool(torch.allclose(heads[:, :5].sum(-1), ones, atol=1e-5)) or not bool(torch.allclose(heads[:, 5:].sum(-1), ones, atol=1e-5)):
        raise ValueError("each predicted head must sum to one")


def base_ablation_cases(old: Tensor, heads: Tensor, gold: Tensor,
                        vocabulary: ConceptVocabulary) -> dict[str, Tensor]:
    """Return the eight frozen decoder inputs used by the base ablation.

    The dedicated participant/time block is always retained.  ``base_zero``
    therefore zeros the old non-slot channels, while event and operators cases
    zero only their vocabulary-defined blocks.  Gold replacement affects only
    the dedicated block and never mutates either input tensor.
    """
    _validate_three(old, heads, gold)
    slices = _field_slices(vocabulary, int(old.shape[1]))
    base_zero = old.clone()
    for field in CONCEPT_FIELDS:
        if field not in _SLOT_FIELDS:
            base_zero[:, slices[field]] = 0

    def changed(field: str | None) -> Tensor:
        value = old.clone()
        if field is None:
            value = base_zero
        else:
            value[:, slices[field]] = 0
        return value

    def joined(base: Tensor, dedicated: Tensor) -> Tensor:
        return torch.cat((base, dedicated), dim=-1)

    return {
        "full_predicted": joined(old, heads),
        "full_both_gold": joined(old, gold),
        "base_zero_predicted": joined(changed(None), heads),
        "base_zero_both_gold": joined(changed(None), gold),
        "event_zero_predicted": joined(changed("event"), heads),
        "event_zero_both_gold": joined(changed("event"), gold),
        "operators_zero_predicted": joined(changed("operators"), heads),
        "operators_zero_both_gold": joined(changed("operators"), gold),
    }


def _gold_head_ids(target: Mapping[str, Any], vocabulary: ConceptVocabulary) -> tuple[int, int] | None:
    concept = target.get("concept")
    if not isinstance(concept, Mapping):
        return None
    values = []
    for field in _SLOT_FIELDS:
        value = concept.get(field)
        if value not in vocabulary.fields[field]:
            raise ValueError(f"unknown training label for {field}: {value!r}")
        values.append(vocabulary.fields[field][value] - 1)
    return values[0], values[1]


def confident_train_subsets(heads: Tensor, train_targets: Sequence[Mapping[str, Any]],
                            vocabulary: ConceptVocabulary, *, threshold: float = .8) -> dict[str, list[int]]:
    """Select confident rows using only source predictions and train labels."""
    if not isinstance(heads, Tensor) or heads.ndim != 2 or heads.shape[1] != 10:
        raise ValueError("heads must have shape [B, 10]")
    if len(heads) != len(train_targets):
        raise ValueError("heads and train_targets must have equal row counts")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if not heads.is_floating_point() or not bool(torch.isfinite(heads).all()) or bool((heads < 0).any()):
        raise ValueError("train heads must be finite nonnegative probabilities")
    sums = heads.reshape(len(heads), 2, 5).sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6):
        raise ValueError("train head probabilities must sum to one")
    gold_ids = [_gold_head_ids(target, vocabulary) for target in train_targets]
    if any(gold is None for gold in gold_ids):
        raise ValueError("all train targets must have known participant/time labels")
    confident_correct: list[int] = []
    confident_wrong: list[int] = []
    for row, target in enumerate(train_targets):
        participant, time = heads[row, :5], heads[row, 5:]
        if float(participant.max()) < threshold or float(time.max()) < threshold:
            continue
        gold = gold_ids[row]
        if gold is None:
            raise ValueError(f"train target {row} has no participant/time labels")
        correct = int(participant.argmax()) == gold[0] and int(time.argmax()) == gold[1]
        (confident_correct if correct else confident_wrong).append(row)
    return {"confident_correct": confident_correct, "confident_wrong": confident_wrong}


def _both_slots(slot_result: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = 0
    for example, target in zip(slot_result.get("examples", []), targets):
        parsed = example.get("parsed")
        gold = target.get("concept") if isinstance(target, Mapping) else None
        if parsed is not None and isinstance(gold, Mapping) and parsed.get("participant") == gold.get("participant") and parsed.get("time") == gold.get("time"):
            correct += 1
    count = len(targets)
    return {"correct": correct, "count": count, "accuracy": correct / count if count else None}


def pair_following(generated: Sequence[Mapping[str, Any]], targets: Sequence[Mapping[str, Any]],
                   vocabulary: ConceptVocabulary, seed: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Group an existing generation by its gold participant/time pair."""
    if len(generated) != len(targets):
        raise ValueError("generated and targets must have equal length")
    if any(_gold_head_ids(target, vocabulary) is None for target in targets):
        raise ValueError("pair scoring requires known participant/time labels")
    participants = [v for v, _ in sorted(vocabulary.fields["participant"].items(), key=lambda item: item[1])]
    times = [v for v, _ in sorted(vocabulary.fields["time"].items(), key=lambda item: item[1])]
    result: dict[str, dict[str, Any]] = {}
    for pi, participant in enumerate(participants):
        for ti, time in enumerate(times):
            indexes = [i for i, target in enumerate(targets)
                       if isinstance(target.get("concept"), Mapping)
                       and target["concept"].get("participant") == participant
                       and target["concept"].get("time") == time]
            scores = score_slots([generated[i] for i in indexes], [targets[i] for i in indexes], seed)
            result[f"{pi},{ti}"] = {
                "rows": len(indexes),
                "participant_correct": scores["fields"]["participant"]["correct"],
                "time_correct": scores["fields"]["time"]["correct"],
                "both_correct": sum(1 for example, target in zip(scores["examples"], [targets[i] for i in indexes])
                                     if example.get("parsed") is not None and isinstance(target.get("concept"), Mapping)
                                     and example["parsed"].get("participant") == target["concept"].get("participant")
                                     and example["parsed"].get("time") == target["concept"].get("time")),
                "fullframe_correct": scores["fullframe_correct"],
            }
            n = len(indexes)
            result[f"{pi},{ti}"]["rates"] = (None if not n else {
                "participant": result[f"{pi},{ti}"]["participant_correct"] / n,
                "time": result[f"{pi},{ti}"]["time_correct"] / n,
                "both": result[f"{pi},{ti}"]["both_correct"] / n,
                "fullframe": result[f"{pi},{ti}"]["fullframe_correct"] / n,
            })
    return result


@torch.no_grad()
def evaluate_condition(model: Any, rows: Sequence[Mapping[str, Any]], probs: Tensor,
                       tensorizer: Any, vocabulary: ConceptVocabulary, corpus: Any,
                       strict: Any) -> dict[str, Any]:
    """Evaluate one frozen conditioning tensor, including parse-based slots."""
    rows = list(rows)
    targets = [row["targets"] for row in rows]
    if probs.ndim != 2 or len(probs) != len(rows):
        raise ValueError("probs must have one row per example")
    seed = corpus.seed_data()
    if not rows:
        empty_slots = score_slots([], [], seed)
        return {"rows": 0, "lm": None, "generation": None, "slots": empty_slots,
                "both_slots": {"correct": 0, "count": 0, "accuracy": None},
                "by_gold_pair": pair_following([], [], vocabulary, seed), "teacher_forced": None, "generations": []}
    model.eval()
    generations = [asdict(value) for value in greedy_generate_local(model.decoder, probs)]
    slots = score_slots(generations, targets, seed)
    ids = pad([strict.target_history(row)["input_ids"] for row in rows])
    labels = pad([strict.target_history(row)["labels"] for row in rows], -100)
    reference_logits = model.decoder.decode_with_concept_intervention(ids, probs)["logits"]
    spans = [authored_slot_spans(target, seed) for target in targets]
    teacher_forced = None
    if any(span is None for span in spans):
        raise ValueError("unrecognized authored participant/time span")
        teacher_forced = {"reference_history_oracle": True,
                          "comparison": "self baseline; byte quality only, not intervention sensitivity",
                      "metrics": span_metrics(reference_logits, reference_logits, labels, spans)}
    return {"rows": len(rows),
            "lm": conditional_loss(model, rows, probs, tensorizer, corpus, strict),
            "generation": generation_metrics(generations, [target["text"] for target in targets]),
            "slots": slots, "both_slots": _both_slots(slots, targets),
            "by_gold_pair": pair_following(generations, targets, vocabulary, seed),
            "teacher_forced": teacher_forced,
            "generations": generations}


__all__ = ["base_ablation_cases", "confident_train_subsets", "pair_following", "evaluate_condition"]
