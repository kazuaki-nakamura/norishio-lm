"""Fixed-budget validation diagnosis of byte collapse and EOS failures."""
from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .concept_model import ConceptVocabulary
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import (conditional_loss, generation_metrics, predict_concepts,
                           train_control)
from .toy_experiment import ToyModel, batch, fixture_module
from .toy_generation import greedy_generate


def utf8_failure(tokens: Sequence[int]) -> dict[str, Any]:
    """Distinguish an illegal byte transition from a merely incomplete final codepoint."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    for position, token in enumerate(tokens):
        if token == 2:
            break
        if not 4 <= token <= 259:
            return {"kind": "special_token", "position": position, "token": token}
        try:
            decoder.decode(bytes([token - 4]), final=False)
        except UnicodeDecodeError:
            return {"kind": "illegal_transition", "position": position, "token": token}
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return {"kind": "incomplete_tail", "position": len(tokens), "token": None}
    return {"kind": "valid", "position": None, "token": None}


@torch.no_grad()
def teacher_diagnostics(model: ToyModel, rows: list[dict[str, Any]],
                        probabilities: torch.Tensor, tensorizer: SemanticTensorizer,
                        corpus: Any, strict: Any, batch_size: int = 16) -> dict[str, Any]:
    """Measure EOS specifically at true end positions; do not count PAD as bytes."""
    byte_correct = byte_count = eos_correct = eos_count = 0
    eos_probability = 0.0
    model.eval()
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        ids, labels, _ = batch(part, "C", tensorizer, corpus, strict)
        logits = model.decoder.decode_with_concept_intervention(
            ids, probabilities[start:start + len(part)])["logits"]
        predictions = logits.argmax(-1)
        byte_mask, eos_mask = labels >= 4, labels == 2
        byte_count += int(byte_mask.sum())
        byte_correct += int(((predictions == labels) & byte_mask).sum())
        eos_count += int(eos_mask.sum())
        eos_correct += int(((predictions == labels) & eos_mask).sum())
        eos_probability += float(logits.softmax(-1)[..., 2][eos_mask].sum())
    return {"byte_targets": byte_count, "byte_correct": byte_correct,
            "byte_accuracy": byte_correct / byte_count,
            "eos_targets": eos_count, "eos_correct": eos_correct,
            "eos_accuracy_at_gold_end": eos_correct / eos_count,
            "mean_eos_probability_at_gold_end": eos_probability / eos_count}


@torch.no_grad()
def free_trace(model: ToyModel, probability: torch.Tensor, max_new_tokens: int = 128) -> dict[str, Any]:
    """One preselected validation example, using full-history decoder as an oracle.

    This oracle verifies implementation equivalence, not gold semantic output.
    No references or reference lengths are available here.
    """
    history = torch.ones((1, 1), dtype=torch.long)
    tokens = []
    trace = []
    max_eos_probability = 0.0
    for step in range(max_new_tokens):
        logits = model.decoder.decode_with_concept_intervention(history, probability)["logits"][0, -1]
        probabilities = logits.softmax(-1)
        token = int(logits.argmax())
        max_eos_probability = max(max_eos_probability, float(probabilities[2]))
        if step < 12 or token == 2:
            trace.append({"position": step, "token": token,
                          "chosen_probability": float(probabilities[token]),
                          "eos_probability": float(probabilities[2]),
                          "eos_rank": int((logits > logits[2]).sum()) + 1})
        tokens.append(token)
        if token == 2:
            break
        history = torch.cat([history, torch.tensor([[token]])], dim=1)
    incremental = greedy_generate(model.decoder, probability, max_new_tokens)[0]
    return {"first_steps": trace, "tokens": tokens,
            "utf8": utf8_failure(tokens), "max_eos_probability": max_eos_probability,
            "incremental_matches_full_history": tokens == incremental.token_ids}


def run() -> dict[str, Any]:
    """Pre-fixed seed7, budgets60/600, batch16 and cap128; no test inspection."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train_rows, validation = bundle["train"], bundle["validation"]
    del bundle
    train_sources = [corpus.model_inputs(r) for r in train_rows]
    validation_sources = [corpus.model_inputs(r) for r in validation]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in train_sources])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train_rows])
    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary)
    fixed = predict_concepts(initial, train_sources, tensorizer).mean(0, keepdim=True)
    schedule = torch.randint(len(train_rows), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    report: dict[str, Any] = {"version": "byte-eos-diagnosis-1", "seed": 7,
        "budgets": [60, 600], "batch_size": 16, "max_new_tokens": 128,
        "torch": torch.__version__, "device": "cpu", "threads": 1,
        "test_evaluated": False, "conditions": {},
        "scope": "post-hoc validation diagnosis, fixed two budgets, not model selection"}
    for steps in (60, 600):
        for condition in ("predicted", "constant"):
            name = f"{condition}_{steps}"
            print(f"training {name}", flush=True)
            model, record = train_control(initial, condition, schedule[:steps], train_rows,
                                           tensorizer, vocabulary, corpus, strict, fixed)
            probabilities = (predict_concepts(model, validation_sources, tensorizer)
                             if condition == "predicted" else fixed.expand(len(validation), -1))
            generated = [asdict(g) for g in greedy_generate(model.decoder, probabilities, 128)]
            failures = [utf8_failure(g["token_ids"]) for g in generated]
            record.update({
                "validation": conditional_loss(model, validation, probabilities, tensorizer, corpus, strict),
                "teacher": teacher_diagnostics(model, validation, probabilities, tensorizer, corpus, strict),
                "generation": generation_metrics(generated, [r["targets"]["text"] for r in validation]),
                "utf8_failure_counts": {kind: sum(f["kind"] == kind for f in failures)
                                        for kind in ("valid", "illegal_transition", "incomplete_tail", "special_token")},
                "first_example_trace": free_trace(model, probabilities[:1]),
                "examples": [{"source": source, "authored_reference": row["targets"]["text"], **g}
                             for source, row, g in zip(validation_sources, validation, generated)]})
            report["conditions"][name] = record
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new path")
    report = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
