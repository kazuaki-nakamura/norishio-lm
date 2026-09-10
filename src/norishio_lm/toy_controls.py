"""Validation-only controls for concept content, separate from the v1 experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, encode_targets
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_experiment import ToyModel, batch, evaluate, fixture_module


def state_digest(model: ToyModel) -> str:
    """Portable in this CPU experiment; no NumPy dependency or checkpoint file."""
    payload = {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "values": value.detach().cpu().reshape(-1).tolist()}
               for name, value in sorted(model.state_dict().items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def predict_concepts(model: ToyModel, sources: Sequence[dict[str, str]],
                     tensorizer: SemanticTensorizer, batch_size: int = 16) -> torch.Tensor:
    """Source allowlist only: no reference history or annotations are accepted."""
    if not sources or batch_size < 1 or model.pathway != "C":
        raise ValueError("nonempty sources, positive batch size and pathway C required")
    model.eval()
    outputs = []
    for start in range(0, len(sources), batch_size):
        semantic = tensorizer.encode([source_record(s) for s in sources[start:start + batch_size]])
        latent = model.encoder(semantic).fused
        outputs.append(model.decoder.bottleneck(latent).probabilities())
    return torch.cat(outputs).detach()


def majority_baseline(train_targets: Sequence[Mapping[str, Any]],
                      validation_targets: Sequence[Mapping[str, Any]],
                      vocabulary: ConceptVocabulary) -> dict[str, Any]:
    """Fit each majority on train; ties choose the smallest train vocabulary ID."""
    train = encode_targets(train_targets, vocabulary)
    validation = encode_targets(validation_targets, vocabulary)
    correct = count = 0
    fields = {}
    for name in CONCEPT_FIELDS:
        ids = train.fields[name][train.masks[name]]
        mask = validation.masks[name]
        selected = int(torch.bincount(ids).argmax()) if ids.numel() else None
        n = int(mask.sum()) if selected is not None else 0
        successes = int(((validation.fields[name] == selected) & mask).sum()) if n else 0
        inverse = {i: value for value, i in vocabulary.fields[name].items()}
        fields[name] = {"value": inverse.get(selected), "id": selected, "count": n,
                        "correct": successes, "accuracy": successes / n if n else None}
        count += n
        correct += successes
    return {"fields": fields, "count": count, "correct": correct,
            "accuracy": correct / count if count else None}


def train_control(initial: ToyModel, condition: str, schedule: Sequence[Sequence[int]],
                  train_rows: list[dict[str, Any]], tensorizer: SemanticTensorizer,
                  vocabulary: ConceptVocabulary, corpus: Any, strict: Any,
                  fixed: torch.Tensor) -> tuple[ToyModel, dict[str, Any]]:
    """Clone identical weights, then train with the same externally fixed schedule.

    The constant vector is the initial model's mean train prediction, not labels
    or a trained-model oracle. Auxiliary training remains enabled in both arms.
    """
    if condition not in {"predicted", "constant"}:
        raise ValueError("condition must be predicted or constant")
    if fixed.ndim != 2 or fixed.shape[0] != 1:
        raise ValueError("fixed must be a single concept probability row")
    model = deepcopy(initial)
    frozen = fixed.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trace = []
    started = time.perf_counter()
    model.train()
    for step, indices in enumerate(schedule):
        rows = [train_rows[i] for i in indices]
        ids, labels, semantic = batch(rows, "C", tensorizer, corpus, strict)
        intervention = frozen.expand(len(rows), -1) if condition == "constant" else None
        optimizer.zero_grad(set_to_none=True)
        output, aux = model(ids, semantic, labels=labels, intervention=intervention)
        targets = encode_targets([r["targets"] for r in rows], vocabulary)
        total = model.decoder.losses(output, targets=targets, bottleneck_output=aux)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == len(schedule):
            losses = model.decoder.bottleneck.loss(aux, targets)
            trace.append({"step": step + 1, "total": float(total.detach()),
                          "lm": float(output["lm_loss"].detach()),
                          "sense": float(losses["sense"].detach()),
                          "sememe": float(losses["sememes"].detach()),
                          "concept": float(losses["concept"].detach())})
    return model, {"condition": condition, "steps": len(schedule),
                   "initial_state_sha256": state_digest(initial),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "training_seconds": time.perf_counter() - started, "trace": trace}


@torch.no_grad()
def conditional_loss(model: ToyModel, rows: list[dict[str, Any]],
                     probabilities: torch.Tensor, tensorizer: SemanticTensorizer,
                     corpus: Any, strict: Any, batch_size: int = 16) -> dict[str, Any]:
    """Explicit global row alignment, independent of evaluation mini-batches."""
    if probabilities.ndim != 2 or len(probabilities) != len(rows) or not rows or batch_size < 1:
        raise ValueError("one probability row per example and positive batch size required")
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        ids, labels, _ = batch(part, "C", tensorizer, corpus, strict)
        output = model.decoder.decode_with_concept_intervention(
            ids, probabilities[start:start + len(part)], labels=labels)
        n = int((labels != -100).sum())
        total += float(output["lm_loss"]) * n
        count += n
    return {"rows": len(rows), "lm_tokens": count, "lm": total / count}


def generation_metrics(generated: list[dict[str, Any]], references: Sequence[str]) -> dict[str, Any]:
    """Exact byte+EOS match; no reference is given to the generation routine."""
    if not references or len(generated) != len(references):
        raise ValueError("nonempty aligned generations and references required")
    expected = [[*(b + 4 for b in reference.encode("utf-8")), 2] for reference in references]
    return {"rows": len(generated),
            "exact_match": sum(g["token_ids"] == e for g, e in zip(generated, expected)) / len(generated),
            "eos_rate": sum(g["ended_eos"] for g in generated) / len(generated),
            "valid_utf8_rate": sum(g["valid_utf8"] for g in generated) / len(generated),
            "special_token_rows": sum(bool(g["invalid_special_tokens"]) for g in generated),
            "unique_sequences": len({tuple(g["token_ids"]) for g in generated})}


def run(*, seed: int = 7, steps: int = 60, batch_size: int = 16,
        max_new_tokens: int = 128) -> dict[str, Any]:
    from .toy_generation import greedy_generate

    if min(steps, batch_size, max_new_tokens) < 1:
        raise ValueError("steps, batch size and generation cap must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train_rows, validation = bundle["train"], bundle["validation"]
    del bundle  # No test/diagnostic inspection, fitting, scoring or selection in this follow-up.
    train_sources = [corpus.model_inputs(r) for r in train_rows]
    validation_sources = [corpus.model_inputs(r) for r in validation]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in train_sources])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train_rows])
    torch.manual_seed(seed)
    initial = ToyModel("C", tensorizer, vocabulary)
    fixed = predict_concepts(initial, train_sources, tensorizer, batch_size).mean(0, keepdim=True)
    generator = torch.Generator().manual_seed(seed)
    schedule = torch.randint(len(train_rows), (steps, batch_size), generator=generator).tolist()
    report: dict[str, Any] = {
        "version": "concept-content-controls-1", "seed": seed, "steps": steps,
        "batch_size": batch_size, "max_new_tokens": max_new_tokens,
        "device": "cpu", "threads": 1, "torch": torch.__version__,
        "train_rows": len(train_rows), "validation_rows": len(validation), "test_evaluated": False,
        "initial_state_sha256": state_digest(initial),
        "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "constant_definition": "detached mean of INITIAL model predictions over train sources; frozen throughout training",
        "constant": fixed.tolist(),
        "majority": majority_baseline([r["targets"] for r in train_rows],
                                      [r["targets"] for r in validation], vocabulary),
        "training": {}, "validation": {}, "generation": {},
        "scope": "single-seed post-review validation diagnostics; no test selection or semantic superiority claim"}
    models = {}
    for condition in ("predicted", "constant"):
        print(f"training {condition}: {steps} CPU steps", flush=True)
        models[condition], report["training"][condition] = train_control(
            initial, condition, schedule, train_rows, tensorizer, vocabulary, corpus, strict, fixed)
    normal, constant = models["predicted"], models["constant"]
    probabilities = predict_concepts(normal, validation_sources, tensorizer, batch_size)
    final_mean = predict_concepts(normal, train_sources, tensorizer, batch_size).mean(0, keepdim=True)
    report["predicted_auxiliary_validation"] = evaluate(
        normal, validation, tensorizer, vocabulary, corpus, strict, batch_size)
    # Reversal is global to the fixed validation order, not to each mini-batch.
    controls = {"predicted": (normal, probabilities),
                "final_train_mean": (normal, final_mean.expand(len(validation), -1)),
                "reversed": (normal, probabilities.flip(0)),
                "constant_retrained": (constant, fixed.expand(len(validation), -1))}
    for name, (model, conditioning) in controls.items():
        report["validation"][name] = conditional_loss(
            model, validation, conditioning, tensorizer, corpus, strict, batch_size)
        generated = [asdict(g) for g in greedy_generate(
            model.decoder, conditioning, max_new_tokens=max_new_tokens)]
        report["generation"][name] = {
            **generation_metrics(generated, [r["targets"]["text"] for r in validation]),
            "examples": [{"source": s, "authored_reference": r["targets"]["text"], **g}
                         for s, r, g in zip(validation_sources, validation, generated)]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; choose a new path")
    report = run(seed=args.seed, steps=args.steps, batch_size=args.batch_size,
                 max_new_tokens=args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
