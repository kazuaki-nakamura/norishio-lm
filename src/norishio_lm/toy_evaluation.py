"""Issue 7 calibrated validation and reloadable local CPU experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary
from .concept_metrics import concept_metrics, majority_predictions, permutation_diagnostics
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_checkpoint import load_checkpoint, save_checkpoint
from .toy_controls import (conditional_loss, generation_metrics, predict_concepts,
                           state_digest, train_control)
from .toy_experiment import ToyModel, fixture_module, pad
from .toy_generation import greedy_generate


def concept_ids(probabilities: torch.Tensor, vocabulary: ConceptVocabulary) -> dict[str, torch.Tensor]:
    """Named groups retain ordered operator IDs; never merge negation scopes."""
    if probabilities.ndim != 2 or probabilities.shape[1] != sum(len(vocabulary.fields[f]) + 1 for f in CONCEPT_FIELDS):
        raise ValueError("concept probability dimensions do not match vocabulary")
    result = {}
    offset = 0
    for field in CONCEPT_FIELDS:
        width = len(vocabulary.fields[field]) + 1
        result[field] = probabilities[:, offset:offset + width].argmax(-1)
        offset += width
    if probabilities.ndim != 2 or offset != probabilities.shape[1]:
        raise ValueError("concept probability dimensions do not match vocabulary")
    return result


def parameter_changes(before: Mapping[str, torch.Tensor], model: ToyModel) -> dict[str, Any]:
    """End-state changes, not an estimate of how many optimizer operations ran."""
    by_tensor = {}
    for name, value in model.named_parameters():
        changed = int((before[name] != value.detach()).sum())
        by_tensor[name] = {"total": value.numel(), "changed_elements": changed}
    return {"registered": sum(v["total"] for v in by_tensor.values()),
            "changed_elements": sum(v["changed_elements"] for v in by_tensor.values()),
            "elements_in_changed_tensors": sum(v["total"] for v in by_tensor.values() if v["changed_elements"]),
            "definition": "exact initial versus final parameter differences; transient changes that cancel are not counted",
            "by_tensor": by_tensor}


def length_stats(rows: list[dict[str, Any]], strict: Any) -> dict[str, Any]:
    lengths = [len(strict.target_history(r)["input_ids"]) for r in rows]
    return {"rows": len(rows), "minimum": min(lengths), "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths), "unpadded_tokens": sum(lengths),
            "source_prefix": False}


def readable_concepts(probabilities: torch.Tensor, vocabulary: ConceptVocabulary) -> list[dict[str, Any]]:
    ids = concept_ids(probabilities, vocabulary)
    return [{field: {"id": int(ids[field][i]),
                     "value": {n: value for value, n in vocabulary.fields[field].items()}.get(int(ids[field][i]))}
             for field in CONCEPT_FIELDS} for i in range(len(probabilities))]


@torch.no_grad()
def score_condition(model: ToyModel, rows: list[dict[str, Any]], probabilities: torch.Tensor,
                    tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
                    corpus: Any, strict: Any, predicted: torch.Tensor | None = None) -> dict[str, Any]:
    model.eval()
    generated = [asdict(row) for row in greedy_generate(model.decoder, probabilities, 128)]
    concepts = readable_concepts(probabilities, vocabulary)
    normal_concepts = readable_concepts(predicted if predicted is not None else probabilities, vocabulary)
    return {"lm": conditional_loss(model, rows, probabilities, tensorizer, corpus, strict),
            "generation": generation_metrics(generated, [r["targets"]["text"] for r in rows]),
            "examples": [{"source": corpus.model_inputs(row),
                          "source_predicted_concepts": original,
                          "decoder_conditioning": conditioning,
                          "decoder_probabilities": probabilities[i].tolist(),
                          "generation": {**g, "stop_reason": "eos" if g["ended_eos"] else "max_new_tokens"},
                          "authored_reference": row["targets"]["text"]}
                         for i, (row, original, conditioning, g) in enumerate(zip(rows, normal_concepts, concepts, generated))]}


@torch.no_grad()
def reload_check(path: Path, model: ToyModel, tensorizer: SemanticTensorizer,
                 vocabulary: ConceptVocabulary, rows: list[dict[str, Any]],
                 corpus: Any, strict: Any, constant: torch.Tensor | None) -> dict[str, Any]:
    restored = load_checkpoint(path, expected_dataset_version=corpus.VERSION,
                               expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    sources = [corpus.model_inputs(r) for r in rows]
    before = predict_concepts(model, sources, tensorizer)
    after = predict_concepts(restored.model, sources, restored.tensorizer)
    if constant is not None:
        before = constant.expand(len(rows), -1)
        after = restored.constant.expand(len(rows), -1)
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    original_logits = model.decoder.decode_with_concept_intervention(ids, before)["logits"]
    loaded_logits = restored.model.decoder.decode_with_concept_intervention(ids, after)["logits"]
    original_generation = greedy_generate(model.decoder, before, 128)
    loaded_generation = greedy_generate(restored.model.decoder, after, 128)
    result = {"rows": len(rows), "logits_equal": torch.equal(original_logits, loaded_logits),
              "concepts_equal": torch.equal(before, after),
              "generation_equal": original_generation == loaded_generation,
              "model_state_equal": state_digest(model) == state_digest(restored.model)}
    if not all(result[k] for k in result if k != "rows"):
        raise RuntimeError("checkpoint reload changed model outputs")
    return result


def run(out: Path) -> dict[str, Any]:
    """Fixed seed7/60step scope; existing outputs are never overwritten."""
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train_rows, validation = bundle["train"], bundle["validation"]
    del bundle
    train_sources = [corpus.model_inputs(r) for r in train_rows]
    validation_sources = [corpus.model_inputs(r) for r in validation]
    targets = [r["targets"] for r in validation]
    tensorizer = SemanticTensorizer.fit([source_record(source) for source in train_sources])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train_rows])
    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary)
    before = {name: value.detach().clone() for name, value in initial.named_parameters()}
    fixed = predict_concepts(initial, train_sources, tensorizer).mean(0, keepdim=True)
    schedule = torch.randint(len(train_rows), (60, 16), generator=torch.Generator().manual_seed(7)).tolist()
    permutation_seed = 17
    permutation = torch.randperm(len(validation), generator=torch.Generator().manual_seed(permutation_seed))
    report: dict[str, Any] = {"version": "issue7-evaluation-1", "seed": 7, "steps": 60,
        "batch_size": 16, "max_new_tokens": 128, "device": "cpu", "threads": 1,
        "optimizer": {"name": "Adam", "learning_rate": 0.003, "clip_grad_norm": 1.0},
        "loss_weights": {"lm": 1.0, "sense": 1.0, "sememe": 1.0, "concept": 1.0},
        "torch": torch.__version__, "dataset_version": corpus.VERSION, "test_evaluated": False,
        "initial_state_sha256": state_digest(initial),
        "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "train_lengths": length_stats(train_rows, strict), "validation_lengths": length_stats(validation, strict),
        "sampled_training_lengths": length_stats([train_rows[i] for group in schedule for i in group], strict),
        "permutation": {"seed": permutation_seed, "indices": permutation.tolist(),
                        **permutation_diagnostics(targets, permutation.tolist(), vocabulary)},
        "majority": concept_metrics(majority_predictions([r["targets"] for r in train_rows], vocabulary,
                                                         len(validation)), targets, vocabulary),
        "training": {}, "concept_metrics": {}, "checkpoints": {}, "conditions": {}}
    models = {}
    predictions = {}
    for condition in ("predicted", "constant"):
        print(f"training {condition}: 60 CPU steps", flush=True)
        model, details = train_control(initial, condition, schedule, train_rows, tensorizer,
                                       vocabulary, corpus, strict, fixed)
        models[condition] = model
        details["parameter_updates"] = parameter_changes(before, model)
        report["training"][condition] = details
        probabilities = predict_concepts(model, validation_sources, tensorizer)
        predictions[condition] = probabilities
        report["concept_metrics"][condition] = concept_metrics(concept_ids(probabilities, vocabulary), targets, vocabulary)
        checkpoint = out / f"{condition}.pt"
        frozen = fixed if condition == "constant" else None
        save_checkpoint(checkpoint, model, tensorizer, vocabulary, dataset_version=corpus.VERSION, constant=frozen)
        report["checkpoints"][condition] = {"file": checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "reload": reload_check(checkpoint, model, tensorizer, vocabulary, validation, corpus, strict, frozen)}
    normal = models["predicted"]
    probabilities = predictions["predicted"]
    final_mean = predict_concepts(normal, train_sources, tensorizer).mean(0, keepdim=True)
    hard = torch.cat([torch.nn.functional.one_hot(ids, len(vocabulary.fields[field]) + 1).float()
                      for field, ids in concept_ids(probabilities, vocabulary).items()], dim=-1)
    conditions = {"soft": (normal, probabilities), "hard": (normal, hard),
                  "zero": (normal, torch.zeros_like(probabilities)),
                  "train_mean": (normal, final_mean.expand(len(validation), -1)),
                  "permuted": (normal, probabilities[permutation]),
                  "constant_retrained": (models["constant"], fixed.expand(len(validation), -1))}
    for name, (model, conditioning) in conditions.items():
        own_predictions = predictions["constant" if name == "constant_retrained" else "predicted"]
        report["conditions"][name] = score_condition(model, validation, conditioning, tensorizer,
                                                      vocabulary, corpus, strict, own_predictions)
    return report


def replay(path: Path, out: Path) -> dict[str, Any]:
    """Re-evaluate the same saved model; no training or refitting on validation."""
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    loaded = load_checkpoint(path, expected_dataset_version=corpus.VERSION)
    rows = corpus.build()["validation"]
    predicted = predict_concepts(loaded.model, [corpus.model_inputs(r) for r in rows], loaded.tensorizer)
    conditioning = loaded.constant.expand(len(rows), -1) if loaded.constant is not None else predicted
    return {"checkpoint": str(path), "dataset_version": loaded.dataset_version,
            "test_evaluated": False, "trained": False,
            "concept_metrics": concept_metrics(concept_ids(predicted, loaded.vocabulary),
                                               [r["targets"] for r in rows], loaded.vocabulary),
            "validation": score_condition(loaded.model, rows, conditioning, loaded.tensorizer,
                                          loaded.vocabulary, corpus, strict, predicted)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    result = replay(args.restore, args.out_dir) if args.restore else run(args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
