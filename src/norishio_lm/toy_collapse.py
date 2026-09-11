"""Fixed-budget validation-only diagnosis of source/concept/decoder boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .collapse_metrics import concept_head_stats, representation_stats
from .concept_metrics import concept_metrics, majority_predictions, permutation_diagnostics
from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, encode_targets
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_checkpoint import save_checkpoint
from .toy_controls import state_digest, train_control
from .toy_evaluation import concept_ids, reload_check, score_condition
from .toy_experiment import ToyModel, fixture_module
from .toy_probe import train_probe


@torch.no_grad()
def source_encodings(model: ToyModel, sources: Sequence[dict[str, str]],
                     tensorizer: SemanticTensorizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept only source objects; neither labels nor reference histories enter."""
    if not sources or model.pathway != "C":
        raise ValueError("nonempty sources and strict C model required")
    model.eval()
    latents, probabilities = [], []
    for start in range(0, len(sources), 16):
        semantic = tensorizer.encode([source_record(s) for s in sources[start:start + 16]])
        latent = model.encoder(semantic).fused
        latents.append(latent)
        probabilities.append(model.decoder.bottleneck(latent).probabilities())
    return torch.cat(latents).detach(), torch.cat(probabilities).detach()


def oracle_probabilities(concepts: Sequence[Mapping[str, Any]], vocabulary: ConceptVocabulary) -> torch.Tensor:
    """Explicit authored-gold oracle; reject partial/unknown frames, no text input."""
    if not concepts or any(set(c) != set(CONCEPT_FIELDS) for c in concepts):
        raise ValueError("oracle needs exactly the seven concept fields")
    targets = encode_targets([{"concept": c} for c in concepts], vocabulary)
    if not all(bool(mask.all()) for mask in targets.masks.values()):
        raise ValueError("oracle requires complete known concept frames")
    return torch.cat([torch.nn.functional.one_hot(targets.fields[f], len(vocabulary.fields[f]) + 1).float()
                      for f in CONCEPT_FIELDS], dim=-1)


def source_conditions(predicted: torch.Tensor, train_predictions: torch.Tensor,
                      permutation: Sequence[int]) -> dict[str, torch.Tensor]:
    """Normal interventions have no gold argument; permutation is a bijection."""
    if sorted(permutation) != list(range(len(predicted))):
        raise ValueError("permutation must be a bijection")
    return {"predicted": predicted, "train_mean": train_predictions.mean(0, keepdim=True).expand_as(predicted),
            "permuted": predicted[list(permutation)]}


def run(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, validation = bundle["train"], bundle["validation"]
    del bundle
    train_sources = [corpus.model_inputs(r) for r in train]
    validation_sources = [corpus.model_inputs(r) for r in validation]
    train_targets, validation_targets = [r["targets"] for r in train], [r["targets"] for r in validation]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in train_sources])
    vocabulary = ConceptVocabulary.fit(train_targets)
    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary)
    _, initial_predictions = source_encodings(initial, train_sources, tensorizer)
    schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    print("normal C: seed7, 600 CPU updates", flush=True)
    model, training = train_control(initial, "predicted", schedule, train, tensorizer,
                                    vocabulary, corpus, strict, initial_predictions.mean(0, keepdim=True))
    model.eval().requires_grad_(False)
    frozen_hash = state_digest(model)
    train_latent, train_probs = source_encodings(model, train_sources, tensorizer)
    valid_latent, valid_probs = source_encodings(model, validation_sources, tensorizer)
    print("frozen linear probe: seed7, 300 train-only updates", flush=True)
    probe, probe_details = train_probe(train_latent, train_targets, vocabulary,
                                      seed=7, steps=300, learning_rate=.01)
    if state_digest(model) != frozen_hash:
        raise RuntimeError("probe modified frozen source model")
    with torch.no_grad():
        probe.eval()
        probe_ids = {f: v.argmax(-1) for f, v in probe(valid_latent).items()}
    permutation = torch.randperm(len(validation), generator=torch.Generator().manual_seed(17)).tolist()
    conditions = source_conditions(valid_probs, train_probs, permutation)
    conditions["gold_oracle"] = oracle_probabilities([r["concept"] for r in validation_targets], vocabulary)
    report: dict[str, Any] = {
        "version": "issue9-collapse-1", "dataset_version": corpus.VERSION,
        "split": "validation", "test_evaluated": False, "rows": {"train": len(train), "validation": len(validation)},
        "budget": {"model_seed": 7, "model_steps": 600, "batch_size": 16,
                   "probe_seed": 7, "probe_steps": 300, "permutation_seed": 17,
                   "max_new_tokens": 128, "device": "cpu", "threads": 1},
        "torch": torch.__version__, "training": training,
        "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "frozen_model_state_sha256": frozen_hash, "probe_model_unchanged": True,
        "majority": concept_metrics(majority_predictions(train_targets, vocabulary, len(validation)), validation_targets, vocabulary),
        "probe": {"training": probe_details, "metrics": concept_metrics(probe_ids, validation_targets, vocabulary)},
        "encoder": representation_stats(valid_latent),
        "concept_head": concept_head_stats(valid_probs, validation_targets, vocabulary, permutation),
        "permuted_concept_metrics": concept_metrics(concept_ids(valid_probs[permutation], vocabulary), validation_targets, vocabulary),
        "permutation": {"seed": 17, "indices": permutation,
                        **permutation_diagnostics(validation_targets, permutation, vocabulary)},
        "conditions": {}}
    for name, probabilities in conditions.items():
        result = score_condition(model, validation, probabilities, tensorizer, vocabulary, corpus, strict, valid_probs)
        result["oracle"] = name == "gold_oracle"
        result["conditioning_diversity"] = representation_stats(probabilities)
        for example in result["examples"]:
            example["generation"]["raw_bytes"] = [t - 4 for t in example["generation"]["token_ids"] if 4 <= t < 260]
        report["conditions"][name] = result
    checkpoint = out / "normal600.pt"
    save_checkpoint(checkpoint, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
    report["checkpoint"] = {"file": checkpoint.name, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "reload": reload_check(checkpoint, model, tensorizer, vocabulary, validation, corpus, strict, None)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
