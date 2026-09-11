"""Matched initial-only/per-step concept conditioning on fixed authored data."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .concept_metrics import concept_metrics, permutation_diagnostics
from .concept_model import ConceptVocabulary, TinyConceptDecoder
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_checkpoint import save_checkpoint
from .toy_collapse import oracle_probabilities, source_conditions, source_encodings
from .toy_controls import state_digest, train_control
from .toy_evaluation import concept_ids, parameter_changes, reload_check, score_condition
from .toy_experiment import ToyModel, fixture_module, pad
from .toy_slots import score_slots


def decision_history(tokens: Sequence[int]) -> list[int]:
    """Prefix for each emitted decision, including cap rows; never a cap+1 step."""
    if not tokens:
        raise ValueError("at least one emitted decision required")
    return [1, *tokens[:-1]]


@torch.no_grad()
def conditioning_sensitivity(decoder: TinyConceptDecoder, histories: Sequence[Sequence[int]],
                             baseline: torch.Tensor, intervention: torch.Tensor) -> dict[str, Any]:
    """Hold a baseline self-generated history fixed; never run intervention feedback."""
    if not histories or len(histories) != len(baseline) or baseline.shape != intervention.shape:
        raise ValueError("one nonempty history per aligned concept pair is required")
    if any(not h or h[0] != 1 for h in histories):
        raise ValueError("histories must start at BOS")
    decoder.eval()
    ids = pad([list(h) for h in histories])
    a = decoder.decode_with_concept_intervention(ids, baseline)["logits"].double()
    b = decoder.decode_with_concept_intervention(ids, intervention)["logits"].double()
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    l1 = (a - b).abs().mean(-1)
    kl = (log_a.exp() * (log_a - log_b)).sum(-1).clamp_min(0)
    changed = a.argmax(-1) != b.argmax(-1)
    lengths = torch.tensor([len(h) for h in histories])
    positions = []
    for pos in range(ids.shape[1]):
        active = lengths > pos
        positions.append({"position": pos, "rows": int(active.sum()),
                          "mean_logit_l1": float(l1[active, pos].mean()),
                          "mean_kl": float(kl[active, pos].mean()),
                          "argmax_change_rate": float(changed[active, pos].double().mean())})
    return {"history": "fixed predicted greedy prefix; BOS plus self-emitted tokens",
            "l1_definition": "mean absolute logit difference across vocabulary then active rows",
            "kl_definition": "KL(baseline || intervention), natural log",
            "positions": positions}


def run(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, validation = bundle["train"], bundle["validation"]
    del bundle
    seed_grammar = corpus.seed_data()
    train_sources = [corpus.model_inputs(r) for r in train]
    sources = [corpus.model_inputs(r) for r in validation]
    train_targets, targets = [r["targets"] for r in train], [r["targets"] for r in validation]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in train_sources])
    vocabulary = ConceptVocabulary.fit(train_targets)
    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary)
    initial_hash = state_digest(initial)
    initial_parameters = {n: p.detach().clone() for n, p in initial.named_parameters()}
    _, init_probs = source_encodings(initial, train_sources, tensorizer)
    schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    permutation = torch.randperm(len(validation), generator=torch.Generator().manual_seed(17)).tolist()
    report: dict[str, Any] = {
        "version": "issue12-step-conditioning-1", "dataset_version": corpus.VERSION,
        "test_evaluated": False, "split": "validation", "torch": torch.__version__,
        "budget": {"seed": 7, "updates": 600, "batch_size": 16, "optimizer": "Adam",
                   "learning_rate": .003, "clip": 1., "all_four_loss_weights": 1.,
                   "threads": 1, "device": "cpu", "max_new_tokens": 128},
        "initial_state_sha256": initial_hash,
        "schedule_sha256": hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
        "permutation": {"seed": 17, "indices": permutation,
                        **permutation_diagnostics(targets, permutation, vocabulary)}, "arms": {}}
    for mode in ("initial_only", "per_step_additive"):
        arm_initial = deepcopy(initial)
        arm_initial.decoder.conditioning_mode = mode
        if state_digest(arm_initial) != initial_hash:
            raise RuntimeError("common initial parameter values differ")
        print(f"{mode}: seed7, 600 CPU updates", flush=True)
        model, training = train_control(arm_initial, "predicted", schedule, train, tensorizer,
                                        vocabulary, corpus, strict, init_probs.mean(0, keepdim=True))
        training["parameter_updates"] = parameter_changes(initial_parameters, model)
        training["common_initial_weights_equal"] = True
        training["additional_parameters"] = 0
        _, train_probs = source_encodings(model, train_sources, tensorizer)
        _, predicted = source_encodings(model, sources, tensorizer)
        conditions = source_conditions(predicted, train_probs, permutation)
        conditions["gold_oracle"] = oracle_probabilities([t["concept"] for t in targets], vocabulary)
        arm = {"mode": mode, "training": training,
               "concept_metrics": concept_metrics(concept_ids(predicted, vocabulary), targets, vocabulary),
               "conditions": {}, "sensitivity": {}}
        for name, probs in conditions.items():
            result = score_condition(model, validation, probs, tensorizer, vocabulary, corpus, strict, predicted)
            result["oracle"] = name == "gold_oracle"
            for example in result["examples"]:
                generated = example["generation"]
                generated["raw_bytes"] = [t - 4 for t in generated["token_ids"] if 4 <= t < 260]
            result["slots"] = score_slots([e["generation"] for e in result["examples"]], targets, seed_grammar)
            arm["conditions"][name] = result
        histories = [decision_history(e["generation"]["token_ids"]) for e in arm["conditions"]["predicted"]["examples"]]
        for name in ("train_mean", "permuted", "gold_oracle"):
            arm["sensitivity"][name] = conditioning_sensitivity(model.decoder, histories, predicted, conditions[name])
        path = out / f"{mode}.pt"
        save_checkpoint(path, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
        arm["checkpoint"] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "reload": reload_check(path, model, tensorizer, vocabulary, validation, corpus, strict, None)}
        report["arms"][mode] = arm
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
