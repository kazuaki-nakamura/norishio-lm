"""Fixed three-seed joint pair head control; no test tuning."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from .concept_model import ConceptVocabulary, encode_targets
from .explicit_slot_model import slot_head_loss
from .head_joint_metrics import head_joint_metrics
from .head_following_diagnostics import evaluate_condition, pair_following
from .local_slot_generation import greedy_generate_local
from .local_slot_checkpoint import load_local_checkpoint
from .pair_slot_model import PairSlotModel, pair_head_loss, source_pair_distributions
from .pair_slot_checkpoint import save_pair_checkpoint, load_pair_checkpoint
from .pair_head_metrics import pair_head_metrics
from .pair_metrics import pair_support
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest
from .toy_experiment import ToyModel, fixture_module, batch, pad
from .toy_explicit_slots import gold_heads
from .toy_head_following import score_seed, SEEDS
from .toy_local_slots import checked_report
from .toy_slots import score_slots

PRIOR_SHA = "964ab4d7dd7fc8b2fdddac816af51ac2ea8ab28a806a33dff32e40631250170f"
PRIOR24_SHA = "eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation_cases(old: torch.Tensor, marginals: torch.Tensor,
                     gold: torch.Tensor) -> dict[str, torch.Tensor]:
    """Keep gold replacement explicitly outside normal source inference."""
    if old.shape != (len(marginals), 33) or marginals.shape != gold.shape or marginals.shape[1:] != (10,):
        raise ValueError("expected aligned old33 and marginal/gold10")
    return {"predicted": torch.cat((old, marginals), -1),
            "participant_gold": torch.cat((old, gold[:, :5], marginals[:, 5:]), -1),
            "time_gold": torch.cat((old, marginals[:, :5], gold[:, 5:]), -1),
            "both_gold": torch.cat((old, gold), -1)}


@torch.no_grad()
def factorial(model: PairSlotModel, old: torch.Tensor, rows: list[dict],
              vocabulary: ConceptVocabulary, corpus: Any, support: dict) -> dict:
    cells = {}
    targets = [r["targets"] for r in rows]
    values = support["class_values"]
    for p in range(5):
        for t in range(5):
            assigned = torch.cat((F.one_hot(torch.full((len(rows),), p), 5),
                                  F.one_hot(torch.full((len(rows),), t), 5)), -1).float()
            generated = [asdict(g) for g in greedy_generate_local(model.decoder, torch.cat((old, assigned), -1))]
            counterfactual = [{"concept": {**target["concept"],
                "participant": values["participant"][p], "time": values["time"][t]}} for target in targets]
            scores = score_slots(generated, counterfactual, corpus.seed_data())
            train_count = support["train_table5x5"][p][t]
            validation_count = support["validation_table5x5"][p][t]
            both = pair_following(generated, counterfactual, vocabulary, corpus.seed_data())[f"{p},{t}"]["both_correct"]
            cells[f"{p},{t}"] = {"assignment": {"participant": values["participant"][p], "time": values["time"][t]},
                "rows": len(rows), "participant_correct": scores["fields"]["participant"]["correct"],
                "time_correct": scores["fields"]["time"]["correct"], "both_correct": both,
                "fullframe_correct": scores["fullframe_correct"], "parseable": scores["parseable_count"],
                "train_count": train_count, "validation_count": validation_count,
                "pair_status": "train_seen" if train_count else "validation_unseen" if validation_count else "neither_train_nor_validation",
                "generations": generated}
        print(f"factorial participant {p}: five assignments", flush=True)
    return {"counterfactual_assignment_oracle": True, "old_concepts": "predicted", "cells": cells}


@torch.no_grad()
def evaluate_b(model: PairSlotModel, train: list[dict], rows: list[dict], tensorizer: SemanticTensorizer,
               vocabulary: ConceptVocabulary, corpus: Any, strict: Any, support: dict) -> dict:
    splits = {}
    validation_old = validation_marginals = None
    for split, part in (("train", train), ("validation", rows)):
        old, independent, pair, marginals = source_pair_distributions(model, [corpus.model_inputs(r) for r in part], tensorizer)
        targets = [r["targets"] for r in part]
        splits[split] = {"rows": len(part), "independent": head_joint_metrics(independent, targets, vocabulary),
                        "pair": pair_head_metrics(pair, targets, vocabulary, support["train_table5x5"]),
                        "marginals": head_joint_metrics(marginals, targets, vocabulary),
                        "pair_probabilities": pair.tolist(), "marginal_probabilities": marginals.tolist()}
        if split == "validation":
            validation_old, validation_marginals = old, marginals
    assert validation_old is not None and validation_marginals is not None
    conditions = {}
    for name, probs in generation_cases(validation_old, validation_marginals,
                                         gold_heads([r["targets"] for r in rows], vocabulary)).items():
        print(f"B condition {name}", flush=True)
        conditions[name] = {**evaluate_condition(model, rows, probs, tensorizer, vocabulary, corpus, strict),
                            "head_oracle": name != "predicted", "decoder_probabilities": probs.tolist()}
    return {"heads": splits, "conditions": conditions,
            "factorial": factorial(model, validation_old, rows, vocabulary, corpus, support)}


@torch.no_grad()
def replay_checkpoint(model: PairSlotModel, restored: PairSlotModel, rows: list[dict],
                      tensorizer: SemanticTensorizer, corpus: Any, strict: Any) -> dict:
    sources = [corpus.model_inputs(r) for r in rows]
    a, b = source_pair_distributions(model, sources, tensorizer), source_pair_distributions(restored, sources, tensorizer)
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    x, y = torch.cat((a[0], a[3]), -1), torch.cat((b[0], b[3]), -1)
    checks = {"state": state_digest(model) == state_digest(restored),
              **{name: torch.equal(left, right) for name, left, right in zip(("old", "independent", "pair", "marginals"), a, b)},
              "logits": torch.equal(model.decoder.decode_with_concept_intervention(ids, x)["logits"],
                                    restored.decoder.decode_with_concept_intervention(ids, y)["logits"]),
              "greedy": greedy_generate_local(model.decoder, x) == greedy_generate_local(restored.decoder, y)}
    if not all(checks.values()):
        raise RuntimeError("pair checkpoint replay differs")
    return checks


def scalar_means(values: list[Any]) -> Any:
    """Average only corresponding scalar leaves, retaining null as missing."""
    if all(isinstance(x, dict) for x in values):
        return {k: scalar_means([v[k] for v in values]) for k in values[0]
                if all(k in v for v in values) and isinstance(values[0][k], (dict, int, float))
                and not isinstance(values[0][k], bool)}
    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in values):
        return sum(values) / len(values)
    return None


def run(baseline_report: Path, out: Path) -> dict:
    prior = checked_report(baseline_report, PRIOR_SHA)
    prior24_path = baseline_report.parent.parent / "issue24-seed7-v1" / "report.json"
    prior24 = checked_report(prior24_path, PRIOR24_SHA)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, rows = bundle["train"], bundle["validation"]
    del bundle
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train])
    support = pair_support([r["targets"] for r in train], [r["targets"] for r in rows], vocabulary)
    if support != prior["support"]:
        raise RuntimeError("dataset pair support differs")
    expected = dict(expected_dataset_version=corpus.VERSION, expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    baseline = {}
    for seed in SEEDS:
        saved = prior24_path.parent / "D.pt" if seed == 7 else baseline_report.parent / f"D-seed{seed}.pt"
        metadata = prior24["arms"]["D"]["checkpoint"] if seed == 7 else prior["seeds"][str(seed)]["checkpoint"]
        if _sha(saved) != metadata["sha256"]:
            raise ValueError("historical checkpoint hash differs")
        model = load_local_checkpoint(saved, **expected).model
        before = state_digest(model)
        if before != metadata["state_sha256"]:
            raise RuntimeError("historical state differs")
        print(f"A replay seed{seed}", flush=True)
        evaluation = score_seed(model, rows, tensorizer, vocabulary, corpus, strict)
        if json.loads(json.dumps(evaluation)) != prior["seeds"][str(seed)]["evaluation"] or before != state_digest(model):
            raise RuntimeError("A historical evaluation/state does not replay")
        baseline[str(seed)] = {"evaluation": evaluation, "state_sha256": before, "historical_replay_equal": True}
    seeds = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
        model = PairSlotModel(initial, vocabulary, new_seed=20, pair_seed=28)
        if sum(p.numel() for p in model.parameters()) != 43514:
            raise RuntimeError("pair model parameter count differs")
        initial_hash = state_digest(model)
        schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(seed)).tolist()
        schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
        expected_schedule = prior24["schedule_sha256"] if seed == 7 else prior["seeds"][str(seed)]["schedule_sha256"]
        if schedule_hash != expected_schedule:
            raise RuntimeError("historical sampling schedule differs")
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        trace = []
        model.train()
        started = time.perf_counter()
        for step, indices in enumerate(schedule):
            part = [train[i] for i in indices]
            ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
            targets = [r["targets"] for r in part]
            optimizer.zero_grad(set_to_none=True)
            output, auxiliary, independent_logits, pair_logits = model(ids, semantic, labels=labels)
            base = model.decoder.losses(output, targets=encode_targets(targets, vocabulary), bottleneck_output=auxiliary)
            independent = slot_head_loss(independent_logits, targets, vocabulary)
            pair = pair_head_loss(pair_logits, targets, vocabulary)
            total = base + independent + pair
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                trace.append({"step": step + 1, "base": float(base.detach()), "independent_ce": float(independent.detach()),
                              "pair_ce": float(pair.detach()), "total": float(total.detach())})
            if (step + 1) % 200 == 0:
                print(f"B seed{seed} update{step + 1}", flush=True)
        seconds = time.perf_counter() - started
        evaluation = evaluate_b(model, train, rows, tensorizer, vocabulary, corpus, strict, support)
        saved = out / f"B-seed{seed}.pt"
        save_pair_checkpoint(saved, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
        restored = load_pair_checkpoint(saved, **expected).model
        seeds[str(seed)] = {"seed": seed, "initial_sha256": initial_hash, "schedule_sha256": schedule_hash,
            "training_seconds": seconds, "trace": trace, "evaluation": evaluation,
            "checkpoint": {"file": saved.name, "sha256": _sha(saved), "state_sha256": state_digest(model),
                           "reload": replay_checkpoint(model, restored, rows, tensorizer, corpus, strict)}}
    return {"version": "issue28-pair-head-1", "dataset_version": corpus.VERSION, "test_evaluated": False,
        "prior_report_sha256": PRIOR_SHA, "support": support, "A": baseline, "B": seeds,
        "budget": {"seeds": list(SEEDS), "extension_seed": 20, "pair_seed": 28, "steps": 600, "batch_size": 16,
                   "optimizer": "Adam", "lr": .003, "clip": 1., "cpu_threads": 1,
                   "base_loss_weights": [1, 1, 1, 1], "independent_ce_weights": [1, 1], "pair_ce_weight": 1.,
                   "A_parameters": 42689, "B_parameters": 43514, "added_parameters": 825},
        "means": {"n": 3, "aggregation": "unweighted arithmetic mean of scalar metrics; no best seed",
            "A": scalar_means([x["evaluation"] for x in baseline.values()]),
            "B": scalar_means([x["evaluation"] for x in seeds.values()])}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_report, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
