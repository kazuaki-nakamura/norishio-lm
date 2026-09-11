"""Fixed head-prediction, decoder-following and three-seed diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .concept_model import ConceptVocabulary, encode_targets
from .explicit_slot_model import source_distributions, slot_head_loss
from .head_joint_metrics import head_joint_metrics
from .head_following_diagnostics import (
    base_ablation_cases, confident_train_subsets, evaluate_condition, pair_following,
)
from .local_slot_model import DuplicateRemovedModel
from .local_slot_generation import greedy_generate_local
from .local_slot_checkpoint import save_local_checkpoint, load_local_checkpoint
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest
from .toy_experiment import ToyModel, fixture_module, batch, pad
from .toy_explicit_slots import gold_heads
from .toy_joint_slots import evaluate
from .toy_local_slots import checked_report

PRIOR_SHA = "eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8"
SEEDS = (7, 17, 29)


def checkpoint_replay(model: DuplicateRemovedModel, restored: DuplicateRemovedModel,
                      rows: list[dict], tensorizer: SemanticTensorizer, corpus: Any,
                      strict: Any) -> dict[str, bool]:
    sources = [corpus.model_inputs(r) for r in rows]
    a, b = source_distributions(model, sources, tensorizer), source_distributions(restored, sources, tensorizer)
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    with torch.no_grad():
        x, y = torch.cat(a, -1), torch.cat(b, -1)
        result = {"state": state_digest(model) == state_digest(restored),
                  "old_concepts": torch.equal(a[0], b[0]), "heads": torch.equal(a[1], b[1]),
                  "logits": torch.equal(model.decoder.decode_with_concept_intervention(ids, x)["logits"],
                                        restored.decoder.decode_with_concept_intervention(ids, y)["logits"]),
                  "greedy": greedy_generate_local(model.decoder, x) == greedy_generate_local(restored.decoder, y)}
    if not all(result.values()):
        raise RuntimeError("checkpoint replay differs")
    return result


def score_seed(model: DuplicateRemovedModel, rows: list[dict], tensorizer: SemanticTensorizer,
               vocabulary: ConceptVocabulary, corpus: Any, strict: Any) -> dict[str, Any]:
    old, heads = source_distributions(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    targets = [r["targets"] for r in rows]
    conditions = {}
    for name, hp in (("predicted", heads), ("both_gold", gold_heads(targets, vocabulary))):
        print(f"seed condition {name}", flush=True)
        conditions[name] = evaluate_condition(model, rows, torch.cat((old, hp), -1),
                                              tensorizer, vocabulary, corpus, strict)
    return {"head_metrics": head_joint_metrics(heads, targets, vocabulary), "conditions": conditions}


def seed_means(seeds: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for condition in ("predicted", "both_gold"):
        collected = []
        for record in seeds.values():
            value = record["evaluation"]["conditions"][condition]
            generation, slots = value["generation"], value["slots"]
            collected.append({"lm": value["lm"]["lm"], "eos_rate": generation["eos_rate"],
                              "valid_utf8_rate": generation["valid_utf8_rate"],
                              "unique_sequences": generation["unique_sequences"],
                              "participant_correct": slots["fields"]["participant"]["correct"],
                              "time_correct": slots["fields"]["time"]["correct"],
                              "both_correct": value["both_slots"]["correct"],
                              "fullframe_correct": slots["fullframe_correct"],
                              "head_joint_correct": record["evaluation"]["head_metrics"]["joint_exact"]["correct"]})
        result[condition] = {key: sum(v[key] for v in collected) / len(collected) for key in collected[0]}
    head_means = {}
    for field in ("participant", "time"):
        values = [record["evaluation"]["head_metrics"]["per_head"][field] for record in seeds.values()]
        head_means[field] = {key: sum(v[key] for v in values) / len(values)
                             for key in ("accuracy", "balanced_accuracy", "brier", "nll")}
        head_means[field]["ece"] = sum(v["ece"]["value"] for v in values) / len(values)
    return {"n": len(seeds), "aggregation": "unweighted arithmetic mean; counts per150 validation rows",
            "conditions": result, "per_head": head_means}


def run(baseline_report: Path, out: Path) -> dict[str, Any]:
    prior = checked_report(baseline_report, PRIOR_SHA)
    path = baseline_report.parent / "D.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != prior["arms"]["D"]["checkpoint"]["sha256"]:
        raise ValueError("historical checkpoint hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, rows = bundle["train"], bundle["validation"]
    del bundle
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train])
    expected = dict(expected_dataset_version=corpus.VERSION, expected_tensorizer=tensorizer,
                    expected_vocabulary=vocabulary)
    frozen = load_local_checkpoint(path, **expected).model
    frozen_hash = state_digest(frozen)
    if frozen_hash != prior["arms"]["D"]["checkpoint"]["state_sha256"] or not frozen.decoder.local:
        raise RuntimeError("historical D state/mode differs")
    print("replaying complete seed7 D", flush=True)
    replay = evaluate(frozen, rows, tensorizer, vocabulary, corpus, strict,
                      prior["support"]["validation_seen"], generate=greedy_generate_local)
    if json.loads(json.dumps(replay)) != prior["evaluations"]["D"]:
        raise RuntimeError("historical D evaluation does not replay")

    targets = [r["targets"] for r in rows]
    pair_scores = {name: pair_following(replay["conditions"][name]["generations"], targets,
                                       vocabulary, corpus.seed_data())
                   for name in ("predicted", "participant_gold", "time_gold", "both_gold")}
    factorial = {}
    for key, cell in replay["factorial"]["cells"].items():
        p, t = map(int, key.split(","))
        train_count = prior["support"]["train_table5x5"][p][t]
        val_count = prior["support"]["validation_table5x5"][p][t]
        factorial[key] = {**cell, "train_count": train_count, "validation_count": val_count,
                          "pair_status": "train_seen" if train_count else
                              "validation_unseen" if val_count else "neither_train_nor_validation"}

    train_old, train_heads = source_distributions(frozen, [corpus.model_inputs(r) for r in train], tensorizer)
    subsets = confident_train_subsets(train_heads, [r["targets"] for r in train], vocabulary, threshold=.8)
    subset_scores = {}
    for name, indices in subsets.items():
        print(f"train subset {name}: {len(indices)} rows", flush=True)
        subset_scores[name] = {"indices": indices, "split": "train", "threshold_each_head": .8,
            "evaluation": evaluate_condition(frozen, [train[i] for i in indices],
                torch.cat((train_old, train_heads), -1)[indices], tensorizer, vocabulary, corpus, strict)}

    old, heads = source_distributions(frozen, [corpus.model_inputs(r) for r in rows], tensorizer)
    ablations = {}
    for name, probs in base_ablation_cases(old, heads, gold_heads(targets, vocabulary), vocabulary).items():
        print(f"base intervention {name}", flush=True)
        ablations[name] = evaluate_condition(frozen, rows, probs, tensorizer, vocabulary, corpus, strict)
    for name, original in (("full_predicted", "predicted"), ("full_both_gold", "both_gold")):
        if (ablations[name]["generations"] != replay["conditions"][original]["generations"]
                or ablations[name]["lm"] != replay["conditions"][original]["groups"]["all"]["lm"]):
            raise RuntimeError("new scorer does not preserve full-condition historical results")
    if state_digest(frozen) != frozen_hash:
        raise RuntimeError("frozen seed7 model was mutated")
    seeds = {"7": {"seed": 7, "trained_this_run": False, "state_sha256": frozen_hash,
                    "evaluation": score_seed(frozen, rows, tensorizer, vocabulary, corpus, strict)}}
    if state_digest(frozen) != frozen_hash:
        raise RuntimeError("seed7 scoring mutated the frozen model")
    for seed in SEEDS[1:]:
        torch.manual_seed(seed)
        initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
        model = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=True)
        if sum(p.numel() for p in model.parameters()) != 42689:
            raise RuntimeError("parameter count changed")
        initial_hash = state_digest(model)
        schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(seed)).tolist()
        schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        trace = []
        model.train()
        started = time.perf_counter()
        print(f"training D seed{seed}:600 CPU updates", flush=True)
        for step, indices in enumerate(schedule):
            part = [train[i] for i in indices]
            ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
            batch_targets = [r["targets"] for r in part]
            optimizer.zero_grad(set_to_none=True)
            output, auxiliary, head_logits = model(ids, semantic, labels=labels)
            base = model.decoder.losses(output, targets=encode_targets(batch_targets, vocabulary), bottleneck_output=auxiliary)
            head = slot_head_loss(head_logits, batch_targets, vocabulary)
            total = base + head
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                trace.append({"step": step + 1, "base": float(base.detach()),
                              "head_ce": float(head.detach()), "total": float(total.detach())})
            if (step + 1) % 200 == 0:
                print(f"seed{seed} update{step + 1}", flush=True)
        seconds = time.perf_counter() - started
        scored = score_seed(model, rows, tensorizer, vocabulary, corpus, strict)
        saved = out / f"D-seed{seed}.pt"
        save_local_checkpoint(saved, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
        restored = load_local_checkpoint(saved, **expected).model
        seeds[str(seed)] = {"seed": seed, "trained_this_run": True, "extension_seed": 20,
                            "initial_sha256": initial_hash, "schedule_sha256": schedule_hash,
                            "training_seconds": seconds, "trace": trace, "evaluation": scored,
                            "checkpoint": {"file": saved.name, "sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                                "state_sha256": state_digest(model),
                                "reload": checkpoint_replay(model, restored, rows, tensorizer, corpus, strict)}}
    return {"version": "issue26-head-following-1", "prior_report_sha256": PRIOR_SHA,
            "historical_replay_equal": True, "seed7_frozen_state_unchanged": True,
            "dataset_version": corpus.VERSION, "test_evaluated": False, "support": prior["support"],
            "budget": {**prior["budget"], "seeds": list(SEEDS), "extension_seed": 20},
            "seed7_pair_scores": pair_scores, "seed7_factorial": factorial,
            "train_subsets": subset_scores, "train_head_probabilities": train_heads.tolist(),
            "seed7_base_ablations": ablations, "seeds": seeds, "means": seed_means(seeds)}


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
