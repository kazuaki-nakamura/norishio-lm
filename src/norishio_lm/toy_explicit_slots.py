"""Historical A/B replay and a preregistered explicit slot-head comparison."""
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

from .collapse_metrics import concept_head_stats
from .concept_model import ConceptVocabulary, encode_targets
from .explicit_slot_model import ExplicitSlotModel, source_distributions, slot_head_loss
from .explicit_slot_checkpoint import save_explicit_checkpoint, load_explicit_checkpoint
from .span_metrics import span_metrics
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_checkpoint import load_checkpoint
from .toy_controls import state_digest, conditional_loss, generation_metrics
from .toy_evaluation import readable_concepts, parameter_changes
from .toy_experiment import ToyModel, batch, fixture_module, pad
from .toy_generation import greedy_generate
from .toy_slot_objective import evaluate_arm, INITIAL_SHA, SCHEDULE_SHA
from .toy_slot_retention import single_field_conditions
from .toy_collapse import oracle_probabilities
from .toy_slots import score_slots
from .toy_spans import authored_slot_spans

PRIOR_SHA = "9db04ef46231cfb79439f1dc4f90883567b3f80433a7b8fc295cb2f2fcc1a382"
FIELDS = ("participant", "time")


def gold_heads(targets: list[dict[str, Any]], vocabulary: ConceptVocabulary) -> torch.Tensor:
    encoded = encode_targets(targets, vocabulary)
    if any(len(vocabulary.fields[f]) != 5 or not bool(encoded.masks[f].all()) for f in FIELDS):
        raise ValueError("five known training classes required for each explicit oracle head")
    return torch.cat([F.one_hot(encoded.fields[f] - 1, 5).float() for f in FIELDS], -1)


def head_statistics(probabilities: torch.Tensor, targets: list[dict[str, Any]],
                    vocabulary: ConceptVocabulary) -> dict[str, Any]:
    gold = gold_heads(targets, vocabulary)
    result = {}
    for i, field in enumerate(FIELDS):
        p = probabilities[:, i*5:(i+1)*5].double()
        truth = gold[:, i*5:(i+1)*5].argmax(-1)
        predicted = p.argmax(-1)
        support = torch.bincount(truth, minlength=5)
        confusion = torch.bincount(truth * 5 + predicted, minlength=25).reshape(5, 5)
        recall = [int(confusion[j, j]) / int(support[j]) if int(support[j]) else None for j in range(5)]
        result[field] = {
            "class_values": [k for k, _ in sorted(vocabulary.fields[field].items(), key=lambda x: x[1])],
            "class_ids": list(range(5)), "support": support.tolist(), "confusion": confusion.tolist(),
            "correct": int((truth == predicted).sum()), "count": len(truth),
            "accuracy": float((truth == predicted).double().mean()),
            "balanced_accuracy": sum(v for v in recall if v is not None) / sum(v is not None for v in recall),
            "entropy_mean": float(-(p * p.clamp_min(1e-300).log()).sum(-1).mean()),
            "probabilities": p.tolist()}
    return result


def conditioning_cases(old: torch.Tensor, heads: torch.Tensor, old_gold: torch.Tensor,
                       head_gold: torch.Tensor, train_mean: torch.Tensor,
                       permutation: list[int], vocabulary: ConceptVocabulary) -> dict[str, torch.Tensor]:
    """Oracle and head-only cases are evaluation-only, with unchanged other groups."""
    old_cases = single_field_conditions(old, old_gold, vocabulary)
    result = {"predicted": torch.cat((old, heads), -1)}
    for name, section in (("participant_oracle", slice(0, 5)), ("time_oracle", slice(5, 10))):
        updated = heads.clone()
        updated[:, section] = head_gold[:, section]
        result[name] = torch.cat((old_cases[name], updated), -1)
    result["full_oracle"] = torch.cat((old_gold, head_gold), -1)
    result["head_train_mean"] = torch.cat((old, train_mean.expand(len(old), -1)), -1)
    result["head_permuted"] = torch.cat((old, heads[permutation]), -1)
    result["head_gold"] = torch.cat((old, head_gold), -1)
    return result


def sensitivity(base: torch.Tensor, changed: torch.Tensor,
                spans: list[dict[str, dict[str, int]]]) -> dict[str, Any]:
    """All byte positions are held-reference next-token decisions, never +4 shifted."""
    a, b = base.double(), changed.double()
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    l1 = (a - b).abs().mean(-1)
    kl = (la.exp() * (la - lb)).sum(-1).clamp_min(0)
    differing = a.argmax(-1) != b.argmax(-1)
    result = {"reference_history_oracle": True, "fields": {}}
    for field in FIELDS:
        regions = {}
        for region in ("before", "inside"):
            values = []
            for row, span in enumerate(spans):
                start, end = span[field]["start"], span[field]["end"]
                positions = range(start - 1, start) if region == "before" else range(start, end)
                for j in positions:
                    if j < 0:
                        raise ValueError("slot must have a preceding byte")
                    values.append({"row": row, "position": j, "logit_l1": float(l1[row, j]),
                                   "kl": float(kl[row, j]), "argmax_changed": bool(differing[row, j])})
            regions[region] = {"rows": len(spans), "byte_count": len(values),
                               "mean_logit_l1": sum(x["logit_l1"] for x in values) / len(values),
                               "mean_kl": sum(x["kl"] for x in values) / len(values),
                               "argmax_change_rate": sum(x["argmax_changed"] for x in values) / len(values),
                               "examples": values}
        result["fields"][field] = regions
    return result


@torch.no_grad()
def evaluate_c(model: ExplicitSlotModel, train: list[dict[str, Any]], rows: list[dict[str, Any]],
               tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary, corpus: Any, strict: Any) -> dict[str, Any]:
    model.eval()
    targets = [r["targets"] for r in rows]
    old, heads = source_distributions(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    _, train_heads = source_distributions(model, [corpus.model_inputs(r) for r in train], tensorizer)
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(17)).tolist()
    cases = conditioning_cases(old, heads, oracle_probabilities([t["concept"] for t in targets], vocabulary),
                               gold_heads(targets, vocabulary), train_heads.mean(0, keepdim=True), permutation, vocabulary)
    spans = [authored_slot_spans(t, corpus.seed_data()) for t in targets]
    if any(s is None for s in spans):
        raise ValueError("unrecognized authored target")
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    labels = pad([strict.target_history(r)["labels"] for r in rows], -100)
    base = model.decoder.decode_with_concept_intervention(ids, cases["predicted"])["logits"]
    result = {"dedicated_heads": head_statistics(heads, targets, vocabulary),
              "old_concepts": concept_head_stats(old, targets, vocabulary, permutation),
              "permutation": {"seed": 17, "indices": permutation},
              "train_mean": train_heads.mean(0).tolist(), "conditions": {}, "sensitivity": {}}
    for name, probs in cases.items():
        print(f"evaluating C/{name}", flush=True)
        generated = [asdict(g) for g in greedy_generate(model.decoder, probs, 128)]
        for g in generated:
            g["raw_bytes"] = [t - 4 for t in g["token_ids"] if 4 <= t < 260]
        logits = base if name == "predicted" else model.decoder.decode_with_concept_intervention(ids, probs)["logits"]
        result["conditions"][name] = {
            "concept_oracle": name in {"participant_oracle", "time_oracle", "full_oracle"},
            "head_oracle": name in {"participant_oracle", "time_oracle", "full_oracle", "head_gold"},
            "lm": conditional_loss(model, rows, probs, tensorizer, corpus, strict),
            "generation": generation_metrics(generated, [t["text"] for t in targets]),
            "slots": score_slots(generated, targets, corpus.seed_data()),
            "teacher_forced": {"reference_history_oracle": True, "metrics": span_metrics(base, logits, labels, spans)},
            "examples": [{"source": corpus.model_inputs(row), "source_predicted_concepts": old[i].tolist(),
                          "source_predicted_slot_heads": heads[i].tolist(), "decoder_probabilities": probs[i].tolist(),
                          "generation": g, "authored_reference": row["targets"]["text"]}
                         for i, (row, g) in enumerate(zip(rows, generated))]}
    for i, field in enumerate(FIELDS):
        swapped = heads.clone()
        swapped[:, i*5:(i+1)*5] = heads[permutation, i*5:(i+1)*5]
        logits = model.decoder.decode_with_concept_intervention(ids, torch.cat((old, swapped), -1))["logits"]
        result["sensitivity"][f"{field}_head_permuted"] = sensitivity(base, logits, spans)
    return result


def run(baseline_report: Path, out: Path) -> dict[str, Any]:
    raw = baseline_report.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_SHA:
        raise ValueError("historical A/B report hash mismatch")
    prior = json.loads(raw)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, rows = bundle["train"], bundle["validation"]
    del bundle
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train])
    replay = {}
    for arm in ("A", "B"):
        path = baseline_report.parent / f"{arm}.pt"
        expected = prior["arms"][arm]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["checkpoint"]["sha256"]:
            raise ValueError("historical checkpoint file mismatch")
        loaded = load_checkpoint(path, expected_dataset_version=corpus.VERSION,
                                 expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
        print(f"replaying historical {arm}", flush=True)
        evaluation = evaluate_arm(loaded.model, rows, tensorizer, vocabulary, corpus, strict)
        prior_eval = dict(expected["evaluation"])
        prior_eval.pop("historical_replay_equal", None)  # A-only prior provenance flag, not a metric.
        if json.loads(json.dumps(evaluation)) != prior_eval or state_digest(loaded.model) != expected["checkpoint"]["state_sha256"]:
            raise RuntimeError(f"historical {arm} evaluation/state does not replay")
        replay[arm] = {"evaluation_equal": True, "state_equal": True, "checkpoint": expected["checkpoint"]}
    torch.manual_seed(7)
    base = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
    if state_digest(base) != INITIAL_SHA:
        raise RuntimeError("common initial weights differ")
    model = ExplicitSlotModel(base, vocabulary, new_seed=20)
    parameters = dict(model.named_parameters())
    for name, value in base.named_parameters():
        current = parameters[name]
        if name == "decoder.concept_projection.weight":
            current = current[:, :value.shape[1]]
        if not torch.equal(value, current):
            raise RuntimeError(f"shared initial parameter differs: {name}")
    count = sum(p.numel() for p in model.parameters())
    if count != 43073:
        raise RuntimeError("precommitted C parameter count differs")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    initial_hash = state_digest(model)
    schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    if schedule_hash != SCHEDULE_SHA:
        raise RuntimeError("sampling schedule differs")
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    trace = []
    started = time.perf_counter()
    model.train()
    print("training C: seed7/600 CPU updates, head CE weights1 each", flush=True)
    for step, indices in enumerate(schedule):
        part = [train[i] for i in indices]
        ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
        targets = [r["targets"] for r in part]
        optimizer.zero_grad(set_to_none=True)
        output, auxiliary, slot_logits = model(ids, semantic, labels=labels)
        existing = model.decoder.losses(output, targets=encode_targets(targets, vocabulary), bottleneck_output=auxiliary)
        head_loss = slot_head_loss(slot_logits, targets, vocabulary)
        total = existing + head_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            trace.append({"step": step + 1, "base": float(existing.detach()), "head_ce_sum": float(head_loss.detach()),
                          "total": float(total.detach()), "head_targets_each": len(part)})
    elapsed = time.perf_counter() - started
    evaluation = evaluate_c(model, train, rows, tensorizer, vocabulary, corpus, strict)
    path = out / "C.pt"
    save_explicit_checkpoint(path, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
    restored = load_explicit_checkpoint(path, expected_dataset_version=corpus.VERSION,
                                        expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    sources = [corpus.model_inputs(r) for r in rows]
    a = source_distributions(model, sources, tensorizer)
    b = source_distributions(restored.model, sources, tensorizer)
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    with torch.no_grad():
        x = torch.cat(a, -1); y = torch.cat(b, -1)
        equal = {"state": state_digest(model) == state_digest(restored.model),
                 "old_concepts": torch.equal(a[0], b[0]), "dedicated_heads": torch.equal(a[1], b[1]),
                 "logits": torch.equal(model.decoder.decode_with_concept_intervention(ids, x)["logits"],
                                       restored.model.decoder.decode_with_concept_intervention(ids, y)["logits"]),
                 "greedy": greedy_generate(model.decoder, x, 128) == greedy_generate(restored.model.decoder, y, 128)}
    if not all(equal.values()):
        raise RuntimeError("C checkpoint replay differs")
    return {"version": "issue20-explicit-slots-1", "dataset_version": corpus.VERSION,
            "baseline_report_sha256": PRIOR_SHA, "historical_replay": replay,
            "test_evaluated": False, "train_rows": len(train), "validation_rows": len(rows),
            "budget": prior["budget"], "new_parameter_seed": 20, "conditioning_dim": 43,
            "parameters": count, "additional_parameters": 650, "common_initial_equal": True,
            "common_initial_sha256": INITIAL_SHA, "initial_C_sha256": initial_hash,
            "schedule_sha256": schedule_hash, "head_ce_weights": {"participant": 1., "time": 1.},
            "slot_byte_ce_weight": 0., "training_seconds": elapsed, "trace": trace,
            "parameter_changes": parameter_changes(before, model), "evaluation": evaluation,
            "checkpoint": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "state_sha256": state_digest(model), "reload": equal}}


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
