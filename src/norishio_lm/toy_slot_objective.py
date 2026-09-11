"""Matched CPU comparison of unchanged versus slot-weighted decoder objectives."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .collapse_metrics import concept_head_stats
from .concept_model import ConceptVocabulary
from .slot_objective import train_slot_control
from .span_metrics import span_metrics
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_checkpoint import load_checkpoint, save_checkpoint
from .toy_collapse import source_encodings, oracle_probabilities
from .toy_controls import state_digest, train_control
from .toy_evaluation import score_condition, reload_check, parameter_changes
from .toy_experiment import ToyModel, fixture_module, pad
from .toy_slot_retention import single_field_conditions, verify_baseline
from .toy_slots import score_slots
from .toy_spans import authored_slot_spans

INITIAL_SHA = "f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70"
SCHEDULE_SHA = "6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66"
PRIOR_REPORT_SHA = "3b08a5982c89fd63b4c5887f03ee7c1276a9ce51b3e9f8b072e830c448b791d2"


@torch.no_grad()
def evaluate_arm(model: ToyModel, rows: list[dict[str, Any]], tensorizer: SemanticTensorizer,
                 vocabulary: ConceptVocabulary, corpus: Any, strict: Any) -> dict[str, Any]:
    targets = [r["targets"] for r in rows]
    _, predicted = source_encodings(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    oracle = oracle_probabilities([t["concept"] for t in targets], vocabulary)
    conditions = single_field_conditions(predicted, oracle, vocabulary)
    del conditions["both_oracle"]
    spans = [authored_slot_spans(t, corpus.seed_data()) for t in targets]
    if any(s is None for s in spans):
        raise ValueError("unrecognized authored target")
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    labels = pad([strict.target_history(r)["labels"] for r in rows], -100)
    baseline_logits = model.decoder.decode_with_concept_intervention(ids, predicted)["logits"]
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(17)).tolist()
    result = {"concept_head": concept_head_stats(predicted, targets, vocabulary, permutation),
              "slot_spans": spans, "conditions": {}}
    for name, probabilities in conditions.items():
        logits = baseline_logits if name == "predicted" else model.decoder.decode_with_concept_intervention(ids, probabilities)["logits"]
        free = score_condition(model, rows, probabilities, tensorizer, vocabulary, corpus, strict, predicted)
        free["oracle"] = name != "predicted"
        for example in free["examples"]:
            g = example["generation"]
            g["raw_bytes"] = [t - 4 for t in g["token_ids"] if 4 <= t < 260]
        free["slots"] = score_slots([e["generation"] for e in free["examples"]], targets, corpus.seed_data())
        result["conditions"][name] = {
            "free_running": free,
            "teacher_forced": {"reference_history_oracle": True, "concept_oracle": name != "predicted",
                               "metrics": span_metrics(baseline_logits, logits, labels, spans)}}
    return result


def run(baseline_report: Path, baseline_checkpoint: Path, out: Path) -> dict[str, Any]:
    raw = baseline_report.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PRIOR_REPORT_SHA:
        raise ValueError("fixed historical report hash mismatch")
    prior = json.loads(raw)
    prior_arm = verify_baseline(baseline_checkpoint, prior)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    train, validation = bundle["train"], bundle["validation"]
    del bundle
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train])
    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
    initial_state = state_digest(initial)
    initial_parameters = {n: p.detach().clone() for n, p in initial.named_parameters()}
    _, init_probs = source_encodings(initial, [corpus.model_inputs(r) for r in train], tensorizer)
    schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    if initial_state != INITIAL_SHA or schedule_hash != SCHEDULE_SHA:
        raise RuntimeError("precommitted initialization or sampling schedule differs")
    historical = load_checkpoint(baseline_checkpoint, expected_dataset_version=corpus.VERSION,
                                 expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    report: dict[str, Any] = {
        "version": "issue18-slot-objective-1", "dataset_version": corpus.VERSION,
        "budget": prior["budget"], "slot_weight": {"A": 0., "B": 1.},
        "existing_auxiliary_weights": {"lm": 1., "concept": 1., "sense": 1., "sememe": 1.},
        "slot_loss_definition": "mean CE across union of participant/time bytes; no EOS or padding",
        "optional_slot_head": {"performed": False, "reason": "preregistered objective-only A/B"},
        "initial_state_sha256": initial_state, "schedule_sha256": schedule_hash,
        "prior_report_sha256": PRIOR_REPORT_SHA,
        "prior_checkpoint_sha256": hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest(),
        "train_rows": len(train), "validation_rows": len(validation), "test_evaluated": False,
        "arms": {}}
    for arm, weight in (("A", 0.), ("B", 1.)):
        print(f"training {arm}: slot weight {weight}, seed7/600 CPU updates", flush=True)
        if arm == "A":
            model, training = train_control(initial, "predicted", schedule, train, tensorizer,
                                            vocabulary, corpus, strict, init_probs.mean(0, keepdim=True))
        else:
            model, training = train_slot_control(initial, schedule, train, tensorizer,
                                                 vocabulary, corpus, strict, weight=weight)
        if state_digest(initial) != initial_state:
            raise RuntimeError("training modified shared initial model")
        training["parameter_changes"] = parameter_changes(initial_parameters, model)
        training["additional_parameters"] = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in initial.parameters())
        print(f"evaluating {arm}: validation and oracle diagnostics", flush=True)
        evaluation = evaluate_arm(model, validation, tensorizer, vocabulary, corpus, strict)
        if arm == "A":
            if state_digest(model) != state_digest(historical.model):
                raise RuntimeError("A weights do not reproduce historical per-step baseline")
            free = evaluation["conditions"]["predicted"]["free_running"]
            if json.loads(json.dumps(free)) != prior_arm["conditions"]["predicted"]:
                raise RuntimeError("A normal generation or metrics do not reproduce historical baseline")
            evaluation["historical_replay_equal"] = True
        path = out / f"{arm}.pt"
        save_checkpoint(path, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
        checkpoint = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "state_sha256": state_digest(model),
                      "reload": reload_check(path, model, tensorizer, vocabulary, validation, corpus, strict, None)}
        report["arms"][arm] = {"slot_weight": weight, "training": training,
                               "evaluation": evaluation, "checkpoint": checkpoint}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-report", "baseline-checkpoint", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_report, args.baseline_checkpoint, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
