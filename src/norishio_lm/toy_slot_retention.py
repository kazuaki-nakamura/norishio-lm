"""Replay a fixed per-step model to diagnose participant/time retention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .collapse_metrics import concept_head_stats
from .concept_model import CONCEPT_FIELDS, ConceptVocabulary
from .span_metrics import span_metrics
from .toy_checkpoint import load_checkpoint
from .toy_collapse import oracle_probabilities, source_encodings
from .toy_controls import state_digest
from .toy_evaluation import score_condition
from .toy_experiment import fixture_module, pad
from .toy_slots import score_slots
from .toy_spans import authored_slot_spans


def single_field_conditions(predicted: torch.Tensor, oracle: torch.Tensor,
                             vocabulary: ConceptVocabulary) -> dict[str, torch.Tensor]:
    """Explicit oracle probability groups; other five fields remain untouched."""
    width = sum(len(vocabulary.fields[f]) + 1 for f in CONCEPT_FIELDS)
    if predicted.ndim != 2 or predicted.shape != oracle.shape or predicted.shape[1] != width:
        raise ValueError("aligned concept probability groups required")
    slices = {}
    offset = 0
    for field in CONCEPT_FIELDS:
        end = offset + len(vocabulary.fields[field]) + 1
        slices[field] = slice(offset, end)
        offset = end
    result = {"predicted": predicted}
    for name, fields in (("participant_oracle", ("participant",)), ("time_oracle", ("time",)),
                         ("both_oracle", ("participant", "time")), ("full_oracle", CONCEPT_FIELDS)):
        intervention = predicted.clone()
        for field in fields:
            intervention[:, slices[field]] = oracle[:, slices[field]]
        result[name] = intervention
    return result


def verify_baseline(checkpoint: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Require the previous matched diagnostic's checkpoint and fixed budget."""
    expected = {"seed": 7, "updates": 600, "batch_size": 16, "optimizer": "Adam",
                "learning_rate": .003, "clip": 1., "all_four_loss_weights": 1.,
                "threads": 1, "device": "cpu", "max_new_tokens": 128}
    if report.get("version") != "issue12-step-conditioning-1" or report.get("budget") != expected:
        raise ValueError("baseline must be the fixed Issue12 diagnostic")
    arm = report["arms"]["per_step_additive"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != arm["checkpoint"]["sha256"]:
        raise ValueError("checkpoint file hash differs from baseline report")
    return arm


@torch.no_grad()
def run(checkpoint: Path, baseline_path: Path, out: Path) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
    prior_arm = verify_baseline(checkpoint, baseline_report)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    loaded = load_checkpoint(checkpoint, expected_dataset_version=corpus.VERSION)
    model, tensorizer, vocabulary = loaded.model, loaded.tensorizer, loaded.vocabulary
    if model.pathway != "C" or model.decoder.conditioning_mode != "per_step_additive" or loaded.constant is not None:
        raise ValueError("normal per-step strict C checkpoint required")
    model.eval().requires_grad_(False)
    before = state_digest(model)
    rows = corpus.build()["validation"]
    seed = corpus.seed_data()
    sources = [corpus.model_inputs(r) for r in rows]
    targets = [r["targets"] for r in rows]
    _, predicted = source_encodings(model, sources, tensorizer)
    oracle = oracle_probabilities([t["concept"] for t in targets], vocabulary)
    conditions = single_field_conditions(predicted, oracle, vocabulary)
    spans = [authored_slot_spans(t, seed) for t in targets]
    if any(span is None for span in spans):
        raise ValueError("validation target does not match a unique authored slot template")
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    labels = pad([strict.target_history(r)["labels"] for r in rows], -100)
    baseline_logits = model.decoder.decode_with_concept_intervention(ids, predicted)["logits"]
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(17)).tolist()
    head = concept_head_stats(predicted, targets, vocabulary, permutation)
    if head["concept_metrics"] != prior_arm["concept_metrics"]:
        raise RuntimeError("checkpoint concept metrics do not replay the baseline")
    result: dict[str, Any] = {
        "version": "issue14-slot-retention-1", "dataset_version": corpus.VERSION,
        "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "baseline_report_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "training_performed": False, "vocabulary_refit": False, "test_evaluated": False,
        "budget_origin": baseline_report["budget"], "model_state_sha256": before,
        "concept_head": head, "source_predicted_probabilities": predicted.tolist(),
        "slot_spans": spans, "teacher_forced_diagnostic": {}, "free_running": {},
        "prefix_oracle": {"performed": False, "reason": "optional diagnostic deferred; history propagation is not causally isolated"}}
    for name, probabilities in conditions.items():
        print(f"evaluating {name}: fixed model, validation only", flush=True)
        logits = baseline_logits if name == "predicted" else model.decoder.decode_with_concept_intervention(ids, probabilities)["logits"]
        result["teacher_forced_diagnostic"][name] = {
            "reference_history_oracle": True, "concept_oracle": name != "predicted",
            "metrics": span_metrics(baseline_logits, logits, labels, spans)}
        free = score_condition(model, rows, probabilities, tensorizer, vocabulary, corpus, strict, predicted)
        free["oracle"] = name != "predicted"
        for example in free["examples"]:
            g = example["generation"]
            g["raw_bytes"] = [t - 4 for t in g["token_ids"] if 4 <= t < 260]
        free["slots"] = score_slots([e["generation"] for e in free["examples"]], targets, seed)
        result["free_running"][name] = free
    # JSON checkpoints record ordered operator tuples as lists; compare the
    # exact persisted representation, without tolerances or dropping metrics.
    replay = json.loads(json.dumps(result["free_running"]["predicted"], allow_nan=False))
    if replay != prior_arm["conditions"]["predicted"]:
        raise RuntimeError("checkpoint free generation and scoring do not replay baseline")
    result["baseline_replay_equal"] = True
    if state_digest(model) != before:
        raise RuntimeError("diagnostics modified model state")
    result["model_unchanged"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkpoint, args.baseline_report, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
