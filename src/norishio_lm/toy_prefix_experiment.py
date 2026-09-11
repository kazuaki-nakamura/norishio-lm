"""Frozen oracle-prefix diagnostics; no reference continuation enters generation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .prefix_generation import prefix_generate
from .prefix_metrics import score_prefix_rows
from .toy_checkpoint import load_checkpoint
from .toy_collapse import oracle_probabilities, source_encodings
from .toy_controls import state_digest
from .toy_experiment import fixture_module
from .toy_slot_retention import run as replay_slots, single_field_conditions
from .toy_spans import authored_slot_spans

CHECKPOINT_SHA = "0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057"
SLOT_REPORT_SHA = "a4d18bb7fab3d63a94d0d0e5b026b59956aae6db276241eba60b248e03621c92"


def checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("saved report hash mismatch")
    return json.loads(raw)


def boundary_positions(spans: list[dict[str, dict[str, int]]]) -> dict[str, list[int]]:
    """Six preregistered byte boundaries; never round to a Unicode character."""
    result = {}
    for field in ("participant", "time"):
        for name in ("before", "start", "end"):
            positions = [s[field]["start"] - 1 if name == "before" else s[field][name] for s in spans]
            if any(p < 0 for p in positions):
                raise ValueError("slot must have a preceding byte for fixed boundaries")
            result[f"{field}_{name}"] = positions
    return result


@torch.no_grad()
def run(checkpoint: Path, baseline_report: Path, slot_report: Path, out: Path) -> dict[str, Any]:
    prior = checked_json(slot_report, SLOT_REPORT_SHA)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("fixed checkpoint hash mismatch")
    out.mkdir(parents=True, exist_ok=False)
    # Run prior checks BEFORE all prefix measurements. Exact JSON normalizes only
    # tuple/list representation; no tolerances and no metric exclusions.
    replay = replay_slots(checkpoint, baseline_report, out / "baseline-replay")
    if json.loads(json.dumps(replay, allow_nan=False)) != prior:
        raise RuntimeError("complete Issue14 report does not replay")
    corpus = fixture_module("toy_corpus")
    loaded = load_checkpoint(checkpoint, expected_dataset_version=corpus.VERSION)
    model, vocabulary = loaded.model, loaded.vocabulary
    model.eval().requires_grad_(False)
    before = state_digest(model)
    rows = corpus.build()["validation"]
    targets = [r["targets"] for r in rows]
    _, predicted = source_encodings(model, [corpus.model_inputs(r) for r in rows], loaded.tensorizer)
    empty = prefix_generate(model.decoder, predicted, [[] for _ in rows], max_total_tokens=128)
    previous = prior["free_running"]["predicted"]["examples"]
    if [g["token_ids"] for g in empty] != [e["generation"]["token_ids"] for e in previous]:
        raise RuntimeError("empty-prefix decoder does not replay normal generation")
    oracle = oracle_probabilities([t["concept"] for t in targets], vocabulary)
    conditions = single_field_conditions(predicted, oracle, vocabulary)
    del conditions["both_oracle"]  # Plan specifies four concept conditions.
    spans = [authored_slot_spans(t, corpus.seed_data()) for t in targets]
    if any(s is None for s in spans):
        raise ValueError("unrecognized authored target")
    references = [[b + 4 for b in t["text"].encode("utf-8")] for t in targets]
    boundaries = boundary_positions(spans)
    report: dict[str, Any] = {
        "version": "issue16-prefix-1", "dataset_version": corpus.VERSION,
        "checkpoint_sha256": CHECKPOINT_SHA, "slot_report_sha256": SLOT_REPORT_SHA,
        "baseline_report_sha256": hashlib.sha256(baseline_report.read_bytes()).hexdigest(),
        "baseline_replay_equal": True, "empty_prefix_replay_equal": True, "model_state_sha256": before,
        "training_performed": False, "vocabulary_refit": False, "test_evaluated": False,
        "rows": len(rows), "max_total_tokens": 128, "cpu_threads": 1,
        "counterfactual_prefix": {"performed": False, "reason": "preregistered deferral; length/content effects not isolated"},
        "slot_spans": spans, "conditions": {}}
    report["normal_predicted_baseline"] = score_prefix_rows(empty, references, spans)
    for boundary, positions in boundaries.items():
        prefixes = [ref[:pos] for ref, pos in zip(references, positions)]
        for condition, probabilities in conditions.items():
            print(f"evaluating {boundary}/{condition}: frozen, validation only", flush=True)
            # Only allowed prefix tokens cross this boundary. Reference scoring
            # begins after self-running generation returns.
            generated = prefix_generate(model.decoder, probabilities, prefixes, max_total_tokens=128)
            metrics = score_prefix_rows(generated, references, spans)
            report["conditions"][f"{boundary}/{condition}"] = {
                "prefix_oracle": True, "concept_oracle": condition != "predicted",
                "prefix_lengths": positions, "metrics": metrics,
                "generations": [{k: v for k, v in g.items() if k != "decision_logits"} for g in generated]}
    if state_digest(model) != before:
        raise RuntimeError("prefix diagnosis changed model weights")
    report["model_unchanged"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "baseline-report", "slot-report", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.checkpoint, args.baseline_report, args.slot_report, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
