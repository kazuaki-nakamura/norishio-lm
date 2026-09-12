"""Three-seed slot-gradient routing control (Issue #32)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .concept_model import ConceptVocabulary, encode_targets
from .explicit_slot_model import slot_head_loss, source_distributions
from .head_following_diagnostics import evaluate_condition
from .local_slot_checkpoint import save_local_checkpoint, load_local_checkpoint
from .pair_metrics import pair_support
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest
from .toy_experiment import ToyModel, batch, fixture_module, pad
from .toy_explicit_slots import gold_heads
from .toy_head_following import checked_report, score_seed, SEEDS
from .toy_local_slots import PRIOR_SHA as HISTORICAL_LOCAL_SHA

BASELINE_SHA = "964ab4d7dd7fc8b2fdddac816af51ac2ea8ab28a806a33dff32e40631250170f"
HISTORICAL_SHA = "eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8"
ROUTING_MODES = ("end_to_end", "stop_slot_lm")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _vector(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    values = [torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1)
              for grad, param in zip(grads, params)]
    return torch.cat(values).double()


def _comparison(lm: torch.Tensor, ce: torch.Tensor) -> dict[str, Any]:
    lm_norm, ce_norm = float(lm.norm()), float(ce.norm())
    dot = float(torch.dot(lm, ce))
    cosine = dot / (lm_norm * ce_norm) if lm_norm and ce_norm else None
    return {"lm_norm": lm_norm, "ce_norm": ce_norm, "dot": dot, "cosine": cosine,
            "lm_exact_zero": bool(torch.equal(lm, torch.zeros_like(lm))),
            "ce_exact_zero": bool(torch.equal(ce, torch.zeros_like(ce)))}


def probe_gradients(model: Any, part: list[dict], tensorizer: SemanticTensorizer,
                    corpus: Any, strict: Any, vocabulary: ConceptVocabulary,
                    *, seed: int = 3200) -> dict[str, Any]:
    """Measure separate LM/CE gradients without mutating model or RNG."""
    ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
    targets = [row["targets"] for row in part]
    was_training = model.training
    model.train()
    output, _, head_logits = model(ids, semantic, labels=labels)
    params_by_field = {field: [model.slot_heads[field].weight, model.slot_heads[field].bias]
                       for field in ("participant", "time")}
    all_params = [param for values in params_by_field.values() for param in values]
    lm_grads = torch.autograd.grad(output["lm_loss"], all_params, retain_graph=True, allow_unused=True)
    comparisons = {}
    at = 0
    for field in ("participant", "time"):
        field_params = params_by_field[field]
        lm = _vector(lm_grads[at:at + 2], field_params)
        at += 2
        ce = torch.nn.functional.cross_entropy(head_logits[field],
                                               model.vocabulary.fields[field] and
                                               torch.tensor([model.vocabulary.fields[field][t["concept"][field]] - 1
                                                             for t in targets], dtype=torch.long))
        ce_grads = torch.autograd.grad(ce, field_params, retain_graph=True, allow_unused=True)
        comparisons[field] = _comparison(lm, _vector(ce_grads, field_params))
    if was_training is False:
        model.eval()
    return {"probe_rows": len(part), "generator_seed": seed, "fields": comparisons}


def _clip_record(model: torch.nn.Module, total: torch.Tensor) -> tuple[float, float]:
    total.backward()
    pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    coefficient = min(1., 1. / pre) if pre > 0 else 1.
    return pre, coefficient


def train_arm(model: Any, train: list[dict], tensorizer: SemanticTensorizer,
              corpus: Any, strict: Any, vocabulary: ConceptVocabulary,
              schedule: list[list[int]], probe_indices: list[int], mode: str) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    trace: list[dict[str, Any]] = []
    probes: dict[str, Any] = {}
    if 0 in (0, 100, 300, 600):
        probes["0"] = probe_gradients(model, [train[i] for i in probe_indices], tensorizer,
                                       corpus, strict, vocabulary)
    model.train()
    started = time.perf_counter()
    for step, indices in enumerate(schedule, start=1):
        part = [train[i] for i in indices]
        ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
        targets = [row["targets"] for row in part]
        optimizer.zero_grad(set_to_none=True)
        output, aux, heads = model(ids, semantic, labels=labels)
        base = model.decoder.losses(output, targets=encode_targets(targets, vocabulary), bottleneck_output=aux)
        head = slot_head_loss(heads, targets, vocabulary)
        total = base + head
        pre, coefficient = _clip_record(model, total)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            trace.append({"step": step, "base": float(base.detach()), "head_ce": float(head.detach()),
                          "total": float(total.detach()), "clip_pre_norm": pre,
                          "clip_coefficient": coefficient})
        if step in (100, 300, 600):
            probes[str(step)] = probe_gradients(model, [train[i] for i in probe_indices], tensorizer,
                                                corpus, strict, vocabulary)
    return {"training_seconds": time.perf_counter() - started, "trace": trace,
            "probes": probes, "routing_mode": mode}


def _byte_v2(model: Any, rows: list[dict], old: torch.Tensor, heads: torch.Tensor,
             tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
             corpus: Any, strict: Any) -> dict[str, Any]:
    targets = [row["targets"] for row in rows]
    gold = gold_heads(targets, vocabulary)
    cases = {
        "predicted": torch.cat((old, heads), -1),
        "participant_gold": torch.cat((old, torch.cat((gold[:, :5], heads[:, 5:]), -1)), -1),
        "time_gold": torch.cat((old, torch.cat((heads[:, :5], gold[:, 5:]), -1)), -1),
        "both_gold": torch.cat((old, gold), -1),
    }
    return {name: evaluate_condition(model, rows, probs, tensorizer, vocabulary, corpus, strict,
                                     include_teacher_forced=True)["teacher_forced"]
            for name, probs in cases.items()}


def run(baseline_report: Path, historical_report: Path, out: Path) -> dict[str, Any]:
    prior = checked_report(baseline_report, BASELINE_SHA)
    historical = checked_report(historical_report, HISTORICAL_SHA)
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
        raise RuntimeError("dataset support changed")
    expected = {"expected_dataset_version": corpus.VERSION, "expected_tensorizer": tensorizer,
                "expected_vocabulary": vocabulary}
    probe_generator = torch.Generator().manual_seed(3200)
    probe_indices = torch.randperm(len(train), generator=probe_generator)[:16].tolist()
    arms: dict[str, Any] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
        initial_digest = state_digest(initial)
        schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(seed)).tolist()
        schedule_sha = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
        seed_records: dict[str, Any] = {"seed": seed, "initial_sha256": initial_digest,
                                        "schedule_sha256": schedule_sha, "probe_indices": probe_indices}
        for mode in ROUTING_MODES:
            model = __import__("norishio_lm.local_slot_model", fromlist=["DuplicateRemovedModel"]).DuplicateRemovedModel(
                initial, vocabulary, new_seed=20, local=True, gradient_routing=mode)
            if state_digest(model) != state_digest(__import__("norishio_lm.local_slot_model", fromlist=["DuplicateRemovedModel"]).DuplicateRemovedModel(
                    initial, vocabulary, new_seed=20, local=True, gradient_routing="end_to_end")):
                raise RuntimeError("arm initial states differ")
            params = sum(parameter.numel() for parameter in model.parameters())
            if params != 42689:
                raise RuntimeError("parameter count changed")
            result = train_arm(model, train, tensorizer, corpus, strict, vocabulary,
                               schedule, probe_indices, mode)
            evaluation = score_seed(model, rows, tensorizer, vocabulary, corpus, strict)
            sources = [corpus.model_inputs(r) for r in rows]
            old, heads = source_distributions(model, sources, tensorizer)
            result["byte_v2"] = _byte_v2(model, rows, old, heads, tensorizer, vocabulary, corpus, strict)
            path = out / f"{mode}-seed{seed}.pt"
            save_local_checkpoint(path, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
            restored = load_local_checkpoint(path, **expected).model
            ids = pad([strict.target_history(r)["input_ids"] for r in rows])
            a, b = source_distributions(model, sources, tensorizer), source_distributions(restored, sources, tensorizer)
            with torch.no_grad():
                replay = {"state": state_digest(model) == state_digest(restored),
                          "old_concepts": torch.equal(a[0], b[0]), "heads": torch.equal(a[1], b[1]),
                          "logits": torch.equal(model.decoder.decode_with_concept_intervention(ids, torch.cat(a, -1))["logits"],
                                                restored.decoder.decode_with_concept_intervention(ids, torch.cat(b, -1))["logits"])}
            if not all(replay.values()):
                raise RuntimeError("routing checkpoint replay differs")
            result.update({"parameters": params, "initial_state_sha256": initial_digest,
                           "checkpoint": {"file": path.name, "sha256": _sha(path),
                                          "state_sha256": state_digest(model), "reload": replay},
                           "evaluation": evaluation})
            seed_records[mode] = result
        arms[str(seed)] = seed_records
    return {"version": "issue32-gradient-routing-1", "baseline_report_sha256": BASELINE_SHA,
            "historical_report_sha256": HISTORICAL_SHA, "dataset_version": corpus.VERSION,
            "test_evaluated": False, "support": support, "probe_seed": 3200,
            "probe_indices": probe_indices, "routing_modes": list(ROUTING_MODES),
            "budget": {"seeds": list(SEEDS), "updates": 600, "batch_size": 16,
                       "optimizer": "Adam", "lr": .003, "clip": 1., "cpu_threads": 1,
                       "base_loss_weights": [1, 1, 1, 1], "slot_ce_weights": [1, 1],
                       "parameters": 42689}, "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_report, args.historical_report, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
