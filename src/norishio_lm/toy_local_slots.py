"""Preregistered duplicate removal and causal local slot experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary, encode_targets
from .explicit_slot_checkpoint import load_explicit_checkpoint
from .explicit_slot_model import ExplicitSlotModel, source_distributions, slot_head_loss
from .factorized_slot_checkpoint import load_factorized_checkpoint
from .local_slot_model import DuplicateRemovedModel
from .local_slot_generation import greedy_generate_local
from .local_slot_checkpoint import save_local_checkpoint, load_local_checkpoint
from .pair_metrics import pair_support
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest
from .toy_experiment import ToyModel, fixture_module, batch, pad
from .toy_explicit_slots import evaluate_c
from .toy_joint_slots import evaluate, PRIOR_SHA as HISTORICAL_SHA
from .toy_slot_objective import INITIAL_SHA, SCHEDULE_SHA

PRIOR_SHA = "a303e74fab83ccc11b4350579988b2e88e2077ba44abc263f32549890de2be39"


def checked_report(path: Path, expected: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("historical report hash mismatch")
    return json.loads(raw)


def kept_columns(vocabulary: ConceptVocabulary) -> tuple[int, ...]:
    offset = 0
    kept = []
    for field in CONCEPT_FIELDS:
        width = len(vocabulary.fields[field]) + 1
        if field not in {"participant", "time"}:
            kept.extend(range(offset, offset + width))
        offset += width
    if offset != 33 or len(kept) != 21:
        raise ValueError("unexpected old concept layout")
    return tuple(kept)


def check_initialization(shared: ExplicitSlotModel, model: DuplicateRemovedModel,
                         vocabulary: ConceptVocabulary) -> dict[str, Any]:
    prefix = "decoder.concept_projection."
    new = model.state_dict()
    common = all(torch.equal(v, new[k]) for k, v in shared.state_dict().items()
                 if not k.startswith(prefix))
    a, b = shared.decoder.concept_projection, model.decoder.concept_projection
    kept = kept_columns(vocabulary)
    exact = (torch.equal(a.weight[:, kept], b.base_proj.weight)
             and torch.equal(a.bias, b.base_proj.bias)
             and torch.equal(a.weight[:, 33:38], b.participant_proj.weight)
             and torch.equal(a.weight[:, 38:], b.time_proj.weight))
    if not common or not exact:
        raise RuntimeError("new arm did not preserve common weights")
    return {"common_parameters_exact": common, "projection_columns_exact": exact,
            "kept_old_columns": list(kept), "transport_dim": 43, "effective_dim": 31}


def run(baseline_report: Path, historical_report: Path, out: Path) -> dict[str, Any]:
    prior = checked_report(baseline_report, PRIOR_SHA)
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
        raise RuntimeError("pair support changed")
    expected = dict(expected_dataset_version=corpus.VERSION, expected_tensorizer=tensorizer,
                    expected_vocabulary=vocabulary)
    evaluations = {}
    for name, report, directory, filename, loader in (
        ("A", historical, historical_report.parent, "C.pt", load_explicit_checkpoint),
        ("B", prior, baseline_report.parent, "B.pt", load_factorized_checkpoint),
    ):
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["checkpoint"]["sha256"]:
            raise ValueError("historical checkpoint hash mismatch")
        loaded = loader(path, **expected)
        if state_digest(loaded.model) != report["checkpoint"]["state_sha256"]:
            raise RuntimeError("historical state differs")
        print(f"replaying historical {name}", flush=True)
        if name == "A":
            replay = evaluate_c(loaded.model, train, rows, tensorizer, vocabulary, corpus, strict)
            if json.loads(json.dumps(replay)) != historical["evaluation"]:
                raise RuntimeError("Issue20 A does not replay")
        replay = evaluate(loaded.model, rows, tensorizer, vocabulary, corpus, strict, support["validation_seen"])
        if json.loads(json.dumps(replay)) != prior["evaluations"][name]:
            raise RuntimeError(f"Issue22 {name} does not replay")
        evaluations[name] = replay

    torch.manual_seed(7)
    initial = ToyModel("C", tensorizer, vocabulary, conditioning_mode="per_step_additive")
    if state_digest(initial) != INITIAL_SHA:
        raise RuntimeError("common initial state differs")
    shared = ExplicitSlotModel(initial, vocabulary, new_seed=20)
    models = {name: DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=local)
              for name, local in (("C", False), ("D", True))}
    if state_digest(models["C"]) != state_digest(models["D"]):
        raise RuntimeError("C/D initial weights differ")
    initialization = {n: check_initialization(shared, m, vocabulary) for n, m in models.items()}
    schedule = torch.randint(len(train), (600, 16), generator=torch.Generator().manual_seed(7)).tolist()
    if hashlib.sha256(json.dumps(schedule).encode()).hexdigest() != SCHEDULE_SHA:
        raise RuntimeError("training schedule changed")
    arms = {}
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        if params != 42689:
            raise RuntimeError("new arm parameter count differs")
        initial_hash = state_digest(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        trace = []
        model.train()
        started = time.perf_counter()
        print(f"training {name}: 600 CPU updates", flush=True)
        for step, indices in enumerate(schedule):
            part = [train[i] for i in indices]
            ids, labels, semantic = batch(part, "C", tensorizer, corpus, strict)
            targets = [r["targets"] for r in part]
            optimizer.zero_grad(set_to_none=True)
            output, aux, head_logits = model(ids, semantic, labels=labels)
            base = model.decoder.losses(output, targets=encode_targets(targets, vocabulary), bottleneck_output=aux)
            head = slot_head_loss(head_logits, targets, vocabulary)
            total = base + head
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 0 or (step + 1) % 10 == 0:
                trace.append({"step": step + 1, "base": float(base.detach()),
                              "head_ce": float(head.detach()), "total": float(total.detach())})
            if (step + 1) % 200 == 0:
                print(f"{name} update {step + 1}", flush=True)
        seconds = time.perf_counter() - started
        evaluations[name] = evaluate(model, rows, tensorizer, vocabulary, corpus, strict,
                                     support["validation_seen"], generate=greedy_generate_local)
        path = out / f"{name}.pt"
        save_local_checkpoint(path, model, tensorizer, vocabulary, dataset_version=corpus.VERSION)
        restored = load_local_checkpoint(path, **expected).model
        sources = [corpus.model_inputs(r) for r in rows]
        a, b = source_distributions(model, sources, tensorizer), source_distributions(restored, sources, tensorizer)
        ids = pad([strict.target_history(r)["input_ids"] for r in rows])
        with torch.no_grad():
            x, y = torch.cat(a, -1), torch.cat(b, -1)
            same = {"state": state_digest(model) == state_digest(restored),
                    "old_concepts": torch.equal(a[0], b[0]), "heads": torch.equal(a[1], b[1]),
                    "logits": torch.equal(model.decoder.decode_with_concept_intervention(ids, x)["logits"],
                                          restored.decoder.decode_with_concept_intervention(ids, y)["logits"]),
                    "greedy": greedy_generate_local(model.decoder, x, 128) == greedy_generate_local(restored.decoder, y, 128)}
        if not all(same.values()):
            raise RuntimeError("new checkpoint replay differs")
        arms[name] = {"parameters": params, "parameter_delta_from_AB": -384,
                      "initial_sha256": initial_hash, "initialization": initialization[name],
                      "training_seconds": seconds, "trace": trace,
                      "checkpoint": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "state_sha256": state_digest(model), "reload": same}}
    return {"version": "issue24-local-slots-1", "dataset_version": corpus.VERSION,
            "historical_report_sha256": HISTORICAL_SHA, "prior_report_sha256": PRIOR_SHA,
            "historical_replay_equal": {"A_issue20": True, "A_issue22": True, "B_issue22": True},
            "support": support, "test_evaluated": False, "schedule_sha256": SCHEDULE_SHA,
            "budget": prior["budget"], "arms": arms, "evaluations": evaluations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--historical-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.baseline_report, args.historical_report, args.out_dir)
    with (args.out_dir / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
