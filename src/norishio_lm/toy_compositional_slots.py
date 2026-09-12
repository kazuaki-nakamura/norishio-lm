"""Frozen compositional pair-space control and versioned byte diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .compositional_slots import factorized_pairs, pair_space_metrics
from .concept_model import ConceptVocabulary
from .explicit_slot_model import source_distributions
from .head_following_diagnostics import evaluate_condition
from .head_joint_metrics import head_joint_metrics
from .local_slot_checkpoint import load_local_checkpoint
from .pair_slot_checkpoint import load_pair_checkpoint
from .pair_slot_model import source_pair_distributions, pair_marginals
from .pair_metrics import pair_support
from .tensorizer import SemanticTensorizer
from .toy_adapter import source_record
from .toy_controls import state_digest
from .toy_experiment import fixture_module, pad
from .toy_explicit_slots import gold_heads
from .toy_head_following import score_seed, SEEDS
from .toy_local_slots import checked_report
from .toy_pair_head import evaluate_b, replay_checkpoint, generation_cases, scalar_means

PRIOR_SHA = "902545c14f91c68845d45b0af880432aa8c004137e5ff9cbd458ae84506f69c1"
PRIOR26_SHA = "964ab4d7dd7fc8b2fdddac816af51ac2ea8ab28a806a33dff32e40631250170f"
PRIOR24_SHA = "eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_relation(a: torch.Tensor, c: torch.Tensor) -> dict[str, Any]:
    if a.shape != c.shape:
        raise ValueError("comparison shapes differ")
    return {"exact": torch.equal(a, c), "allclose": torch.allclose(a, c, atol=1e-5, rtol=1e-5),
            "max_abs": float((a - c).abs().max()), "atol": 1e-5, "rtol": 1e-5}


@torch.no_grad()
def score_c(model: Any, rows: list[dict], tensorizer: SemanticTensorizer,
            vocabulary: ConceptVocabulary, corpus: Any, strict: Any, support: dict,
            historical: dict) -> dict:
    targets = [r["targets"] for r in rows]
    old, heads = source_distributions(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    pair = factorized_pairs(heads)
    marginals = pair_marginals(pair)
    a_input, c_input = torch.cat((old, heads), -1), torch.cat((old, marginals), -1)
    ids = pad([strict.target_history(r)["input_ids"] for r in rows])
    a_logits = model.decoder.decode_with_concept_intervention(ids, a_input)["logits"]
    c_logits = model.decoder.decode_with_concept_intervention(ids, c_input)["logits"]
    conditions = {}
    for name, probs in (("predicted", c_input), ("both_gold", torch.cat((old, gold_heads(targets, vocabulary)), -1))):
        print(f"C {name}", flush=True)
        conditions[name] = evaluate_condition(model, rows, probs, tensorizer, vocabulary, corpus, strict)
    greedy_equal = {name: conditions[name]["generations"] == historical["conditions"][name]["generations"]
                    for name in conditions}
    return {"pair_space": pair_space_metrics(pair, targets, vocabulary, support["train_table5x5"]),
            "head_metrics": head_joint_metrics(marginals, targets, vocabulary),
            "conditions": conditions, "numerical_relation": {"marginals": numerical_relation(heads, marginals),
                "logits": numerical_relation(a_logits, c_logits), "greedy_equal": greedy_equal},
            "pair_probabilities": pair.tolist(), "marginal_probabilities": marginals.tolist()}


@torch.no_grad()
def run(baseline_report: Path, out: Path) -> dict:
    prior = checked_report(baseline_report, PRIOR_SHA)
    base = baseline_report.parent.parent
    prior26_path = base / "issue26-fixed-v1" / "report.json"
    prior24_path = base / "issue24-seed7-v1" / "report.json"
    prior26 = checked_report(prior26_path, PRIOR26_SHA)
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
        raise RuntimeError("pair support changed")
    expected = dict(expected_dataset_version=corpus.VERSION, expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    records = {}
    seed7_model = None
    for seed in SEEDS:
        key = str(seed)
        path_a = prior24_path.parent / "D.pt" if seed == 7 else prior26_path.parent / f"D-seed{seed}.pt"
        meta_a = prior24["arms"]["D"]["checkpoint"] if seed == 7 else prior26["seeds"][key]["checkpoint"]
        path_b = baseline_report.parent / prior["B"][key]["checkpoint"]["file"]
        meta_b = prior["B"][key]["checkpoint"]
        if _sha(path_a) != meta_a["sha256"] or _sha(path_b) != meta_b["sha256"]:
            raise ValueError("historical checkpoint hash differs")
        a, b = load_local_checkpoint(path_a, **expected).model, load_pair_checkpoint(path_b, **expected).model
        hash_a, hash_b = state_digest(a), state_digest(b)
        if hash_a != meta_a["state_sha256"] or hash_b != meta_b["state_sha256"]:
            raise ValueError("historical checkpoint state differs")
        print(f"seed{seed}: complete historical A replay", flush=True)
        evaluation_a = score_seed(a, rows, tensorizer, vocabulary, corpus, strict)
        if json.loads(json.dumps(evaluation_a)) != prior["A"][key]["evaluation"]:
            raise RuntimeError("historical A differs")
        print(f"seed{seed}: complete historical B replay", flush=True)
        evaluation_b = evaluate_b(b, train, rows, tensorizer, vocabulary, corpus, strict, support)
        if json.loads(json.dumps(evaluation_b)) != prior["B"][key]["evaluation"]:
            raise RuntimeError("historical B differs")
        replay_b = replay_checkpoint(b, load_pair_checkpoint(path_b, **expected).model, rows, tensorizer, corpus, strict)
        c = score_c(a, rows, tensorizer, vocabulary, corpus, strict, support, evaluation_a)
        pair_splits = {}
        for split, part in (("train", train), ("validation", rows)):
            targets = [r["targets"] for r in part]
            sources = [corpus.model_inputs(r) for r in part]
            _, independent = source_distributions(a, sources, tensorizer)
            _, _, flat, _ = source_pair_distributions(b, sources, tensorizer)
            pair_splits[split] = {"C": pair_space_metrics(factorized_pairs(independent), targets, vocabulary, support["train_table5x5"]),
                                  "B": pair_space_metrics(flat, targets, vocabulary, support["train_table5x5"])}
        if state_digest(a) != hash_a or state_digest(b) != hash_b:
            raise RuntimeError("frozen model mutated")
        records[key] = {"historical_A_equal": True, "historical_B_equal": True,
            "A_state_sha256": hash_a, "B_state_sha256": hash_b, "frozen_states_unchanged": True,
            "B_checkpoint_replay": replay_b, "A": evaluation_a, "B": evaluation_b,
            "C": c, "pair_comparison": pair_splits}
        if seed == 7:
            seed7_model = a
    assert seed7_model is not None
    before = state_digest(seed7_model)
    old, heads = source_distributions(seed7_model, [corpus.model_inputs(r) for r in rows], tensorizer)
    byte_diagnosis = {}
    for name, probs in generation_cases(old, heads, gold_heads([r["targets"] for r in rows], vocabulary)).items():
        print(f"new byte-v2 seed7 {name}", flush=True)
        value = evaluate_condition(seed7_model, rows, probs, tensorizer, vocabulary, corpus, strict, include_teacher_forced=True)
        if value["teacher_forced"] is None:
            raise RuntimeError("byte-v2 was not measured")
        if name in records["7"]["A"]["conditions"]:
            legacy = dict(value)
            legacy["teacher_forced"] = None
            if legacy != records["7"]["A"]["conditions"][name]:
                raise RuntimeError("byte repair changed legacy generation or LM")
        byte_diagnosis[name] = value
    if before != state_digest(seed7_model):
        raise RuntimeError("byte diagnosis mutated seed7 model")
    return {"version": "issue30-compositional-1", "prior_report_sha256": PRIOR_SHA,
        "dataset_version": corpus.VERSION, "test_evaluated": False, "training_updates": 0,
        "additional_parameters": 0, "C_reuses_A_checkpoint": True, "D_trained": False,
        "factorized_NLL": "participant CE + time CE; auxiliary weight1 would double head CE, not add interaction",
        "support": support, "seeds": records,
        "means": {"n": 3, "aggregation": "unweighted arithmetic scalar means; missing is not zero",
                  "C": scalar_means([r["C"] for r in records.values()]),
                  "pair_comparison": scalar_means([r["pair_comparison"] for r in records.values()])},
        "teacher_forced_v2": {"seed": 7, "version": "teacher-forced-byte-v2", "previously_unmeasured": True,
                              "frozen_state_unchanged": True, "conditions": byte_diagnosis},
        "historical_reports_unchanged": all(_sha(p) == sha for p, sha in
            ((baseline_report, PRIOR_SHA), (prior26_path, PRIOR26_SHA), (prior24_path, PRIOR24_SHA)))}


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
