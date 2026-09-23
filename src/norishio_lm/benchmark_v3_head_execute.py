"""One-shot, no-retry executor for the frozen Issue #46 comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from . import benchmark_v3_head_fixture as fixture
from .benchmark_v3_head_metrics import aggregate_head_metrics
from .benchmark_v3_head_protocol import (
    ARMS, BOUND_CODE, FREEZE_PATH, ROOT, RUN_ORDER, SEEDS,
    canonical_bytes, load_protocol,
)
from .benchmark_v3_head_runner import (
    evaluate_head_outputs, paired_initialization_preflight, train_head_mode,
)
from .benchmark_v3_model import BOS_ID, BYTE_OFFSET, EOS_ID, FACTOR_ORDER, VOCAB_SIZE
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_runner import FactorVocabulary, PreparedRow


RESULT_DIR = ROOT / "docs" / "results" / "benchmark-v3-head-learning"
ATTEMPTS_PATH = RESULT_DIR / "attempts.jsonl"
WALL_PATH = RESULT_DIR / "execution-wall.json"
SUMMARY_PATH = RESULT_DIR / "summary.json"
RUN_SCHEMA = "norishio.issue46.head-learning-run.v1"
SUMMARY_SCHEMA = "norishio.issue46.head-learning-summary.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_attempt(value: Mapping[str, Any]) -> None:
    ATTEMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ATTEMPTS_PATH.open("ab") as stream:
        stream.write(canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout


def _committed_freeze_sha() -> str:
    """Bind live protocol and sources to the commit that introduced the freeze."""
    paths = [str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
             "data/benchmark_v3_head_learning/spec.json",
             "data/benchmark_v3_head_learning/expected-manifest.json",
             "src/norishio_lm/benchmark_v3_head_protocol.py", *BOUND_CODE]
    freeze_sha = _git_bytes("log", "-1", "--format=%H", "--", paths[0]).decode("ascii").strip()
    if len(freeze_sha) != 40:
        raise ValueError("freeze descriptor has not been committed")
    for relative in paths:
        tracked = _git_bytes("show", f"{freeze_sha}:{relative}").replace(b"\r\n", b"\n")
        live = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if tracked != live:
            raise ValueError(f"pre-training freeze source is uncommitted or changed: {relative}")
    return freeze_sha


def _vocabulary() -> FactorVocabulary:
    spec = fixture.spec_data()
    values = {
        "participant": tuple(item["id"] for item in spec["participants"]),
        "time": tuple(item["id"] for item in spec["times"]),
        "event": tuple(item["id"] for item in spec["events"]),
        "operator": tuple(spec["operators"]),
    }
    return FactorVocabulary(values)


def prepare_head_row(row: Mapping[str, Any], vocabulary: FactorVocabulary) -> PreparedRow:
    """Expose only authored source bytes and the four supervised labels."""
    source = tuple(fixture.source_ids(row))
    targets = row["targets"]
    frame = targets["frame"]
    text = targets["text"]
    target_bytes = tuple(BYTE_OFFSET + byte for byte in text.encode("utf-8"))
    decoder = (BOS_ID, *target_bytes)
    labels = (*target_bytes, EOS_ID)
    if any(not 0 <= token < VOCAB_SIZE for token in (*source, *decoder, *labels)):
        raise ValueError("fixture token is outside the frozen byte vocabulary")
    return PreparedRow(source, decoder, labels, vocabulary.encode(frame),
                       text, {factor: str(frame[factor]) for factor in FACTOR_ORDER})


def _raw_rows(arm: str, seed: int, split: str,
              source_rows: Sequence[Mapping[str, Any]],
              heads: Sequence[Mapping[str, Any]], vocabulary: FactorVocabulary) -> list[dict[str, Any]]:
    if len(source_rows) != len(heads):
        raise ValueError("head records do not cover every frozen row")
    records: list[dict[str, Any]] = []
    for index, (row, head) in enumerate(zip(source_rows, heads, strict=True)):
        if head["row_index"] != index or head["target_frame"] != row["targets"]["frame"]:
            raise ValueError("head prediction and frozen fixture row differ")
        vectors = head["factor_probability_vectors"]
        argmax = head["factor_argmax"]
        prediction = vocabulary.decode([argmax[factor] for factor in FACTOR_ORDER])
        records.append({
            "arm": arm, "seed": seed, "row_id": row["id"], "split": split,
            "support_group": row["metadata"]["support_class"],
            "target_frame": row["targets"]["frame"],
            "factor_probability_vectors": vectors,
            "factor_argmax": argmax,
            "predicted_frame": prediction,
            "available": True,
        })
    return records


def _run_path(index: int, arm: str, seed: int) -> Path:
    return RESULT_DIR / f"run-{index:02d}-{arm.lower()}-seed{seed}.json"


def _validate_saved_row(raw: Mapping[str, Any], authored: Mapping[str, Any],
                        *, arm: str, seed: int, split: str,
                        vocabulary: FactorVocabulary) -> None:
    if (raw.get("arm") != arm or raw.get("seed") != seed or raw.get("split") != split
            or raw.get("row_id") != authored["id"]
            or raw.get("support_group") != authored["metadata"]["support_class"]
            or raw.get("target_frame") != authored["targets"]["frame"]
            or raw.get("available") is not True):
        raise ValueError("saved raw row metadata differs from frozen fixture")
    vectors = raw.get("factor_probability_vectors")
    argmax = raw.get("factor_argmax")
    if not isinstance(vectors, Mapping) or not isinstance(argmax, Mapping):
        raise ValueError("saved raw row lacks canonical head outputs")
    if set(vectors) != set(FACTOR_ORDER) or set(argmax) != set(FACTOR_ORDER):
        raise ValueError("saved raw row lacks a canonical factor")
    predicted: list[int] = []
    for factor in FACTOR_ORDER:
        probabilities = vectors[factor]
        if (not isinstance(probabilities, list) or len(probabilities) != len(vocabulary.values[factor])
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       or value < 0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6):
            raise ValueError("saved raw row has invalid factor probabilities")
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        if argmax[factor] != winner:
            raise ValueError("saved raw row argmax differs from its probabilities")
        predicted.append(winner)
    if raw.get("predicted_frame") != vocabulary.decode(predicted):
        raise ValueError("saved raw row predicted frame differs from its argmax")


def _validate_attempt_log(runs: Sequence[Mapping[str, Any]], descriptor_sha256: str) -> None:
    try:
        events = [json.loads(line) for line in ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("attempt log is absent or malformed") from exc
    if len(events) != 12:
        raise ValueError("attempt log must contain one start and completion per run")
    for index, ((arm, seed), run) in enumerate(zip(RUN_ORDER, runs, strict=True), start=1):
        start, complete = events[2 * index - 2:2 * index]
        path = _run_path(index, arm, seed)
        if (start.get("event") != "started" or complete.get("event") != "completed"
                or any(event.get("index") != index or event.get("arm") != arm
                       or event.get("seed") != seed for event in (start, complete))
                or start.get("descriptor_sha256") != descriptor_sha256
                or complete.get("raw_name") != path.name
                or complete.get("raw_sha256") != _file_sha256(path)
                or complete.get("optimizer_updates") != run["optimizer_updates"]):
            raise ValueError(f"attempt log does not bind completed run {index}")


def _metric(report: Mapping[str, Any], arm: str, seed: int,
            split: str, support: str, factor: str) -> float:
    group = report["by_arm_seed"][arm][str(seed)]
    values = group["train_resubstitution"] if split == "train" else group["confirmation"][support]
    field = values["fields"][factor]
    if field["unavailable_count"] or field["scheduled_count"] == 0:
        raise ValueError(f"{arm} seed{seed} {split}/{support}/{factor} is unavailable")
    score = field["class_balanced_accuracy"]
    if not isinstance(score, (int, float)):
        raise ValueError("balanced accuracy is unavailable")
    return float(score)


def decide(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
           descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen factor-level branch predicates without a composite score."""
    rules = descriptor["interpretation"]
    train_min = rules["strong_train_balanced_accuracy_min"]
    noncollapse_min = rules["noncollapsed_confirmation_balanced_accuracy_min"]
    diversity_min = rules["noncollapsed_distinct_predicted_classes_min"]
    gap_min = rules["heldout_train_minus_unseen_balanced_accuracy_min"]
    similarity_max = rules["similar_arm_abs_confirmation_balanced_accuracy_delta_max"]
    fields = tuple(FACTOR_ORDER)
    train = {(arm, seed, factor): _metric(report, arm, seed, "train", "all", factor)
             for arm in ARMS for seed in SEEDS for factor in fields}
    confirm = {(arm, seed, factor): _metric(report, arm, seed, "confirmation", "all", factor)
               for arm in ARMS for seed in SEEDS for factor in fields}
    unseen = {(arm, seed, factor): _metric(report, arm, seed, "confirmation", "unseen_pair", factor)
              for arm in ARMS for seed in SEEDS for factor in ("participant", "time")}
    weak = [factor for factor in fields
            if any(train["FACTOR_ONLY", seed, factor] < train_min for seed in SEEDS)]
    diversity = {factor: {str(seed): len({row["predicted_frame"][factor]
                                          for row in rows if row["arm"] == "FACTOR_ONLY"
                                          and row["seed"] == seed and row["split"] == "confirmation"})
                           for seed in SEEDS}
                 for factor in ("event", "operator")}
    interference = (
        all(train["FACTOR_ONLY", seed, factor] >= train_min
            for seed in SEEDS for factor in ("participant", "time"))
        and all(confirm["FACTOR_ONLY", seed, factor] >= noncollapse_min
                and diversity[factor][str(seed)] >= diversity_min
                for seed in SEEDS for factor in ("event", "operator"))
        and all(confirm["FACTOR_ONLY", seed, factor] > confirm["JOINT", seed, factor]
                for seed in SEEDS for factor in ("participant", "time"))
    )
    composition = all(
        train[arm, seed, factor] >= train_min
        and train[arm, seed, factor] - unseen[arm, seed, factor] >= gap_min
        for arm in ARMS for seed in SEEDS for factor in ("participant", "time")
    )
    fixture_instability = all(
        train[arm, seed, factor] >= train_min
        and confirm[arm, seed, factor] >= train_min
        for arm in ARMS for seed in SEEDS for factor in ("participant", "time")
    ) and all(
        abs(confirm["FACTOR_ONLY", seed, factor] - confirm["JOINT", seed, factor]) <= similarity_max
        for seed in SEEDS for factor in ("participant", "time")
    )
    if len(weak) == 1:
        branch = "narrow_factor_follow_up"
    elif any(factor in weak for factor in ("participant", "time")):
        branch = "basic_optimization_or_capacity_unresolved"
    elif interference:
        branch = "joint_objective_interference_leading"
    elif composition:
        branch = "compositional_generalization_leading"
    elif fixture_instability:
        branch = "fixture_specific_instability_leading"
    else:
        branch = "mixed_or_inconclusive"
    return {
        "branch": branch,
        "weak_factor_only_train_factors": weak,
        "predicates": {"interference": interference, "composition": composition,
                       "fixture_instability": fixture_instability},
        "factor_only_confirmation_predicted_class_diversity": diversity,
        "factor_balanced_accuracy": {
            arm: {str(seed): {factor: {"train_resubstitution": train[arm, seed, factor],
                                      "confirmation": confirm[arm, seed, factor],
                                      **({"unseen_pair": unseen[arm, seed, factor]}
                                         if factor in ("participant", "time") else {})}
                              for factor in fields}
                  for seed in SEEDS} for arm in ARMS},
        "claim_scope": "authored_structural_factor_classification_only; no modern lexical or semantic-layer claim",
    }


def aggregate_saved_runs() -> dict[str, Any]:
    freeze = load_protocol()
    descriptor = freeze["descriptor"]
    freeze_sha = _committed_freeze_sha()
    bundle = fixture.build()
    fixture.check_expected(fixture.validate(bundle))
    vocabulary = _vocabulary()
    authored = bundle["train"] + bundle["confirmation"]
    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (arm, seed) in enumerate(RUN_ORDER, start=1):
        run = json.loads(_run_path(index, arm, seed).read_text(encoding="utf-8"))
        if (run.get("schema") != RUN_SCHEMA or run.get("arm") != arm
                or run.get("seed") != seed or run.get("descriptor_sha256") != freeze["descriptor_sha256"]
                or run.get("optimizer_updates") != 600
                or run.get("status") != "complete"
                or run.get("freeze_commit_sha") != freeze_sha
                or run.get("fixture_content_digest_sha256") != descriptor["fixture"]["content_digest_sha256"]
                or run.get("initial_state_sha256") != descriptor["arms"]["paired_initialization"][str(seed)]["state_sha256"]
                or run.get("trainable_parameters") != descriptor["arms"]["paired_initialization"][str(seed)]["parameter_count"]
                or run.get("schedule_sha256") != descriptor["training"]["schedule_sha256"][str(seed)]
                or not isinstance(run.get("training_wall_seconds"), (int, float))
                or not math.isfinite(run["training_wall_seconds"]) or run["training_wall_seconds"] < 0
                or len(run.get("losses", [])) != 600
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in run["losses"])
                or len(run.get("rows", [])) != 480):
            raise ValueError(f"saved run {index} is incomplete or differs from freeze")
        for row_index, (raw, source) in enumerate(zip(run["rows"], authored, strict=True)):
            _validate_saved_row(raw, source, arm=arm, seed=seed,
                                split="train" if row_index < 384 else "confirmation",
                                vocabulary=vocabulary)
        runs.append(run)
        rows.extend(run["rows"])
    _validate_attempt_log(runs, freeze["descriptor_sha256"])
    report = aggregate_head_metrics(rows, vocabulary.values, factors=FACTOR_ORDER)
    if report["row_count"] != 2880:
        raise ValueError("aggregate omitted a frozen raw row")
    wall = json.loads(WALL_PATH.read_text(encoding="utf-8"))
    if (not isinstance(wall.get("executor_total_wall_seconds"), (int, float))
            or not math.isfinite(wall["executor_total_wall_seconds"])
            or wall["executor_total_wall_seconds"] < 0):
        raise ValueError("executor wall time was not retained")
    decision = decide(report, rows, descriptor)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "descriptor_sha256": freeze["descriptor_sha256"],
        "freeze_commit_sha": runs[0]["freeze_commit_sha"],
        "attempted_runs": len(runs),
        "optimizer_updates": sum(run["optimizer_updates"] for run in runs),
        "training_wall_seconds": sum(run["training_wall_seconds"] for run in runs),
        "executor_total_wall_seconds": wall["executor_total_wall_seconds"],
        "run_files": [{"name": _run_path(index, arm, seed).name,
                       "sha256": _file_sha256(_run_path(index, arm, seed))}
                      for index, (arm, seed) in enumerate(RUN_ORDER, start=1)],
        "metrics": report,
        "decision": decision,
    }
    if not SUMMARY_PATH.exists():
        _write_json(SUMMARY_PATH, summary)
    elif json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) != summary:
        raise ValueError("saved summary differs from frozen raw-only recomputation")
    return summary


def execute_once() -> dict[str, Any]:
    """Attempt each preregistered run exactly once and retain every outcome."""
    freeze = load_protocol()
    descriptor = freeze["descriptor"]
    freeze_sha = _committed_freeze_sha()
    if ATTEMPTS_PATH.exists() or WALL_PATH.exists() or SUMMARY_PATH.exists() or any(
        _run_path(index, arm, seed).exists()
        for index, (arm, seed) in enumerate(RUN_ORDER, start=1)
    ):
        raise FileExistsError("head-learning attempts already exist; do not retry")
    bundle = fixture.build()
    fixture.check_expected(fixture.validate(bundle))
    vocabulary = _vocabulary()
    prepared = {split: [prepare_head_row(row, vocabulary) for row in bundle[split]]
                for split in ("train", "confirmation")}
    for seed in SEEDS:
        preflight = paired_initialization_preflight(seed)
        bound = descriptor["arms"]["paired_initialization"][str(seed)]
        if (preflight["initial_state_sha256"] != bound["state_sha256"]
                or preflight["trainable_parameters"] != bound["parameter_count"]
                or not preflight["bitwise_equal"]):
            raise ValueError("paired initialization preflight differs from freeze")
    started = time.perf_counter()
    try:
        for index, (arm, seed) in enumerate(RUN_ORDER, start=1):
            schedule = batch_schedule(seed=seed, train_rows=384, steps=600, batch_size=16)
            if schedule_sha256(schedule) != descriptor["training"]["schedule_sha256"][str(seed)]:
                raise ValueError("run schedule differs from freeze")
            _append_attempt({"event": "started", "index": index, "arm": arm, "seed": seed,
                             "descriptor_sha256": freeze["descriptor_sha256"]})
            try:
                training = train_head_mode(arm, seed, prepared["train"], schedule, descriptor)
                confirmation = evaluate_head_outputs(training.model, prepared["confirmation"],
                                                     descriptor=descriptor, split="confirmation",
                                                     objective=arm, seed=seed)
                raw = _raw_rows(arm, seed, "train", bundle["train"],
                                training.train_head_records, vocabulary)
                raw.extend(_raw_rows(arm, seed, "confirmation", bundle["confirmation"],
                                     confirmation["head_records"], vocabulary))
                if training.optimizer_updates != 600 or training.schedule_sha256 != schedule_sha256(schedule):
                    raise ValueError("optimizer update count or schedule differs from freeze")
                payload = {
                    "schema": RUN_SCHEMA, "status": "complete", "arm": arm, "seed": seed,
                    "descriptor_sha256": freeze["descriptor_sha256"],
                    "freeze_commit_sha": freeze_sha,
                    "fixture_content_digest_sha256": descriptor["fixture"]["content_digest_sha256"],
                    "initial_state_sha256": training.initial_state_sha256,
                    "final_state_sha256": training.final_state_sha256,
                    "trainable_parameters": training.trainable_parameters,
                    "schedule_sha256": training.schedule_sha256,
                    "optimizer_updates": training.optimizer_updates,
                    "training_wall_seconds": training.training_wall_seconds,
                    "losses": training.losses,
                    "rows": raw,
                }
                path = _run_path(index, arm, seed)
                _write_json(path, payload)
                _append_attempt({"event": "completed", "index": index, "arm": arm, "seed": seed,
                                 "raw_name": path.name, "raw_sha256": _file_sha256(path),
                                 "optimizer_updates": 600})
                print(f"completed {index}/6 {arm} seed{seed}", flush=True)
            except BaseException as exc:
                _append_attempt({"event": "failed", "index": index, "arm": arm,
                                 "seed": seed, "error_type": type(exc).__name__,
                                 "reason": "attempt_failed_no_retry"})
                raise
    finally:
        _write_json(WALL_PATH, {"schema": "norishio.issue46.executor-wall.v1",
                                "executor_total_wall_seconds": time.perf_counter() - started,
                                "measurement": "perf_counter around six run attempts and raw writes; excludes raw-only aggregation"})
    return aggregate_saved_runs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    args = parser.parse_args()
    result = execute_once() if args.mode == "run" else aggregate_saved_runs()
    print(json.dumps({"status": result["status"],
                      "decision": result["decision"]["branch"],
                      "descriptor_sha256": result["descriptor_sha256"]},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
