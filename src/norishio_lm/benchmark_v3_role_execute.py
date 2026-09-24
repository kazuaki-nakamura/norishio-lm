"""One-shot, append-only Issue #48 executor and saved-raw-only aggregator."""
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

from . import benchmark_v3_role_fixture as fixture
from .benchmark_v3_head_runner import evaluate_head_outputs
from .benchmark_v3_model import FACTOR_ORDER
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_role_metrics import summarize_role_heads
from .benchmark_v3_role_model import build_initialized_model
from .benchmark_v3_role_protocol import (
    ARMS, BOUND_CODE, FREEZE_PATH, MAX_ATTEMPTS, MAX_UPDATES,
    MAX_WALL_SECONDS, ROOT, RUN_ORDER, SEEDS, STATIC_AUDIT_PATH,
    canonical_bytes, load_protocol, sha256,
)
from .benchmark_v3_role_runner import prepare_role_row, train_factor_only
from .benchmark_v3_runner import FactorVocabulary

RESULT_DIR = ROOT / "docs" / "results" / "benchmark-v3-role-identifiability"
ATTEMPTS_PATH = RESULT_DIR / "attempts.jsonl"
WALL_PATH = RESULT_DIR / "execution-wall.json"
SUMMARY_PATH = RESULT_DIR / "summary.json"
RUN_SCHEMA = "norishio.issue48.role-identifiability-run.v1"
SUMMARY_SCHEMA = "norishio.issue48.role-identifiability-summary.v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
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
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _committed_freeze_sha() -> str:
    descriptor_path = str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/")
    paths = [descriptor_path, "data/benchmark_v3_role_identifiability/spec.json",
             "data/benchmark_v3_role_identifiability/expected-manifest.json",
             "data/benchmark_v3_role_identifiability/static-audit.json",
             "docs/results/benchmark-v3-role-identifiability/PROTOCOL.md",
             *BOUND_CODE]
    freeze_sha = _git_bytes("log", "-1", "--format=%H", "--", descriptor_path).decode("ascii").strip()
    if len(freeze_sha) != 40:
        raise ValueError("role-identifiability descriptor is not committed")
    for relative in paths:
        committed = _git_bytes("show", f"{freeze_sha}:{relative}").replace(b"\r\n", b"\n")
        live = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if committed != live:
            raise ValueError(f"frozen Issue #48 source changed: {relative}")
    return freeze_sha


def _vocabulary() -> FactorVocabulary:
    spec = fixture.spec_data()
    return FactorVocabulary({
        "participant": tuple(item["id"] for item in spec["participants"]),
        "time": tuple(item["id"] for item in spec["times"]),
        "event": tuple(item["id"] for item in spec["events"]),
        "operator": tuple(spec["operators"]),
    })


def _run_path(index: int, arm: str, seed: int) -> Path:
    return RESULT_DIR / f"run-{index:02d}-{arm.lower()}-seed{seed}.json"


def _source_binding(row: Mapping[str, Any]) -> dict[str, str]:
    ids = fixture.source_ids(row)
    return {
        "source_ids_sha256": sha256(ids),
        "source_mask_sha256": sha256([True] * len(ids)),
        "count_signature_sha256": sha256(fixture.count_signature(row)),
        "frequency_signature_sha256": sha256(fixture.normalized_frequency_signature(row)),
    }


def _raw_rows(arm: str, seed: int, split: str,
              authored: Sequence[Mapping[str, Any]],
              heads: Sequence[Mapping[str, Any]],
              vocabulary: FactorVocabulary) -> list[dict[str, Any]]:
    if len(authored) != len(heads):
        raise ValueError("head outputs do not cover every scheduled row")
    output: list[dict[str, Any]] = []
    for index, (row, head) in enumerate(zip(authored, heads, strict=True)):
        if head["row_index"] != index or head["target_frame"] != row["targets"]["frame"]:
            raise ValueError("head output and frozen row differ")
        argmax = head["factor_argmax"]
        output.append({
            "arm": arm, "seed": seed, "row_id": row["id"], "split": split,
            "support_group": row["metadata"]["support_class"],
            "target_frame": row["targets"]["frame"],
            "factor_probability_vectors": head["factor_probability_vectors"],
            "factor_argmax": argmax,
            "predicted_frame": vocabulary.decode([argmax[factor] for factor in FACTOR_ORDER]),
            "available": True,
            **_source_binding(row),
        })
    return output


def _validate_saved_row(raw: Mapping[str, Any], authored: Mapping[str, Any],
                        *, arm: str, seed: int, split: str,
                        vocabulary: FactorVocabulary) -> None:
    expected = {"arm": arm, "seed": seed, "row_id": authored["id"],
                "split": split, "support_group": authored["metadata"]["support_class"],
                "target_frame": authored["targets"]["frame"], "available": True,
                **_source_binding(authored)}
    if any(raw.get(key) != value for key, value in expected.items()):
        raise ValueError("saved raw row differs from frozen fixture or source binding")
    vectors = raw.get("factor_probability_vectors")
    argmax = raw.get("factor_argmax")
    if not isinstance(vectors, Mapping) or not isinstance(argmax, Mapping):
        raise ValueError("saved raw row lacks source-head outputs")
    if set(vectors) != set(FACTOR_ORDER) or set(argmax) != set(FACTOR_ORDER):
        raise ValueError("saved raw row lacks canonical factors")
    winners: list[int] = []
    for factor in FACTOR_ORDER:
        probabilities = vectors[factor]
        if (not isinstance(probabilities, list)
                or len(probabilities) != len(vocabulary.values[factor])
                or any(not isinstance(value, (int, float)) or not math.isfinite(value)
                       or value < 0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-6):
            raise ValueError("saved head probability vector is invalid")
        winner = max(range(len(probabilities)), key=probabilities.__getitem__)
        if argmax[factor] != winner:
            raise ValueError("saved argmax differs from its probability vector")
        winners.append(winner)
    if raw.get("predicted_frame") != vocabulary.decode(winners):
        raise ValueError("saved predicted frame differs from canonical argmax")


def _validate_ledger(runs: Sequence[Mapping[str, Any]], descriptor_sha: str) -> None:
    events = [json.loads(line) for line in ATTEMPTS_PATH.read_text(encoding="utf-8").splitlines()]
    if len(events) != 2 * MAX_ATTEMPTS:
        raise ValueError("attempt ledger is not six starts plus six completions")
    for index, ((arm, seed), run) in enumerate(zip(RUN_ORDER, runs, strict=True), start=1):
        start, complete = events[2 * index - 2:2 * index]
        path = _run_path(index, arm, seed)
        if (start.get("event") != "started" or complete.get("event") != "completed"
                or any(event.get("index") != index or event.get("arm") != arm
                       or event.get("seed") != seed for event in (start, complete))
                or start.get("descriptor_sha256") != descriptor_sha
                or complete.get("raw_name") != path.name
                or complete.get("raw_sha256") != _file_sha256(path)
                or complete.get("optimizer_updates") != run["optimizer_updates"]):
            raise ValueError("attempt ledger does not bind saved raw")


def _collision_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe saved within-signature differences without re-running heads."""
    groups: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["arm"], row["seed"], row["split"], row["count_signature_sha256"])
        groups.setdefault(key, []).append(row)
    colliding = [group for group in groups.values() if len(group) > 1]
    disagreement = 0
    max_probability_delta = 0.0
    for group in colliding:
        first = group[0]
        for later in group[1:]:
            if later["factor_argmax"] != first["factor_argmax"]:
                disagreement += 1
            for factor in FACTOR_ORDER:
                max_probability_delta = max(max_probability_delta, *(
                    abs(a - b) for a, b in zip(
                        first["factor_probability_vectors"][factor],
                        later["factor_probability_vectors"][factor], strict=True)))
    return {"collision_group_count": len(colliding),
            "collision_row_count": sum(len(group) for group in colliding),
            "rows_with_argmax_disagreement_vs_group_first": disagreement,
            "max_probability_delta_vs_group_first": max_probability_delta,
            "interpretation": "descriptive floating-point differences; static signature equivalence unchanged"}


def aggregate_saved_runs() -> dict[str, Any]:
    freeze = load_protocol(raw_only=True)
    descriptor = freeze["descriptor"]
    freeze_sha = _committed_freeze_sha()
    bundle = fixture.build()
    fixture.check_expected(bundle)
    vocabulary = _vocabulary()
    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, (arm, seed) in enumerate(RUN_ORDER, start=1):
        path = _run_path(index, arm, seed)
        run = json.loads(path.read_bytes())
        authored = [*bundle[arm]["train"], *bundle[arm]["confirmation"]]
        if (run.get("schema") != RUN_SCHEMA or run.get("status") != "complete"
                or run.get("arm") != arm or run.get("seed") != seed
                or run.get("descriptor_sha256") != freeze["descriptor_sha256"]
                or run.get("freeze_commit_sha") != freeze_sha
                or run.get("fixture_manifest_sha256") != descriptor["fixture"]["manifest_sha256"]
                or run.get("schedule_sha256") != descriptor["training"]["schedule_sha256"][str(seed)]
                or run.get("initial_state_sha256") != descriptor["arms"]["paired_initialization"][str(seed)]["initial_state_sha256"]
                or run.get("trainable_parameters") != 32_120
                or run.get("optimizer_updates") != 600
                or len(run.get("losses", [])) != 600
                or any(not isinstance(loss, (int, float)) or not math.isfinite(loss)
                       for loss in run["losses"])
                or not isinstance(run.get("training_wall_seconds"), (int, float))
                or not 0 <= run["training_wall_seconds"] <= MAX_WALL_SECONDS
                or len(run.get("rows", [])) != 480):
            raise ValueError(f"saved Issue #48 run {index} differs from freeze")
        for row_index, (raw, authored_row) in enumerate(zip(run["rows"], authored, strict=True)):
            _validate_saved_row(raw, authored_row, arm=arm, seed=seed,
                                split="train" if row_index < 384 else "confirmation",
                                vocabulary=vocabulary)
        runs.append(run)
        rows.extend(run["rows"])
    _validate_ledger(runs, freeze["descriptor_sha256"])
    if sum(run["optimizer_updates"] for run in runs) != MAX_UPDATES:
        raise ValueError("optimizer budget differs from freeze")
    wall = json.loads(WALL_PATH.read_text(encoding="utf-8"))
    total_wall = wall.get("executor_total_wall_seconds")
    if not isinstance(total_wall, (int, float)) or not 0 <= total_wall <= MAX_WALL_SECONDS:
        raise ValueError("executor wall budget unavailable or exceeded")
    ceilings = descriptor["fixture"]["static_preflight"]["aliased_train_ceiling"]
    reduced = summarize_role_heads(
        rows, vocabulary.values, ceilings,
        strong_min=descriptor["interpretation"]["strong_train_balanced_accuracy_min"],
        ceiling_margin=descriptor["interpretation"]["rescue_train_margin_over_aliased_ceiling"],
        generalization_gap=descriptor["interpretation"]["generalization_train_minus_confirmation_stratum_min"],
        event_operator_noncollapse_min=descriptor["interpretation"]["event_operator_noncollapse_balanced_accuracy_min"],
        event_operator_distinct_classes_min=descriptor["interpretation"]["event_operator_noncollapse_distinct_predicted_classes_min"],
    )
    # A deterministic signature-only classifier has this slice-local ceiling.
    # Floating-point discrepancies above it require a separate raw audit, not
    # a silent positive branch.
    for factor in FACTOR_ORDER:
        for seed in SEEDS:
            aliased = reduced["decision"]["by_factor"][factor]["aliased_train_balanced_by_seed"][str(seed)]
            if aliased > ceilings[factor] + 1e-6:
                raise ValueError("learned aliased score exceeds static ceiling; audit saved raw")
    summary = {
        "schema": SUMMARY_SCHEMA, "status": "complete",
        "descriptor_sha256": freeze["descriptor_sha256"],
        "freeze_commit_sha": freeze_sha,
        "attempted_runs": len(runs),
        "optimizer_updates": MAX_UPDATES,
        "training_wall_seconds": sum(run["training_wall_seconds"] for run in runs),
        "executor_total_wall_seconds": total_wall,
        "run_files": [{"name": _run_path(index, arm, seed).name,
                       "sha256": _file_sha256(_run_path(index, arm, seed))}
                      for index, (arm, seed) in enumerate(RUN_ORDER, start=1)],
        **reduced,
        "saved_raw_collision_diagnostics": _collision_diagnostics(rows),
    }
    if not SUMMARY_PATH.exists():
        _write_json_exclusive(SUMMARY_PATH, summary)
    elif json.loads(SUMMARY_PATH.read_bytes()) != summary:
        raise ValueError("saved summary differs from authenticated raw-only aggregation")
    return summary


def execute_once() -> dict[str, Any]:
    freeze = load_protocol()
    descriptor = freeze["descriptor"]
    freeze_sha = _committed_freeze_sha()
    if (ATTEMPTS_PATH.exists() or WALL_PATH.exists() or SUMMARY_PATH.exists()
            or any(_run_path(index, arm, seed).exists()
                   for index, (arm, seed) in enumerate(RUN_ORDER, start=1))):
        raise FileExistsError("Issue #48 has prior attempts; no retry or budget reset")
    bundle = fixture.build()
    fixture.check_expected(bundle)
    vocabulary = _vocabulary()
    prepared = {arm: {split: [prepare_role_row(row, fixture.source_ids(row), vocabulary)
                              for row in bundle[arm][split]] for split in fixture.SPLITS}
                for arm in ARMS}
    started = time.perf_counter()
    try:
        for index, (arm, seed) in enumerate(RUN_ORDER, start=1):
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                raise TimeoutError("executor wall budget exhausted before next attempt")
            schedule = batch_schedule(seed=seed, train_rows=384, steps=600, batch_size=16)
            schedule_hash = schedule_sha256(schedule)
            if schedule_hash != descriptor["training"]["schedule_sha256"][str(seed)]:
                raise ValueError("frozen batch schedule changed")
            model = build_initialized_model(seed)
            from .benchmark_v3_checkpoint import state_sha256
            expected_initial = descriptor["arms"]["paired_initialization"][str(seed)]["initial_state_sha256"]
            if state_sha256(model.state_dict()) != expected_initial:
                raise ValueError("paired initialized model differs from freeze")
            _append_attempt({"event": "started", "index": index, "arm": arm, "seed": seed,
                             "descriptor_sha256": freeze["descriptor_sha256"]})
            try:
                trained = train_factor_only(model, seed, prepared[arm]["train"], schedule, descriptor)
                confirmation = evaluate_head_outputs(
                    model, prepared[arm]["confirmation"], descriptor=descriptor,
                    split="confirmation", objective="FACTOR_ONLY", seed=seed)
                raw = _raw_rows(arm, seed, "train", bundle[arm]["train"],
                                trained["train_head_records"], vocabulary)
                raw.extend(_raw_rows(arm, seed, "confirmation", bundle[arm]["confirmation"],
                                     confirmation["head_records"], vocabulary))
                if trained["optimizer_updates"] != 600:
                    raise ValueError("attempt update count differs from freeze")
                payload = {
                    "schema": RUN_SCHEMA, "status": "complete", "arm": arm, "seed": seed,
                    "descriptor_sha256": freeze["descriptor_sha256"],
                    "freeze_commit_sha": freeze_sha,
                    "fixture_manifest_sha256": descriptor["fixture"]["manifest_sha256"],
                    "initial_state_sha256": trained["initial_state_sha256"],
                    "final_state_sha256": trained["final_state_sha256"],
                    "trainable_parameters": 32_120,
                    "schedule_sha256": schedule_hash,
                    "optimizer_updates": trained["optimizer_updates"],
                    "training_wall_seconds": trained["training_wall_seconds"],
                    "losses": trained["losses"], "rows": raw,
                }
                path = _run_path(index, arm, seed)
                _write_json_exclusive(path, payload)
                _append_attempt({"event": "completed", "index": index, "arm": arm,
                                 "seed": seed, "raw_name": path.name,
                                 "raw_sha256": _file_sha256(path), "optimizer_updates": 600})
                print(f"completed {index}/{MAX_ATTEMPTS} {arm} seed{seed}", flush=True)
            except BaseException as exc:
                _append_attempt({"event": "failed", "index": index, "arm": arm,
                                 "seed": seed, "error_type": type(exc).__name__,
                                 "reason": "attempt_failed_no_retry"})
                raise
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                raise TimeoutError("executor wall budget exhausted after completed attempt")
    finally:
        _write_json_exclusive(WALL_PATH, {
            "schema": "norishio.issue48.executor-wall.v1",
            "executor_total_wall_seconds": time.perf_counter() - started,
            "measurement": "perf_counter around attempts and raw writes; excludes raw-only aggregation",
        })
    return aggregate_saved_runs()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "aggregate"))
    args = parser.parse_args()
    report = execute_once() if args.mode == "run" else aggregate_saved_runs()
    print(json.dumps({"status": report["status"],
                      "decision": report["decision"]["branch"],
                      "descriptor_sha256": report["descriptor_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["aggregate_saved_runs", "execute_once", "prepare_role_row"]
