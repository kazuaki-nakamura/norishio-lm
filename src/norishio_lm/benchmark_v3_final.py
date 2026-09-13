"""Single-invocation final-confirmation gate for the frozen Issue #36 tournament."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping, Sequence

from . import benchmark_v3_data as benchmark
from .benchmark_v3_checkpoint import checkpoint_file_sha256, load_tournament_checkpoint
from .benchmark_v3_protocol import canonical_bytes, collect_terminal_records
from .benchmark_v3_tournament import ARM_COUNTS, BENCHMARK_DIGEST, SEEDS


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]], Any]
def _fixture_module() -> Any:
    return benchmark


def checkpoint_path(root: str | os.PathLike[str], arm: str, seed: int) -> Path:
    return Path(root) / "checkpoints" / f"{arm}-{seed}.pt"


def _atomic_exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    payload = canonical_bytes(dict(value)) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _publish_invocation_marker(root: Path, marker: Mapping[str, Any]) -> Path:
    """Publish the consumed directory only after its marker is complete.

    A temporary sibling keeps a failed marker write from leaving a directory
    that looks consumed but has no marker.  The final rename is exclusive on
    the supported Windows and POSIX filesystems because a successful publish
    always leaves a non-empty destination directory.
    """

    root.mkdir(parents=True, exist_ok=True)
    invocation = root / "final-invocation"
    temporary = Path(tempfile.mkdtemp(prefix=".final-invocation.", dir=root))
    try:
        _atomic_exclusive_json(temporary / "marker.json", marker)
        if invocation.exists():
            raise FileExistsError(str(invocation))
        os.rename(temporary, invocation)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return invocation


_SELECTION_METRICS = (
    "free_generation_exact.accuracy",
    "triple_exact.accuracy",
    "pair_exact.accuracy",
    "mean_balanced_atomic_accuracy",
    "exact_target_text.accuracy",
)


def _selection_metric(result: Mapping[str, Any], name: str) -> float:
    if name == "mean_balanced_atomic_accuracy":
        values = result["all"]["atomic_balanced_accuracy"]
        values = [values[field] for field in ("participant", "time", "event", "operator")]
        value = sum(values) / len(values)
    else:
        group, field = name.split(".", 1)
        value = result["all"][group][field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"selection metric {name} must be finite and numeric")
    return float(value)


def _selection_summary(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate frozen selection metrics and expose the tie-break trace.

    Custom evaluators used by protocol tests may intentionally return a small
    sentinel result.  Such results remain valid final-gate results, but do not
    claim a selection ranking until the frozen metrics are available.
    """

    try:
        by_arm: dict[str, list[Mapping[str, Any]]] = {arm: [] for arm in ARM_COUNTS}
        seen: set[tuple[str, int]] = set()
        for evaluation in evaluations:
            arm = evaluation["arm"]
            seed = evaluation["seed"]
            if arm not in by_arm or (arm, seed) in seen:
                raise ValueError("duplicate or unknown final evaluation arm")
            seen.add((arm, seed))
            by_arm[arm].append(evaluation)
        means: dict[str, dict[str, Any]] = {}
        for arm, rows in by_arm.items():
            seeds = sorted(row["seed"] for row in rows)
            if seeds != sorted(SEEDS):
                means[arm] = {"complete_seeds": len(rows), "three_seed_mean": None}
                continue
            values = {
                metric: [_selection_metric(row["result"], metric) for row in rows]
                for metric in _SELECTION_METRICS
            }
            means[arm] = {
                "complete_seeds": len(rows),
                "three_seed_mean": {
                    metric: sum(metric_values) / len(SEEDS)
                    for metric, metric_values in values.items()
                },
            }
        eligible = [arm for arm, value in means.items() if value["three_seed_mean"] is not None]
        ranking = sorted(
            eligible,
            key=lambda arm: tuple(
                [-means[arm]["three_seed_mean"][metric] for metric in _SELECTION_METRICS]
                + [arm]
            ),
        )

        trace: list[dict[str, Any]] = []
        primary = _SELECTION_METRICS[0]
        primary_groups: dict[float, list[str]] = {}
        for arm in eligible:
            value = means[arm]["three_seed_mean"][primary]
            primary_groups.setdefault(value, []).append(arm)
        trace.append({
            "stage": "primary",
            "metric": primary,
            "groups": [
                {"value": value, "arms": sorted(arms)}
                for value, arms in sorted(primary_groups.items(), key=lambda item: -item[0])
            ],
        })
        tied_groups = [sorted(arms) for arms in primary_groups.values() if len(arms) > 1]
        for index, metric in enumerate(_SELECTION_METRICS[1:], start=1):
            next_groups: list[list[str]] = []
            for candidates in tied_groups:
                scores = {arm: means[arm]["three_seed_mean"][metric] for arm in candidates}
                best = max(scores.values())
                survivors = sorted(arm for arm, value in scores.items() if value == best)
                trace.append({
                    "stage": f"tie_break_{index}",
                    "metric": metric,
                    "candidates": candidates,
                    "scores": scores,
                    "survivors": survivors,
                })
                if len(survivors) > 1:
                    next_groups.append(survivors)
            tied_groups = next_groups
            if not tied_groups:
                break
        if tied_groups:
            trace.append({
                "stage": f"tie_break_{len(_SELECTION_METRICS)}",
                "metric": "arm_id_ascending",
                "candidates": tied_groups,
                "survivors": [group[0] for group in tied_groups],
            })
        return {
            "schema": "norishio.issue36.selection.v1",
            "status": "complete",
            "primary_metric": primary,
            "tie_breakers": [*_SELECTION_METRICS[1:], "arm_id_ascending"],
            "three_seed_means": means,
            "ranking": ranking,
            "tie_break_trace": trace,
        }
    except (KeyError, TypeError, ValueError):
        return {
            "schema": "norishio.issue36.selection.v1",
            "status": "unavailable",
            "reason": "frozen selection metrics unavailable",
        }


def _validate_complete_checkpoints(root: Path, records: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    validated: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for record in records:
        if record["status"] != "complete":
            continue
        path = checkpoint_path(root, record["arm"], record["seed"])
        if checkpoint_file_sha256(path) != record["checkpoint_file_sha256"]:
            raise ValueError("terminal record checkpoint file hash mismatch")
        payload = load_tournament_checkpoint(path, arm=record["arm"], seed=record["seed"])
        for name in (
            "tournament_config_sha256", "benchmark_content_digest_sha256",
            "trainable_parameters", "initial_state_sha256", "final_state_sha256",
            "schedule_sha256",
        ):
            if payload["metadata"].get(name) != record.get(name):
                raise ValueError(f"terminal record checkpoint {name} mismatch")
        validated.append((record, payload))
    return validated


def _default_evaluator(_record: Mapping[str, Any], payload: Mapping[str, Any],
                       rows: Sequence[Mapping[str, Any]]) -> Any:
    from .benchmark_v3_runner import _evaluate_rows

    return _evaluate_rows(payload["model"], rows, intervention_pool=rows)


def evaluate_final_once(root: str | os.PathLike[str], *, evaluate_final: bool,
                        evaluator: Evaluator | None = None) -> dict[str, Any]:
    """Validate every run, consume the one invocation, then expose final rows.

    The invocation directory is the durable one-shot marker.  It is created
    atomically before importing/building the benchmark bundle or requesting
    final rows.  A callback failure is recorded inside the marker directory and
    does not permit another invocation.
    """
    if evaluate_final is not True:
        raise PermissionError("final-confirmation requires explicit --evaluate-final")
    using_default_evaluator = evaluator is None
    if evaluator is None:
        evaluator = _default_evaluator
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    output_root = Path(root)
    records = collect_terminal_records(output_root)
    complete = _validate_complete_checkpoints(output_root, records)
    if len(complete) != len(ARM_COUNTS) * len(SEEDS):
        raise ValueError("all 18 arm/seed checkpoints must complete before final-confirmation")
    record_digest = hashlib.sha256(canonical_bytes(records)).hexdigest()

    invocation = _publish_invocation_marker(output_root, {
        "schema": "norishio.issue36.final-invocation.v1",
        "status": "consumed",
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "terminal_records_sha256": record_digest,
        "terminal": len(records),
        "complete": len(complete),
        "failed": len(records) - len(complete),
    })

    try:
        # Delayed import and construction are part of the access boundary.
        benchmark = _fixture_module()
        rows = benchmark.evaluation_rows(
            benchmark.build(), "final-confirmation", evaluate_final=True,
            manifest_digest=BENCHMARK_DIGEST,
        )
        evaluations = []
        for record, payload in complete:
            evaluations.append({
                "arm": record["arm"],
                "seed": record["seed"],
                "result": evaluator(record, payload, rows),
            })
        selection = _selection_summary(evaluations)
        expected_evaluations = len(ARM_COUNTS) * len(SEEDS)
        complete_ranking = (
            selection.get("status") == "complete"
            and len(evaluations) == expected_evaluations
            and len(selection.get("ranking", [])) == len(ARM_COUNTS)
            and set(selection.get("ranking", [])) == set(ARM_COUNTS)
        )
        if using_default_evaluator and not complete_ranking:
            raise ValueError("default final evaluator did not produce frozen selection metrics")
        result = {
            "schema": "norishio.issue36.final-result.v1",
            "status": "complete",
            "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
            "terminal_records_sha256": record_digest,
            "rows": len(rows),
            "evaluations": evaluations,
            "failed_runs": [dict(record) for record in records if record["status"] == "failed"],
            "selection": selection,
        }
        result_digest = _atomic_exclusive_json(invocation / "result.json", result)
        _atomic_exclusive_json(invocation / "result-attestation.json", {
            "schema": "norishio.issue36.result-attestation.v1",
            "result_sha256": result_digest,
            "result_schema": result["schema"],
            "terminal_records_sha256": record_digest,
            "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        })
        return result
    except Exception as exc:
        if not (invocation / "result.json").exists():
            failure = {
                "schema": "norishio.issue36.final-result.v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "redacted_message": "final evaluation failed after invocation was consumed",
            }
            failure_digest = _atomic_exclusive_json(invocation / "result.json", failure)
            _atomic_exclusive_json(invocation / "result-attestation.json", {
                "schema": "norishio.issue36.result-attestation.v1",
                "result_sha256": failure_digest,
                "result_schema": failure["schema"],
                "terminal_records_sha256": record_digest,
                "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
            })
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consume the frozen tournament final-confirmation once.")
    parser.add_argument("--evaluate-final", action="store_true", required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="exclusive tournament output root containing runs/ and checkpoints/")
    args = parser.parse_args(argv)
    result = evaluate_final_once(args.root, evaluate_final=args.evaluate_final)
    print(json.dumps({
        "status": result["status"], "rows": result["rows"],
        "evaluations": len(result["evaluations"]),
        "failed_runs": len(result["failed_runs"]),
        "result": str(args.root / "final-invocation" / "result.json"),
    }, ensure_ascii=False, indent=2))
    return 0


__all__ = ["Evaluator", "checkpoint_path", "evaluate_final_once", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
