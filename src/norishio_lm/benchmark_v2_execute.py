"""Execution driver for the frozen 18-run benchmark-v2 development tournament."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .benchmark_v2_checkpoint import save_tournament_checkpoint
from .benchmark_v2_final import checkpoint_path
from .benchmark_v2_protocol import (
    canonical_bytes, collect_terminal_records, complete_record, failed_record,
    record_path, validate_terminal_record, write_terminal_record,
)
from .benchmark_v2_runner import TrainingResult, evaluate_benchmark_v2, train_benchmark_v2
from .benchmark_v2_tournament import ARM_COUNTS, SEEDS, required_run_keys, load_tournament


Trainer = Callable[[str, int], TrainingResult]
DevelopmentEvaluator = Callable[[Any], Mapping[str, Any]]


def development_path(root: str | os.PathLike[str], arm: str, seed: int) -> Path:
    return Path(root) / "development" / f"{arm}-{seed}.json"


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
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


def execute_one(root: str | os.PathLike[str], arm: str, seed: int, *,
                trainer: Trainer = train_benchmark_v2,
                evaluator: DevelopmentEvaluator = evaluate_benchmark_v2) -> dict[str, Any]:
    """Train, score, checkpoint, and terminally record one frozen run."""
    output = Path(root)
    terminal = record_path(output, arm, seed)
    if terminal.exists():
        raise FileExistsError(str(terminal))
    if (output / "final-invocation").exists():
        raise FileExistsError("final invocation already consumed for this output root")
    phase = "training"
    try:
        training = trainer(arm, seed)
        if (training.arm, training.seed, training.trainable_parameters) != (
            arm, seed, ARM_COUNTS[arm]
        ):
            raise ValueError("trainer returned a run outside the frozen arm/seed contract")
        phase = "development-evaluation"
        report = dict(evaluator(training.model))
        phase = "checkpoint"
        checkpoint = checkpoint_path(output, arm, seed)
        save_tournament_checkpoint(
            checkpoint, training.model, arm=arm, seed=seed,
            schedule_sha256=training.schedule_sha256,
            initial_state_sha256=training.initial_state_sha256,
        )
        if training.final_state_sha256 != complete_record(
            checkpoint, arm=arm, seed=seed
        )["final_state_sha256"]:
            raise ValueError("trainer final state digest differs from checkpoint")
        phase = "development-report"
        _write_json_exclusive(development_path(output, arm, seed), {
            "schema": "norishio.issue34.development-run.v1",
            "arm": arm,
            "seed": seed,
            "trainable_parameters": training.trainable_parameters,
            "initial_state_sha256": training.initial_state_sha256,
            "final_state_sha256": training.final_state_sha256,
            "schedule_sha256": training.schedule_sha256,
            "loss": {
                "updates": len(training.losses),
                "initial": training.losses[0],
                "final": training.losses[-1],
                "minimum": min(training.losses),
            },
            "metrics": report,
        })
        phase = "terminal-record"
        record = complete_record(checkpoint, arm=arm, seed=seed)
        write_terminal_record(output, record)
        return record
    except Exception as exc:
        if not terminal.exists():
            record = failed_record(
                arm=arm, seed=seed, phase=phase, error_type=type(exc).__name__,
                redacted_message=f"frozen run failed during {phase}",
            )
            write_terminal_record(output, record)
        raise


def _metric_summary(report: Mapping[str, Any]) -> dict[str, float | None]:
    all_metrics = report["all"]
    balanced = [all_metrics["balanced_accuracy"][field]
                for field in ("participant", "time", "event", "operator")]
    locality = [item["preservation_rate"]
                for item in report["intervention_locality"]["by_factor"].values()]
    return {
        "free_generation_exact": all_metrics["free_generation_exact"]["accuracy"],
        "generation_frame_exact": all_metrics["generation_frame_exact"]["accuracy"],
        "triple_exact": all_metrics["participant_time_event_triple_exact"]["accuracy"],
        "mean_balanced_atomic_accuracy": sum(balanced) / len(balanced),
        "teacher_forced_byte_match": report["teacher_forced_bytes"]["byte_match_rate"],
        "intervention_non_target_preservation": sum(locality) / len(locality),
    }


def write_development_summary(root: str | os.PathLike[str],
                              records: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(root)
    per_run = []
    by_arm: dict[str, list[dict[str, float | None]]] = {arm: [] for arm in ARM_COUNTS}
    for record in records:
        row: dict[str, Any] = {"arm": record["arm"], "seed": record["seed"],
                               "status": record["status"]}
        if record["status"] == "complete":
            artifact = json.loads(development_path(
                output, record["arm"], record["seed"]
            ).read_text(encoding="utf-8"))
            metrics = _metric_summary(artifact["metrics"])
            row["metrics"] = metrics
            by_arm[record["arm"]].append(metrics)
        else:
            row.update({key: record[key] for key in ("phase", "error_type", "redacted_message")})
        per_run.append(row)
    arm_means: dict[str, Any] = {}
    for arm, values in by_arm.items():
        keys = next(iter(values), {}).keys()
        arm_means[arm] = {
            "complete_seeds": len(values),
            "three_seed_mean": {
                key: (sum(item[key] for item in values if item[key] is not None) / 3
                      if len(values) == 3 and all(item[key] is not None for item in values)
                      else None)
                for key in keys
            },
        }
    path = output / "development-summary.json"
    _write_json_exclusive(path, {
        "schema": "norishio.issue34.development-summary.v1",
        "aggregation": "unweighted arithmetic mean over exactly three seeds; null if any seed failed",
        "runs": per_run,
        "arms": arm_means,
    })
    return path


def execute_all(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Execute missing frozen runs sequentially and write the development summary."""
    output = Path(root)
    config = load_tournament()
    for arm, seed in required_run_keys(config):
        terminal = record_path(output, arm, seed)
        if terminal.exists():
            record = json.loads(terminal.read_text(encoding="utf-8"))
            validate_terminal_record(output, record)
            continue
        try:
            execute_one(output, arm, seed)
        except Exception:
            # execute_one publishes a strict visible failure record before returning.
            continue
    records = collect_terminal_records(output)
    write_development_summary(output, records)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--all", action="store_true", required=True,
                        help="execute all six arms and three frozen seeds")
    args = parser.parse_args(argv)
    records = execute_all(args.output_root)
    print(json.dumps({
        "terminal": len(records),
        "complete": sum(record["status"] == "complete" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "summary": str(args.output_root / "development-summary.json"),
    }, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "development_path", "execute_all", "execute_one", "main", "write_development_summary",
]


if __name__ == "__main__":
    raise SystemExit(main())
