"""Deterministic scheduling and terminal records for tournament execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from .benchmark_v2_checkpoint import checkpoint_file_sha256, load_tournament_checkpoint
from .benchmark_v2_tournament import (
    ARM_COUNTS,
    BENCHMARK_DIGEST,
    EXPECTED_CONFIG_SHA256,
    SEEDS,
    load_tournament,
    required_run_keys,
    validate_terminal_runs,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def batch_schedule(*, seed: int, train_rows: int = 384,
                   steps: int = 600, batch_size: int = 16) -> list[list[int]]:
    """Create the frozen with-replacement CPU schedule from the run seed."""
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    if type(train_rows) is not int or train_rows != 384:
        raise ValueError("benchmark-v2 requires exactly 384 training rows")
    if type(steps) is not int or steps != 600 or type(batch_size) is not int or batch_size != 16:
        raise ValueError("training schedule differs from the frozen budget")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(train_rows, (steps, batch_size), generator=generator).tolist()


def schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    if (not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)) or
            len(schedule) != 600):
        raise ValueError("schedule must contain 600 steps")
    normalized: list[list[int]] = []
    for batch in schedule:
        if (not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)) or
                len(batch) != 16 or any(type(index) is not int or not 0 <= index < 384 for index in batch)):
            raise ValueError("every schedule batch must contain 16 valid train indices")
        normalized.append(list(batch))
    return hashlib.sha256(canonical_bytes(normalized)).hexdigest()


def failed_record(*, arm: str, seed: int, phase: str,
                  error_type: str, redacted_message: str) -> dict[str, Any]:
    if arm not in ARM_COUNTS or seed not in SEEDS:
        raise ValueError("unknown tournament arm or seed")
    values = {"phase": phase, "error_type": error_type, "redacted_message": redacted_message}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("failure fields must be nonempty strings")
    if len(redacted_message) > 500 or "\n" in redacted_message or "\r" in redacted_message:
        raise ValueError("redacted failure message must be one short line")
    return {
        "arm": arm,
        "seed": seed,
        "status": "failed",
        "tournament_config_sha256": EXPECTED_CONFIG_SHA256,
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "trainable_parameters": ARM_COUNTS[arm],
        **values,
    }


def complete_record(checkpoint_path: str | os.PathLike[str], *, arm: str,
                    seed: int) -> dict[str, Any]:
    payload = load_tournament_checkpoint(checkpoint_path, arm=arm, seed=seed)
    metadata = payload["metadata"]
    return {
        "arm": arm,
        "seed": seed,
        "status": "complete",
        **metadata,
        "checkpoint_file_sha256": checkpoint_file_sha256(checkpoint_path),
    }


def record_path(root: str | os.PathLike[str], arm: str, seed: int) -> Path:
    if arm not in ARM_COUNTS or seed not in SEEDS:
        raise ValueError("unknown tournament arm or seed")
    return Path(root) / "runs" / f"{arm}-{seed}.json"


def validate_terminal_record(root: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Authenticate one strict record, including its checkpoint when complete."""
    failure_keys = {
        "arm", "seed", "status", "tournament_config_sha256",
        "benchmark_content_digest_sha256", "trainable_parameters", "phase",
        "error_type", "redacted_message",
    }
    complete_keys = {
        "arm", "seed", "status", "tournament_config_sha256",
        "benchmark_content_digest_sha256", "trainable_parameters",
        "initial_state_sha256", "final_state_sha256", "schedule_sha256",
        "checkpoint_file_sha256",
    }
    if record.get("status") == "failed":
        if set(record) != failure_keys:
            raise ValueError("failed terminal record has unexpected fields")
        # Rebuild to enforce the public constructor's redaction constraints.
        if dict(record) != failed_record(
            arm=record["arm"], seed=record["seed"], phase=record["phase"],
            error_type=record["error_type"], redacted_message=record["redacted_message"],
        ):
            raise ValueError("failed terminal record differs from frozen bindings")
    elif record.get("status") == "complete":
        if set(record) != complete_keys:
            raise ValueError("complete terminal record has unexpected fields")
        checkpoint = Path(root) / "checkpoints" / f"{record['arm']}-{record['seed']}.pt"
        if dict(record) != complete_record(checkpoint, arm=record["arm"], seed=record["seed"]):
            raise ValueError("complete terminal record does not authenticate its checkpoint")
    else:
        raise ValueError("run status must be complete or failed")
    validate_terminal_runs_partial([record], load_tournament())


def write_terminal_record(root: str | os.PathLike[str], record: Mapping[str, Any]) -> Path:
    """Publish exactly one immutable terminal record for an arm/seed."""
    validate_terminal_record(root, record)
    target = record_path(root, record["arm"], record["seed"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    payload = canonical_bytes(dict(record)) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
            os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


def validate_terminal_runs_partial(records: Sequence[Mapping[str, Any]],
                                   config: Mapping[str, Any]) -> None:
    """Apply the full validator by filling missing keys with inert failed records."""
    expected = required_run_keys(config)
    provided = {(record.get("arm"), record.get("seed")) for record in records}
    if len(provided) != len(records) or not provided <= set(expected):
        raise ValueError("unknown or duplicate partial terminal record")
    filled = list(records)
    for arm, seed in expected:
        if (arm, seed) not in provided:
            filled.append(failed_record(
                arm=arm, seed=seed, phase="validation-placeholder",
                error_type="NotRun", redacted_message="partial validation placeholder",
            ))
    validate_terminal_runs(filled, config)


def collect_terminal_records(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    config = load_tournament()
    directory = Path(root) / "runs"
    expected = required_run_keys(config)
    expected_names = {f"{arm}-{seed}.json" for arm, seed in expected}
    actual_names = {path.name for path in directory.glob("*.json")} if directory.exists() else set()
    if actual_names != expected_names:
        raise ValueError("terminal record files are incomplete or contain extras")
    records = [json.loads(record_path(root, arm, seed).read_text(encoding="utf-8"))
               for arm, seed in expected]
    for record in records:
        validate_terminal_record(root, record)
    validate_terminal_runs(records, config)
    return records


__all__ = [
    "batch_schedule", "canonical_bytes", "collect_terminal_records", "complete_record",
    "failed_record", "record_path", "schedule_sha256", "validate_terminal_record",
    "validate_terminal_runs_partial",
    "write_terminal_record",
]
