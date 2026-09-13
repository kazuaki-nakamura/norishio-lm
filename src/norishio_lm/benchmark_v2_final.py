"""Single-invocation final-holdout gate for the frozen Issue #34 tournament."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .benchmark_v2_checkpoint import checkpoint_file_sha256, load_tournament_checkpoint
from .benchmark_v2_protocol import canonical_bytes, collect_terminal_records
from .benchmark_v2_tournament import BENCHMARK_DIGEST


Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]], Any]
_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark_v2" / "benchmark.py"


def _fixture_module() -> Any:
    spec = importlib.util.spec_from_file_location("norishio_benchmark_v2_final_fixture", _BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark fixture: {_BENCHMARK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_path(root: str | os.PathLike[str], arm: str, seed: int) -> Path:
    return Path(root) / "checkpoints" / f"{arm}-{seed}.pt"


def _atomic_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
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
    from .benchmark_v2_runner import _evaluate_rows

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
        raise PermissionError("final holdout requires explicit --evaluate-final")
    if evaluator is None:
        evaluator = _default_evaluator
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    output_root = Path(root)
    records = collect_terminal_records(output_root)
    complete = _validate_complete_checkpoints(output_root, records)
    record_digest = hashlib.sha256(canonical_bytes(records)).hexdigest()

    invocation = output_root / "final-invocation"
    invocation.parent.mkdir(parents=True, exist_ok=True)
    invocation.mkdir(exist_ok=False)
    _atomic_exclusive_json(invocation / "marker.json", {
        "schema": "norishio.issue34.final-invocation.v1",
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
            benchmark.build(), "final-holdout", evaluate_final=True,
            manifest_digest=BENCHMARK_DIGEST,
        )
        evaluations = []
        for record, payload in complete:
            evaluations.append({
                "arm": record["arm"],
                "seed": record["seed"],
                "result": evaluator(record, payload, rows),
            })
        result = {
            "schema": "norishio.issue34.final-result.v1",
            "status": "complete",
            "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
            "terminal_records_sha256": record_digest,
            "rows": len(rows),
            "evaluations": evaluations,
            "failed_runs": [dict(record) for record in records if record["status"] == "failed"],
        }
        _atomic_exclusive_json(invocation / "result.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema": "norishio.issue34.final-result.v1",
            "status": "failed",
            "error_type": type(exc).__name__,
            "redacted_message": "final evaluation failed after invocation was consumed",
        }
        _atomic_exclusive_json(invocation / "result.json", failure)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Consume the frozen tournament final holdout once.")
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
