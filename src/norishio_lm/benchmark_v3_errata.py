"""Read-only post-merge audit for benchmark-v3 protocol errata.

This module consumes retained JSON artifacts only.  It does not import the
model runner, open benchmark splits, load checkpoints, or perform inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmark_v3_contract import SELECTION_CONTRACT
from .benchmark_v3_tournament import ARM_COUNTS, SEEDS


ARM_IDS = tuple(ARM_COUNTS)
MAIN_ARMS = ("H0L0", "H0L1", "H1L0", "H1L1")
_FACTORS = ("participant", "time", "event", "operator")
_FACTOR_WIDTHS = {"participant": 6, "time": 6, "event": 4, "operator": 4}
EXPECTED_FINAL_SCHEMA = "norishio.issue36.final-result.v1"
EXPECTED_DEVELOPMENT_SCHEMA = "norishio.issue36.development-run.v1"
_METRIC_PATHS = (
    SELECTION_CONTRACT["primary"],
    *SELECTION_CONTRACT["tie_breakers"][:-1],
)


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


def _metric(result: Mapping[str, Any], path: str) -> float:
    derived = SELECTION_CONTRACT.get("derived_metrics", {}).get(path)
    if derived is not None:
        if derived.get("reducer") != "unweighted_arithmetic_mean":
            raise ValueError(f"unsupported metric reducer: {path}")
        source_paths = derived.get("source_paths")
        if not isinstance(source_paths, list) or not source_paths:
            raise ValueError(f"derived metric sources are missing: {path}")
        values = [_metric(result, source_path) for source_path in source_paths]
        return sum(values) / len(values)
    value: Any = result
    for field in path.split("."):
        if not isinstance(value, Mapping) or field not in value:
            raise ValueError(f"missing metric path: {path}")
        value = value[field]
    return _finite(value, path)


def _factorial(values: Mapping[str, float]) -> dict[str, float]:
    return {
        "activation_main_effect": (
            (values["H1L0"] + values["H1L1"]) / 2
            - (values["H0L0"] + values["H0L1"]) / 2
        ),
        "locality_main_effect": (
            (values["H0L1"] + values["H1L1"]) / 2
            - (values["H0L0"] + values["H1L0"]) / 2
        ),
        "interaction": (
            (values["H1L1"] - values["H1L0"])
            - (values["H0L1"] - values["H0L0"])
        ),
    }


def _classify_interventions(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify retained fixed-class probes without inventing omitted vectors."""

    scored = result.get("intermediate_probability_intervention")
    if not isinstance(scored, Mapping):
        return {
            "baseline_argmax_classification": "unavailable",
            "reason": "intermediate_probability_intervention is absent",
            "fixed_class_zero": None,
        }
    by_factor = scored.get("by_factor")
    examples = scored.get("examples")
    if not isinstance(by_factor, Mapping) or not isinstance(examples, list):
        raise ValueError("retained intervention summary is incomplete")
    rows = scored.get("rows")
    if type(rows) is not int or rows != len(_FACTORS) or rows != len(examples):
        raise ValueError("retained intervention summary must contain exactly four examples")
    if set(by_factor) != set(_FACTORS):
        raise ValueError("retained intervention summary must cover each factor exactly once")
    observed_factors: list[str] = []
    for index, item in enumerate(examples):
        if not isinstance(item, Mapping) or item.get("factor") not in _FACTORS:
            raise ValueError(f"retained intervention example[{index}] has an unknown factor")
        factor = str(item["factor"])
        observed_factors.append(factor)
        vector = item.get("one_hot")
        if (
            not isinstance(vector, list)
            or len(vector) != _FACTOR_WIDTHS[factor]
            or vector[0] != 1.0
            or any(value != 0.0 for value in vector[1:])
        ):
            raise ValueError(f"retained intervention example[{index}] is not class-zero one-hot")
    if sorted(observed_factors) != sorted(_FACTORS):
        raise ValueError("retained intervention examples must cover each factor exactly once")
    for factor in _FACTORS:
        item = by_factor[factor]
        if not isinstance(item, Mapping):
            raise ValueError(f"retained intervention factor summary is invalid: {factor}")
        target = item.get("target_changed")
        if (
            not isinstance(target, Mapping)
            or type(target.get("correct")) is not int
            or type(target.get("denominator")) is not int
            or target["denominator"] != 1
            or not 0 <= target["correct"] <= 1
        ):
            raise ValueError(f"retained intervention target count is invalid: {factor}")
    target_changed = sum(
        int(item.get("target_changed", {}).get("correct", 0))
        for item in by_factor.values() if isinstance(item, Mapping)
    )
    denominators = sum(
        int(item.get("target_changed", {}).get("denominator", 0))
        for item in by_factor.values() if isinstance(item, Mapping)
    )
    return {
        "baseline_argmax_classification": "unavailable",
        "reason": (
            "the scored artifact retains fixed one-hot vectors and derived outcomes "
            "but omits baseline_probabilities, so baseline argmax cannot be reconstructed"
        ),
        "fixed_class_zero": {
            "rows": rows,
            "target_changed": target_changed,
            "target_change_denominator": denominators,
            "all_retained_one_hots_select_class_zero": True,
        },
    }


def _audit_evaluations(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    validation_issues: list[str] = []
    for index, evaluation in enumerate(evaluations):
        arm, seed = evaluation.get("arm"), evaluation.get("seed")
        if arm not in ARM_IDS or type(seed) is not int or seed not in SEEDS:
            validation_issues.append(f"evaluation[{index}] has unknown arm/seed: {arm!r}/{seed!r}")
            continue
        key = (str(arm), seed)
        if key in by_key:
            validation_issues.append(f"duplicate evaluation: {arm}/{seed}")
            continue
        by_key[key] = evaluation
    expected = {(arm, seed) for arm in ARM_IDS for seed in SEEDS}
    for arm, seed in sorted(expected - set(by_key)):
        validation_issues.append(f"missing evaluation: {arm}/{seed}")

    per_run: list[dict[str, Any]] = []
    arm_values: dict[str, dict[str, list[float]]] = {
        arm: {path: [] for path in _METRIC_PATHS} for arm in ARM_IDS
    }
    intervention_totals = {
        "main_arms": {"runs": 0, "rows": 0, "target_changed": 0},
        "all_arms": {"runs": 0, "rows": 0, "target_changed": 0},
    }
    baseline_argmax_reasons: set[str] = set()
    all_fixed_zero = True
    for arm in ARM_IDS:
        for seed in SEEDS:
            evaluation = by_key.get((arm, seed))
            if evaluation is None:
                continue
            result = evaluation.get("result")
            if not isinstance(result, Mapping):
                validation_issues.append(f"evaluation result is missing: {arm}/{seed}")
                continue
            try:
                metrics = {path: _metric(result, path) for path in _METRIC_PATHS}
            except ValueError as exc:
                validation_issues.append(f"{arm}/{seed}: {exc}")
                continue
            try:
                intervention = _classify_interventions(result)
            except ValueError as exc:
                validation_issues.append(f"{arm}/{seed}: {exc}")
                intervention = {
                    "reason": f"retained intervention evidence is invalid: {exc}",
                    "fixed_class_zero": None,
                }
            fixed = intervention["fixed_class_zero"]
            if fixed is not None:
                for scope in ("all_arms", "main_arms"):
                    if scope == "main_arms" and arm not in MAIN_ARMS:
                        continue
                    intervention_totals[scope]["runs"] += 1
                    intervention_totals[scope]["rows"] += fixed["target_change_denominator"]
                    intervention_totals[scope]["target_changed"] += fixed["target_changed"]
                all_fixed_zero = all_fixed_zero and fixed["all_retained_one_hots_select_class_zero"]
            baseline_argmax_reasons.add(intervention["reason"])
            per_run.append({"arm": arm, "seed": seed, "metrics": metrics})
            for path, value in metrics.items():
                arm_values[arm][path].append(value)

    arms: dict[str, Any] = {}
    for arm in ARM_IDS:
        complete = len(arm_values[arm][_METRIC_PATHS[0]])
        arms[arm] = {
            "complete_seeds": complete,
            "three_seed_mean": (
                {path: sum(values) / len(SEEDS) for path, values in arm_values[arm].items()}
                if complete == len(SEEDS) and all(len(values) == len(SEEDS) for values in arm_values[arm].values())
                else None
            ),
        }
    eligible = [arm for arm in ARM_IDS if arms[arm]["three_seed_mean"] is not None]
    ranking = sorted(
        eligible,
        key=lambda arm: tuple(
            -arms[arm]["three_seed_mean"][path] for path in _METRIC_PATHS
        ) + (arm,),
    )
    factorial = {}
    for path in _METRIC_PATHS:
        if all(arms[arm]["three_seed_mean"] is not None for arm in MAIN_ARMS):
            factorial[path] = _factorial({
                arm: arms[arm]["three_seed_mean"][path] for arm in MAIN_ARMS
            })
        else:
            factorial[path] = None
    return {
        "expected_evaluations": len(expected),
        "observed_evaluations": len(evaluations),
        "validation_issues": validation_issues,
        "per_run": per_run,
        "arms": arms,
        "ranking": ranking,
        "factorial_effects": factorial,
        "intervention_audit": {
            "historical_intervention": "fixed_artificial_one_hot_class_zero",
            "baseline_argmax_classification": "unavailable",
            "reason": "; ".join(sorted(baseline_argmax_reasons)),
            "all_retained_one_hots_select_class_zero": (
                all_fixed_zero
                if intervention_totals["all_arms"]["runs"] == len(expected)
                else None
            ),
            **intervention_totals,
        },
    }


def build_protocol_errata_audit(
    final_result: str | Path,
    development_directory: str | Path,
    *,
    expected_final_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit retained final and development JSON without executing the benchmark."""

    final_path = Path(final_result)
    development_path = Path(development_directory)
    final_value, final_sha256 = _read_json(final_path)
    issues: list[str] = []
    if expected_final_sha256 is not None and final_sha256 != expected_final_sha256:
        issues.append(
            f"final source SHA-256 mismatch: expected {expected_final_sha256}, observed {final_sha256}"
        )
    if final_value.get("schema") != EXPECTED_FINAL_SCHEMA:
        issues.append(f"unexpected final schema: {final_value.get('schema')!r}")
    if final_value.get("status") != "complete":
        issues.append(f"final status is not complete: {final_value.get('status')!r}")
    final_evaluations = final_value.get("evaluations")
    if not isinstance(final_evaluations, list):
        final_evaluations = []
        issues.append("final evaluations are missing")

    development_sources: list[dict[str, str]] = []
    development_evaluations: list[dict[str, Any]] = []
    for path in sorted(development_path.glob("*.json")):
        value, sha256 = _read_json(path)
        development_sources.append({"path": path.as_posix(), "sha256": sha256})
        if value.get("schema") != EXPECTED_DEVELOPMENT_SCHEMA:
            issues.append(f"unexpected development schema: {path.name}: {value.get('schema')!r}")
            continue
        metrics = value.get("metrics")
        if not isinstance(metrics, Mapping):
            issues.append(f"development metrics are missing: {path.name}")
            continue
        development_evaluations.append({
            "arm": value.get("arm"), "seed": value.get("seed"), "result": metrics,
        })
    if len(development_sources) != len(ARM_IDS) * len(SEEDS):
        issues.append(
            f"expected 18 development artifacts, observed {len(development_sources)}"
        )

    development_audit = _audit_evaluations(development_evaluations)
    final_audit = _audit_evaluations(final_evaluations)
    issues.extend(
        f"development: {issue}" for issue in development_audit["validation_issues"]
    )
    issues.extend(f"final: {issue}" for issue in final_audit["validation_issues"])

    return {
        "schema": "norishio.issue39.benchmark-v3-protocol-errata.v1",
        "audit_mode": "read_only_saved_json_no_training_no_inference",
        "historical_selection_implementation": {
            "primary": "all.free_generation_exact.accuracy",
            "ranking": (
                final_value.get("selection", {}).get("ranking")
                if isinstance(final_value.get("selection"), Mapping) else None
            ),
        },
        "selection_contract": dict(SELECTION_CONTRACT),
        "sources": {
            "final": {
                "path": final_path.as_posix(),
                "sha256": final_sha256,
                "expected_sha256": expected_final_sha256,
                "sha256_matches_expected": (
                    final_sha256 == expected_final_sha256 if expected_final_sha256 is not None else None
                ),
                "schema": final_value.get("schema"),
                "status": final_value.get("status"),
            },
            "development": development_sources,
        },
        "validation_issues": issues,
        "development": development_audit,
        "final": final_audit,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--expected-final-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_protocol_errata_audit(
        args.final_result,
        args.development_directory,
        expected_final_sha256=args.expected_final_sha256,
    )
    _write_exclusive(args.output, audit)
    print(json.dumps({
        "output": str(args.output),
        "validation_issues": audit["validation_issues"],
        "development_ranking": audit["development"]["ranking"],
        "final_ranking": audit["final"]["ranking"],
    }, ensure_ascii=False, indent=2))
    return 0 if not audit["validation_issues"] else 1


__all__ = ["build_protocol_errata_audit", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
