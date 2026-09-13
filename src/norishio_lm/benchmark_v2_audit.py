"""Read-only split audit export for a consumed benchmark-v2 final result."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


ARM_IDS = ("A_G0", "A_G1", "B", "C", "D", "E")
SEEDS = (7, 17, 29)
EXPECTED_SOURCE_SCHEMA = "norishio.issue34.final-result.v1"
SPLITS = (
    ("unseen_pair", ("train_support_groups", "unseen_pair")),
    ("seen_pair", ("train_support_groups", "seen_pair")),
)
METRICS = (
    ("count", ("rows",)),
    ("free_exact", ("free_generation_exact", "accuracy")),
    ("frame_exact", ("generation_frame_exact", "accuracy")),
    ("triple_exact", ("triple_exact", "accuracy")),
    ("parse_coverage", ("parse_coverage",)),
)


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("final result must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _metric(group: Mapping[str, Any], path: Sequence[str]) -> tuple[int | float | None, str | None]:
    value: Any = group
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None, f"missing result field: {'.'.join(path)}"
        value = value[key]
    if path == ("rows",):
        if type(value) is not int or value < 0:
            return None, "result field rows is not a non-negative integer"
        return value, None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None, f"result field {'.'.join(path)} is not finite numeric data"
    return float(value), None


def _split_metrics(evaluation: Mapping[str, Any] | None, source_path: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {name: None for name, _ in METRICS}
    reasons: dict[str, str] = {}
    if evaluation is None:
        reason = "evaluation record is missing"
        return {**values, "missing_reasons": {name: reason for name in values}}
    result = evaluation.get("result")
    group: Any = result
    for key in source_path:
        group = group.get(key) if isinstance(group, Mapping) else None
    if not isinstance(group, Mapping):
        reason = f"result path {'/'.join(source_path)!r} is missing"
        return {**values, "missing_reasons": {name: reason for name in values}}
    for name, path in METRICS:
        value, reason = _metric(group, path)
        values[name] = value
        if reason is not None:
            reasons[name] = reason
    if reasons:
        values["missing_reasons"] = reasons
    return values


def build_split_audit(source: str | Path) -> dict[str, Any]:
    """Build a tracked audit from an existing final result without inference."""

    source_path = Path(source)
    result, source_sha256 = _read_json(source_path)
    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list):
        evaluations = []
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    issues: list[str] = []
    if result.get("schema") != EXPECTED_SOURCE_SCHEMA:
        issues.append(
            f"source schema is not {EXPECTED_SOURCE_SCHEMA!r}: {result.get('schema')!r}"
        )
    if result.get("status") != "complete":
        issues.append(f"source status is not 'complete': {result.get('status')!r}")
    for index, evaluation in enumerate(evaluations):
        if not isinstance(evaluation, Mapping):
            issues.append(f"evaluation[{index}] is not an object")
            continue
        arm, seed = evaluation.get("arm"), evaluation.get("seed")
        if not isinstance(arm, str) or arm not in ARM_IDS or type(seed) is not int or seed not in SEEDS:
            issues.append(f"evaluation[{index}] has unknown arm/seed: {arm!r}/{seed!r}")
            continue
        key = (arm, seed)
        if key in by_key:
            issues.append(f"duplicate evaluation: {arm}/{seed}")
            continue
        by_key[key] = evaluation
    expected = {(arm, seed) for arm in ARM_IDS for seed in SEEDS}
    for arm, seed in sorted(expected - set(by_key)):
        issues.append(f"missing evaluation: {arm}/{seed}")

    rows: list[dict[str, Any]] = []
    for arm in ARM_IDS:
        for seed in SEEDS:
            evaluation = by_key.get((arm, seed))
            rows.append({
                "arm": arm,
                "seed": seed,
                "splits": {
                    label: _split_metrics(evaluation, source_path)
                    for label, source_path in SPLITS
                },
            })
    return {
        "schema": "norishio.issue34.split-audit.v1",
        "source": {
            "path": source_path.as_posix(),
            "sha256": source_sha256,
            "schema": result.get("schema"),
            "status": result.get("status"),
        },
        "expected_evaluations": len(expected),
        "observed_evaluations": len(evaluations),
        "missing_value_policy": "null with missing_reasons entry",
        "split_source_paths": {
            label: "result." + ".".join(source_path)
            for label, source_path in SPLITS
        },
        "validation_issues": issues,
        "evaluations": rows,
    }


def _display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(audit: Mapping[str, Any]) -> str:
    """Render a compact, deterministic split table from an audit object."""

    source = audit["source"]
    lines = [
        "# Benchmark v2 split audit",
        "",
        "This audit reads the consumed final result JSON only; it performs no inference or benchmark evaluation.",
        "",
        f"- source schema: `{source.get('schema')}`",
        f"- source SHA-256: `{source.get('sha256')}`",
        f"- expected evaluations: {audit.get('expected_evaluations')}; observed: {audit.get('observed_evaluations')}",
        "- `seen_pair` is the pair-known, triple-unseen group.",
        "- Missing values are rendered as `null` and explained in the tracked JSON `missing_reasons` fields.",
        "",
        "| Arm | Seed | Split | Count | Free exact | Frame exact | Triple exact | Parse coverage |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation in audit.get("evaluations", []):
        for split in ("unseen_pair", "seen_pair"):
            metrics = evaluation["splits"][split]
            lines.append(
                f"| {evaluation['arm']} | {evaluation['seed']} | {split} | "
                f"{_display(metrics['count'])} | {_display(metrics['free_exact'])} | "
                f"{_display(metrics['frame_exact'])} | {_display(metrics['triple_exact'])} | "
                f"{_display(metrics['parse_coverage'])} |"
            )
    issues = audit.get("validation_issues", [])
    lines.extend(["", "Validation issues: " + ("; ".join(issues) if issues else "none") + ".", ""])
    return "\n".join(lines)


def export_split_audit(source: str | Path, json_path: str | Path, markdown_path: str | Path) -> dict[str, Any]:
    """Read source once and write deterministic tracked JSON and Markdown exports."""

    audit = build_split_audit(source)
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(render_markdown(audit), encoding="utf-8")
    return audit


__all__ = ["build_split_audit", "export_split_audit", "render_markdown"]
