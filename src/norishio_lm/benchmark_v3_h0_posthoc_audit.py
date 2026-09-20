"""Append-only post-hoc audit of the saved Issue #44 H0 raw runs.

This module intentionally has no model, checkpoint, fixture, or training path.
It reads the retained JSON run records and deterministically counts values that
are already present in those records.  The audit is diagnostic metadata and
cannot select a decision, threshold, or ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

RESULT_ROOT = Path(__file__).resolve().parents[2] / "docs" / "results" / "benchmark-v3-h0-confirmation"
AUDIT_PATH = RESULT_ROOT / "head-factor-audit.json"

SCHEMA = "norishio.issue44.h0-head-factor-posthoc-audit.v1"
RUN_SCHEMA = "norishio.issue44.h0-confirmation-run.v1"
# These are copied from the frozen descriptor and fixture manifest.  Keeping
# the binding literals local lets this raw-only audit run without importing
# torch or any checkpoint/model code.
PROTOCOL_SHA256 = "19c11e5fa7b5ae90dd2bc3b6ddca20366e6d2b9798b98e3ccf1c7c8bcdf10533"
FIXTURE_CONTENT_SHA256 = "296f32138f9ee448ca8fe975200047f53a6679343af35cbe7423d1e3435e1945"
ARMS = ("H1L1", "H1L1_ANCHOR")
SEEDS = (7, 17, 29)
FACTORS = ("event", "operator", "participant", "time")
SUPPORT_GROUPS = ("unseen_pair", "seen_pair/unseen_triple")
CONTROL_TYPES = ("alternate_class_one_hot", "alternate_class_donor_soft")
CONTROL_ORDER = {
    "same_class_soft_shape": 0,
    "alternate_class_one_hot": 1,
    "alternate_class_donor_soft": 2,
}
ATOMIC_DENOMINATOR = 96
SUPPORT_DENOMINATOR = 144
CONTROL_DENOMINATOR = 6


def _metric(correct: int, denominator: int) -> dict[str, Any]:
    return {
        "correct": correct,
        "count": denominator,
        "denominator": denominator,
        "rate": (correct / denominator if denominator else None),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "correct": None,
        "count": None,
        "denominator": None,
        "rate": None,
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _run_label(run: Mapping[str, Any]) -> str:
    return f"{run.get('arm')} seed{run.get('seed')}"


def _expected_run_order() -> list[str]:
    return [f"{arm} seed{seed}" for seed in SEEDS for arm in ARMS]


def _check_bindings(run: Mapping[str, Any], *, index: int) -> None:
    if run.get("schema") != RUN_SCHEMA:
        raise ValueError(f"saved run {index} has an unexpected schema")
    if run.get("status") != "complete":
        raise ValueError(f"saved run {index} is not complete")
    if run.get("descriptor_sha256") != PROTOCOL_SHA256:
        raise ValueError(f"saved run {index} descriptor binding mismatch")
    if run.get("fixture_content_digest_sha256") != FIXTURE_CONTENT_SHA256:
        raise ValueError(f"saved run {index} fixture binding mismatch")
    training = _require_mapping(run.get("training"), f"run {index}.training")
    if training.get("seed") != run.get("seed") or training.get("arm") != run.get("arm"):
        raise ValueError(f"saved run {index} training identity mismatch")


def load_saved_runs(root: Path = RESULT_ROOT) -> list[dict[str, Any]]:
    """Load only the six retained ``run-*.json`` artifacts in frozen order."""

    paths = sorted(Path(root).glob("run-*.json"))
    if len(paths) != len(_expected_run_order()):
        raise ValueError("Issue #44 post-hoc audit requires exactly six saved runs")
    try:
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("saved Issue #44 raw run is unreadable") from exc
    if [_run_label(run) for run in runs] != _expected_run_order():
        raise ValueError("saved runs differ from frozen run order")
    return runs


def _validate_ordinary_rows(run: Mapping[str, Any], *, run_index: int, expected_ids: list[str] | None) -> list[dict[str, Any]]:
    ordinary = _require_mapping(run.get("ordinary_confirmation"), f"run {run_index}.ordinary_confirmation")
    rows = ordinary.get("raw_confirmation_rows")
    if not isinstance(rows, list) or len(rows) != ATOMIC_DENOMINATOR:
        raise ValueError(f"run {run_index} must retain exactly 96 ordinary raw rows")
    ids: list[str] = []
    for row_index, raw in enumerate(rows):
        row = _require_mapping(raw, f"run {run_index}.ordinary row {row_index}")
        required = ("arm", "seed", "row_id", "support_group", "target_frame", "intermediate_frame")
        if any(key not in row for key in required):
            raise ValueError(f"run {run_index}.ordinary row {row_index} is missing raw fields")
        if row["arm"] != run["arm"] or row["seed"] != run["seed"]:
            raise ValueError(f"run {run_index}.ordinary row identity mismatch")
        if not isinstance(row["row_id"], str) or row["row_id"] in ids:
            raise ValueError(f"run {run_index}.ordinary row IDs are not unique")
        if row["support_group"] not in SUPPORT_GROUPS:
            raise ValueError(f"run {run_index}.ordinary row has an unknown support group")
        for frame_name in ("target_frame", "intermediate_frame"):
            frame = _require_mapping(row[frame_name], f"{frame_name} row {row_index}")
            if tuple(frame) != FACTORS:
                raise ValueError(f"{frame_name} row {row_index} factor order changed")
        ids.append(row["row_id"])
    if ids != sorted(ids):
        raise ValueError(f"run {run_index}.ordinary raw row order changed")
    if expected_ids is not None and ids != expected_ids:
        raise ValueError(f"run {run_index}.ordinary row identity order differs")
    return [dict(row) for row in rows]


def _validate_control_rows(run: Mapping[str, Any], *, run_index: int, expected_keys: list[tuple[str, str, str, str]] | None) -> list[dict[str, Any]]:
    interventions = _require_mapping(run.get("interventions"), f"run {run_index}.interventions")
    rows = interventions.get("control_records")
    if not isinstance(rows, list):
        raise ValueError(f"run {run_index} has no saved raw control_records")
    keys: list[tuple[str, str, str, str]] = []
    for row_index, raw in enumerate(rows):
        row = _require_mapping(raw, f"run {run_index}.control row {row_index}")
        required = ("control_type", "target_factor", "support_group", "row_id", "nontrivial_joint_success", "missing_donor")
        if any(key not in row for key in required):
            raise ValueError(f"run {run_index}.control row {row_index} is missing raw fields")
        key = (str(row["row_id"]), str(row["control_type"]), str(row["target_factor"]), str(row["support_group"]))
        if row["control_type"] not in CONTROL_ORDER or row["target_factor"] not in FACTORS or row["support_group"] not in SUPPORT_GROUPS:
            raise ValueError(f"run {run_index}.control row {row_index} has an unknown identity")
        if row["arm"] != run["arm"] or row["seed"] != run["seed"]:
            raise ValueError(f"run {run_index}.control row identity mismatch")
        keys.append(key)
    expected_sort = sorted(keys, key=lambda key: (key[0], CONTROL_ORDER[key[1]]))
    if keys != expected_sort:
        raise ValueError(f"run {run_index}.control raw row order changed")
    if expected_keys is not None and keys != expected_keys:
        raise ValueError(f"run {run_index}.control row identity order differs")
    return [dict(row) for row in rows]


def build_head_factor_audit(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute the post-hoc audit from saved raw run mappings only."""

    if len(runs) != 6:
        raise ValueError("Issue #44 post-hoc audit requires exactly six saved runs")
    labels = [_run_label(run) for run in runs]
    if labels != _expected_run_order():
        raise ValueError("saved runs differ from frozen run order")
    for index, run in enumerate(runs):
        _check_bindings(run, index=index)

    ordinary_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    control_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    ordinary_ids: list[str] | None = None
    control_keys: list[tuple[str, str, str, str]] | None = None
    for index, run in enumerate(runs):
        key = (str(run["arm"]), int(run["seed"]))
        ordinary_rows[key] = _validate_ordinary_rows(run, run_index=index, expected_ids=ordinary_ids)
        control_rows[key] = _validate_control_rows(run, run_index=index, expected_keys=control_keys)
        if ordinary_ids is None:
            ordinary_ids = [row["row_id"] for row in ordinary_rows[key]]
        if control_keys is None:
            control_keys = [
                (row["row_id"], row["control_type"], row["target_factor"], row["support_group"])
                for row in control_rows[key]
            ]

    atomic: dict[str, Any] = {}
    support: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    for arm in ARMS:
        atomic[arm] = {}
        support[arm] = {}
        controls[arm] = {}
        for seed in SEEDS:
            rows = ordinary_rows[(arm, seed)]
            atomic[arm][str(seed)] = {}
            for factor in FACTORS:
                correct = sum(row["intermediate_frame"][factor] == row["target_frame"][factor] for row in rows)
                atomic[arm][str(seed)][factor] = _metric(correct, ATOMIC_DENOMINATOR)
        for factor in FACTORS:
            support[arm][factor] = {}
            for support_group in SUPPORT_GROUPS:
                rows = [
                    row
                    for seed in SEEDS
                    for row in ordinary_rows[(arm, seed)]
                    if row["support_group"] == support_group
                ]
                if len(rows) != SUPPORT_DENOMINATOR:
                    raise ValueError(f"{arm}/{factor}/{support_group} does not have denominator 144")
                correct = sum(row["intermediate_frame"][factor] == row["target_frame"][factor] for row in rows)
                support[arm][factor][support_group] = _metric(correct, SUPPORT_DENOMINATOR)
            controls[arm][factor] = {}
            rows_by_control = {
                control: [
                    row
                    for seed in SEEDS
                    for row in control_rows[(arm, seed)]
                    if row["control_type"] == control and row["target_factor"] == factor
                ]
                for control in CONTROL_TYPES
            }
            for control, rows in rows_by_control.items():
                if not rows:
                    controls[arm][factor][control] = {
                        "nontrivial_joint_success": _unavailable("missing_saved_raw_control_records"),
                        "missing": _unavailable("missing_saved_raw_control_records"),
                    }
                    continue
                if len(rows) != CONTROL_DENOMINATOR:
                    raise ValueError(f"{arm}/{factor}/{control} does not have denominator 6")
                controls[arm][factor][control] = {
                    "nontrivial_joint_success": _metric(sum(row["nontrivial_joint_success"] is True for row in rows), CONTROL_DENOMINATOR),
                    "missing": _metric(sum(row["missing_donor"] is True for row in rows), CONTROL_DENOMINATOR),
                }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bindings": {
            "descriptor_sha256": PROTOCOL_SHA256,
            "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
            "all_saved_runs_match_bindings": True,
        },
        "scope": {
            "source": "saved docs/results/benchmark-v3-h0-confirmation/run-*.json raw records only",
            "ordinary_raw_rows_per_run": ATOMIC_DENOMINATOR,
            "run_order": _expected_run_order(),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "factors": list(FACTORS),
            "support_groups": list(SUPPORT_GROUPS),
            "control_types": list(CONTROL_TYPES),
        },
        "decision_policy": {
            "kind": "post_hoc_diagnostic",
            "uses_saved_raw_only": True,
            "changes_frozen_decision": False,
            "changes_frozen_thresholds": False,
            "changes_frozen_ranking": False,
            "frozen_decision_threshold_ranking_are_external": True,
        },
        "atomic_head": atomic,
        "support_aggregate": support,
        "controls": controls,
        "regeneration": {
            "deterministic": True,
            "input_record_order_checked": True,
            "json_encoding": "utf-8",
            "derived_from_raw_fields": ["intermediate_frame", "target_frame", "support_group", "nontrivial_joint_success", "missing_donor"],
            "raw_artifact_mutation": False,
        },
    }
    return result


def render_head_factor_audit(audit: Mapping[str, Any]) -> bytes:
    """Render the canonical UTF-8 bytes used by the tracked audit artifact."""

    if audit.get("schema") != SCHEMA:
        raise ValueError("unexpected post-hoc audit schema")
    return (json.dumps(audit, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def regenerate_head_factor_audit(root: Path = RESULT_ROOT) -> dict[str, Any]:
    """Load saved runs and regenerate the audit without writing any file."""

    return build_head_factor_audit(load_saved_runs(root))


audit_saved_runs = build_head_factor_audit
build_audit = build_head_factor_audit


__all__ = [
    "ARMS", "AUDIT_PATH", "ATOMIC_DENOMINATOR", "CONTROL_DENOMINATOR", "CONTROL_TYPES", "FACTORS",
    "FIXTURE_CONTENT_SHA256", "PROTOCOL_SHA256", "RESULT_ROOT", "SCHEMA", "SEEDS", "SUPPORT_DENOMINATOR",
    "SUPPORT_GROUPS", "audit_saved_runs", "build_audit", "build_head_factor_audit", "load_saved_runs",
    "regenerate_head_factor_audit", "render_head_factor_audit",
]
