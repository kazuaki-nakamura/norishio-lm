"""Bounded executor and aggregation for the Issue #44 confirmation run."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .benchmark_v3_checkpoint import state_sha256
from .benchmark_v3_h0_intervention import evaluate_h0_interventions
from .benchmark_v3_h0_metrics import CONTROL_TYPES, aggregate_control_records
from .benchmark_v3_h0_protocol import (
    FIXTURE_CONTENT_SHA256,
    MAX_ATTEMPTS,
    MAX_UPDATES,
    MAX_WALL_SECONDS,
    PROTOCOL_SHA256,
    load_protocol,
    preflight_protocol,
)
from .benchmark_v3_h0_runner import (
    evaluate_h0_confirmation,
    evaluate_h0_constant_source,
    train_h0_arm,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "results" / "benchmark-v3-h0-confirmation"
RUN_SCHEMA = "norishio.issue44.h0-confirmation-run.v1"
SUMMARY_SCHEMA = "norishio.issue44.h0-confirmation-summary.v1"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_manifest(destination: Path) -> None:
    artifacts = sorted(
        path for path in destination.iterdir()
        if path.is_file() and path.name != "artifact-manifest.json"
    )
    _write_json(destination / "artifact-manifest.json", {
        "schema": "norishio.issue44.h0-confirmation-artifact-manifest.v1",
        "descriptor_sha256": PROTOCOL_SHA256,
        "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
        "artifacts": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        },
    })


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def _rate(count: int, denominator: int) -> dict[str, int | float | None]:
    return {"count": count, "denominator": denominator, "rate": count / denominator if denominator else None}


def _metric(report: Mapping[str, Any], group: str, name: str) -> float:
    container = report[group] if group in report else report["groups"][group]
    value = container[name]["accuracy"]
    if type(value) not in (int, float):
        raise ValueError(f"missing numeric {group}.{name}.accuracy")
    return float(value)


def _run_label(arm: str, seed: int) -> str:
    return f"{arm} seed{seed}"


def _parse_run_label(value: Any) -> tuple[str, int]:
    if not isinstance(value, str) or " seed" not in value:
        raise ValueError("invalid frozen run-order entry")
    arm, seed_text = value.rsplit(" seed", 1)
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError("invalid frozen run-order seed") from exc
    if arm not in ("H1L1", "H1L1_ANCHOR") or seed not in (7, 17, 29):
        raise ValueError("invalid frozen run-order arm or seed")
    return arm, seed


def _aggregate_swaps(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scheduled = len(records)
    if scheduled != 24:
        raise ValueError("source swaps require exactly 24 scheduled observations per arm")
    comparable = sum(not row["baseline_parse_failed"] and not row["changed_parse_failed"] for row in records)
    correct_denominator = sum(not row["changed_parse_failed"] for row in records)
    result = {
        "scheduled_count": scheduled,
        "parse_failure": _rate(sum(row["baseline_parse_failed"] or row["changed_parse_failed"] for row in records), scheduled),
        "target_changed": _rate(sum(bool(row["target_changed"]) for row in records), scheduled),
        "non_target_preserved": _rate(sum(bool(row["non_target_preserved"]) for row in records), scheduled),
        "exact_target_text": _rate(sum(bool(row["exact_target_text"]) for row in records), scheduled),
        "target_change_correct": _rate(sum(bool(row["target_change_correct"]) for row in records), correct_denominator),
        "comparable_count": comparable,
    }
    by_factor: dict[str, Any] = {}
    by_support: dict[str, Any] = {}
    for field in ("participant", "time", "event", "operator"):
        subset = [row for row in records if row["target_factor"] == field]
        by_factor[field] = {"scheduled_count": len(subset), "target_changed": _rate(sum(bool(row["target_changed"]) for row in subset), len(subset))}
    for support in ("unseen_pair", "seen_pair/unseen_triple"):
        subset = [row for row in records if row["support_group"] == support]
        by_support[support] = {"scheduled_count": len(subset), "target_changed": _rate(sum(bool(row["target_changed"]) for row in subset), len(subset))}
    result["by_factor"] = by_factor
    result["by_support_group"] = by_support
    return result


def aggregate_completed_runs(runs: Sequence[Mapping[str, Any]], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate exactly six completed runs under the frozen denominators."""

    if len(runs) != MAX_ATTEMPTS or any(run.get("status") != "complete" for run in runs):
        return {
            "schema": SUMMARY_SCHEMA,
            "status": "incomplete",
            "ranking": None,
            "completed_runs": sum(run.get("status") == "complete" for run in runs),
            "attempted_runs": len(runs),
            "reason": "failed_or_incomplete_run_set_produces_no_ranking",
        }
    expected_order = descriptor["training"]["run_order"]
    actual_order = [_run_label(str(run["arm"]), int(run["seed"])) for run in runs]
    if actual_order != expected_order:
        raise ValueError("run results differ from frozen seed-major order")

    thresholds = descriptor["selection"]["proposed_thresholds"]
    arms: dict[str, Any] = {}
    for arm in ("H1L1", "H1L1_ANCHOR"):
        selected = [run for run in runs if run["arm"] == arm]
        ordinary = [run["ordinary_confirmation"] for run in selected]
        controls = [row for run in selected for row in run["interventions"]["controls"]]
        swaps = [row for run in selected for row in run["interventions"]["source_swaps"]]
        control_aggregates = {
            control: aggregate_control_records(
                [row for row in controls if row["control_type"] == control],
                expected_scheduled_count=24,
            )
            for control in CONTROL_TYPES
        }
        arms[arm] = {
            "seeds": [run["seed"] for run in selected],
            "ordinary": {
                "generation_frame_exact_mean": _mean([_metric(report, "all", "generation_frame_exact") for report in ordinary]),
                "intermediate_frame_exact_mean": _mean([_metric(report, "all", "intermediate_frame_exact") for report in ordinary]),
                "exact_target_text_mean": _mean([_metric(report, "all", "exact_target_text") for report in ordinary]),
                "unseen_pair_generation_frame_exact_mean": _mean([_metric(report, "unseen_pair", "generation_frame_exact") for report in ordinary]),
                "seen_pair_generation_frame_exact_mean": _mean([_metric(report, "seen_pair", "generation_frame_exact") for report in ordinary]),
            },
            "controls": control_aggregates,
            "source_swaps": _aggregate_swaps(swaps),
        }
        if arm == "H1L1":
            diagnostics = [run["constant_source_diagnostic"] for run in selected]
            arms[arm]["constant_source_diagnostic"] = {
                "generation_frame_exact_mean": _mean([_metric(report, "all", "generation_frame_exact") for report in diagnostics]),
                "claim_scope": "out_of_distribution_inference_diagnostic_not_learned_path_evidence",
            }

    ordinary = arms["H1L1"]["ordinary"]
    anchor = arms["H1L1_ANCHOR"]["ordinary"]
    onehot_left = arms["H1L1"]["controls"]["alternate_class_one_hot"]["nontrivial_joint_success_scheduled"]["rate"]
    onehot_anchor = arms["H1L1_ANCHOR"]["controls"]["alternate_class_one_hot"]["nontrivial_joint_success_scheduled"]["rate"]
    assert isinstance(onehot_left, float) and isinstance(onehot_anchor, float)
    preserved = (
        anchor["generation_frame_exact_mean"] >= ordinary["generation_frame_exact_mean"] - thresholds["preserved_max_drop"]
        and anchor["intermediate_frame_exact_mean"] >= ordinary["intermediate_frame_exact_mean"] - thresholds["preserved_max_drop"]
    )
    improved = onehot_anchor - onehot_left >= thresholds["improved_min_gain"]
    constant_close = (
        abs(arms["H1L1"]["constant_source_diagnostic"]["generation_frame_exact_mean"] - ordinary["generation_frame_exact_mean"])
        <= thresholds["constant_source_close_max_gap"]
    )
    compositional_failure = (
        anchor["unseen_pair_generation_frame_exact_mean"] <= thresholds["unseen_pair_near_zero_max"]
        and anchor["seen_pair_generation_frame_exact_mean"] - anchor["unseen_pair_generation_frame_exact_mean"] >= thresholds["seen_vs_unseen_material_gap_min"]
    )
    frame_without_text = anchor["generation_frame_exact_mean"] > anchor["exact_target_text_mean"]
    heads_strong = (
        ordinary["intermediate_frame_exact_mean"] >= thresholds["strong_head_min"]
        and anchor["intermediate_frame_exact_mean"] >= thresholds["strong_head_min"]
    )
    same_class_material_effect = any(
        arms[arm]["controls"]["same_class_soft_shape"]["target_changed"]["rate"]
        >= thresholds["same_class_material_effect_min"]
        or 1.0 - arms[arm]["controls"]["same_class_soft_shape"]["non_target_preserved"]["rate"]
        >= thresholds["same_class_material_effect_min"]
        for arm in ("H1L1", "H1L1_ANCHOR")
    )
    donor_rates = {
        arm: arms[arm]["controls"]["alternate_class_donor_soft"]["nontrivial_joint_success_scheduled"]["rate"]
        for arm in ("H1L1", "H1L1_ANCHOR")
    }
    onehot_rates = {"H1L1": onehot_left, "H1L1_ANCHOR": onehot_anchor}
    onehot_only = any(onehot_rates[arm] > 0.0 and donor_rates[arm] == 0.0 for arm in onehot_rates)
    donor_only = any(donor_rates[arm] > 0.0 and onehot_rates[arm] == 0.0 for arm in onehot_rates)
    if not heads_strong:
        claim = "decoder_path_inconclusive_due_to_weak_heads"
    elif preserved and improved:
        claim = "source_dependent_h0_bypass_supported"
    else:
        claim = "source_dependent_h0_bypass_not_supported"
    decisions = {
        "h0_bypass_supported": preserved and improved,
        "preservation_rule_met": preserved,
        "improvement_rule_met": improved,
        "both_arms_strong_heads": heads_strong,
        "decoder_path_inconclusive_due_to_weak_heads": not heads_strong,
        "constant_source_close_withhold_source_dependence": constant_close,
        "fixture_specific_compositional_failure": compositional_failure,
        "frame_recovery_without_exact_text": frame_without_text,
        "same_class_probability_shape_material_effect": same_class_material_effect,
        "one_hot_only_follow_through": onehot_only,
        "donor_soft_only_follow_through": donor_only,
        "claim": claim,
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "ranking": None,
        "descriptor_sha256": PROTOCOL_SHA256,
        "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
        "run_count": len(runs),
        "optimizer_updates": sum(int(run["training"]["optimizer_updates"]) for run in runs),
        "training_wall_seconds": sum(float(run["training"]["wall_seconds"]) for run in runs),
        "budget": {
            "attempted_runs": len(runs),
            "max_attempts": MAX_ATTEMPTS,
            "optimizer_updates": sum(int(run["training"]["optimizer_updates"]) for run in runs),
            "max_optimizer_updates": MAX_UPDATES,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "wall_budget_breach_detected": False,
        },
        "arms": arms,
        "decisions": decisions,
        "scope": "authored structural-factor confirmation; not a learned modern-semantic or linguistic-quality claim",
    }


def _training_payload(training: Any) -> dict[str, Any]:
    return {
        "arm": training.arm,
        "seed": training.seed,
        "initial_state_sha256": training.initial_state_sha256,
        "final_state_sha256": training.final_state_sha256,
        "schedule_sha256": training.schedule_sha256,
        "losses": training.losses,
        "optimizer_updates": training.optimizer_updates,
        "wall_seconds": training.wall_seconds,
        "trainable_parameters": training.model.parameter_count(),
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Benchmark v3 h0-bypass confirmation",
        "",
        f"Descriptor: `{summary.get('descriptor_sha256', 'unavailable')}`",
        f"Fixture: `{summary.get('fixture_content_digest_sha256', 'unavailable')}`",
        "",
        f"Status: `{summary['status']}`",
        "",
        "This is a bounded authored structural-factor experiment. It is not evidence that a model learned modern lexical semantics, sememes, or concepts.",
        "",
    ]
    if summary["status"] != "complete":
        lines.append(f"No ranking or conclusion was produced: {summary['reason']}.")
        return "\n".join(lines) + "\n"
    lines.extend([
        "| Arm | Generation frame exact | Head frame exact | Alternate one-hot nontrivial joint / 24 |",
        "|---|---:|---:|---:|",
    ])
    for arm in ("H1L1", "H1L1_ANCHOR"):
        data = summary["arms"][arm]
        lines.append(
            f"| {arm} | {data['ordinary']['generation_frame_exact_mean']:.4f} | "
            f"{data['ordinary']['intermediate_frame_exact_mean']:.4f} | "
            f"{data['controls']['alternate_class_one_hot']['nontrivial_joint_success_scheduled']['rate']:.4f} |"
        )
    lines.extend(["", f"Decision: `{summary['decisions']['claim']}`.", ""])
    decisions = summary["decisions"]
    lines.extend([
        f"- Preservation rule: `{decisions['preservation_rule_met']}`",
        f"- Improvement rule: `{decisions['improvement_rule_met']}`",
        f"- Strong-head prerequisite: `{decisions['both_arms_strong_heads']}`",
        f"- Constant-source closeness requires withholding a source-dependent interpretation: `{decisions['constant_source_close_withhold_source_dependence']}`",
        f"- Fixture-specific compositional failure condition: `{decisions['fixture_specific_compositional_failure']}`",
        f"- Frame recovery without exact target text: `{decisions['frame_recovery_without_exact_text']}`",
        "",
        f"The six runs used {summary['optimizer_updates']} optimizer updates. Training took {summary['training_wall_seconds']:.3f} CPU seconds with one Torch thread. No GPU, network data, or paid compute was used.",
        "",
        "The low intermediate head-frame rates make the decoder-path interpretation inconclusive. The outputs are results on an authored structural-factor fixture and are not learned modern-semantic, sememe, concept, or linguistic-quality evidence.",
    ])
    return "\n".join(lines)


def run_frozen_experiment(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Execute the one authorized six-run set, stopping on any failure."""

    descriptor = load_protocol()
    preflight_protocol(descriptor)
    destination = Path(output_dir)
    if destination.exists() and any(destination.glob("run-*.json")):
        raise FileExistsError("existing run artifacts prevent an unregistered retry")
    started = time.monotonic()
    attempts: list[dict[str, Any]] = []
    updates = 0
    for item in descriptor["training"]["run_order"]:
        arm, seed = _parse_run_label(item)
        if len(attempts) >= MAX_ATTEMPTS or updates + 600 > MAX_UPDATES or time.monotonic() - started >= MAX_WALL_SECONDS:
            attempts.append({"status": "failed", "arm": arm, "seed": seed, "reason": "budget_preflight_failed"})
            break
        try:
            preflight_protocol(descriptor, arm=arm, seed=seed)
            training = train_h0_arm(arm, seed)
            updates += training.optimizer_updates
            ordinary = evaluate_h0_confirmation(training.model)
            constant = evaluate_h0_constant_source(training.model) if arm == "H1L1" else None
            interventions = evaluate_h0_interventions(training.model, descriptor)
            if state_sha256(training.model.state_dict()) != training.final_state_sha256:
                raise ValueError("evaluation mutated the trained model state")
            record = {
                "schema": RUN_SCHEMA,
                "status": "complete",
                "descriptor_sha256": PROTOCOL_SHA256,
                "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
                "arm": arm,
                "seed": seed,
                "attempt_index": len(attempts) + 1,
                "training": _training_payload(training),
                "ordinary_confirmation": ordinary,
                "constant_source_diagnostic": constant,
                "interventions": interventions,
            }
        except Exception as exc:
            record = {
                "schema": RUN_SCHEMA,
                "status": "failed",
                "descriptor_sha256": PROTOCOL_SHA256,
                "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
                "arm": arm,
                "seed": seed,
                "attempt_index": len(attempts) + 1,
                "error_type": type(exc).__name__,
                "redacted_message": str(exc).replace("\n", " ")[:500],
            }
        attempts.append(record)
        _write_json(destination / f"run-{len(attempts):02d}-{arm.lower()}-seed{seed}.json", record)
        if record["status"] != "complete":
            break
        if time.monotonic() - started >= MAX_WALL_SECONDS:
            break
    summary = aggregate_completed_runs(attempts, descriptor)
    summary["total_wall_seconds"] = time.monotonic() - started
    summary["attempted_runs"] = len(attempts)
    _write_json(destination / "summary.json", summary)
    (destination / "README.md").write_text(_markdown(summary), encoding="utf-8")
    _write_artifact_manifest(destination)
    return {"runs": attempts, "summary": summary, "output_dir": str(destination)}


def finalize_existing_runs(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Aggregate the one existing six-run set without training or inference."""

    destination = Path(output_dir)
    paths = sorted(destination.glob("run-*.json"))
    if len(paths) != MAX_ATTEMPTS:
        raise ValueError("finalization requires exactly six existing run artifacts")
    runs: list[dict[str, Any]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"run artifact is unreadable: {path.name}") from exc
        if (
            not isinstance(record, Mapping)
            or record.get("schema") != RUN_SCHEMA
            or record.get("descriptor_sha256") != PROTOCOL_SHA256
            or record.get("fixture_content_digest_sha256") != FIXTURE_CONTENT_SHA256
        ):
            raise ValueError(f"run artifact binding mismatch: {path.name}")
        runs.append(dict(record))
    descriptor = load_protocol()
    preflight_protocol(descriptor)
    summary = aggregate_completed_runs(runs, descriptor)
    summary["attempted_runs"] = len(runs)
    summary["total_wall_seconds"] = None
    summary["total_wall_seconds_reason"] = "aggregation_recovered_from_completed_raw_runs_after_summary_path_error"
    _write_json(destination / "summary.json", summary)
    (destination / "README.md").write_text(_markdown(summary), encoding="utf-8")
    _write_artifact_manifest(destination)
    return {"runs": runs, "summary": summary, "output_dir": str(destination)}


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "RUN_SCHEMA",
    "SUMMARY_SCHEMA",
    "aggregate_completed_runs",
    "finalize_existing_runs",
    "run_frozen_experiment",
]
