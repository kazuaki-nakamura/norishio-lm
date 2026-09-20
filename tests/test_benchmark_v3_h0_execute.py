from __future__ import annotations

import copy
import json

from norishio_lm.benchmark_v3_h0_execute import (
    RUN_SCHEMA,
    WALL_TIME_UNAVAILABLE_REASON,
    _markdown,
    aggregate_completed_runs,
    finalize_existing_runs,
)
from norishio_lm.benchmark_v3_h0_protocol import (
    FIXTURE_CONTENT_SHA256,
    MAX_WALL_SECONDS,
    PROTOCOL_SHA256,
    load_protocol,
)
from norishio_lm.benchmark_v3_model import BOS_ID, FACTOR_ORDER, FACTOR_SIZES


def _metric(value: float, denominator: int = 96):
    return {"accuracy": value, "count": int(value * denominator), "denominator": denominator}


def _ordinary(frame: float, head: float, text: float, unseen: float, seen: float):
    return {
        "all": {
            "generation_frame_exact": _metric(frame),
            "intermediate_frame_exact": _metric(head),
            "exact_target_text": _metric(text),
        },
        "groups": {
            "unseen_pair": {"generation_frame_exact": _metric(unseen, 48)},
            "seen_pair": {"generation_frame_exact": _metric(seen, 48)},
        },
    }


def _maps():
    return {field: [1.0, *([0.0] * (FACTOR_SIZES[field] - 1))] for field in FACTOR_ORDER}


def _frame(participant: str):
    return {"participant": participant, "time": "t0", "event": "e0", "operator": "o0"}


def _control(control: str, *, success: bool):
    replacement = _maps()
    replacement["participant"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "control_type": control,
        "prefix_stratum": "bos_start_primary",
        "prefix_token_ids": [BOS_ID],
        "target_factor": "participant",
        "requested_value": "p1",
        "baseline_probability_map": _maps(),
        "replacement_probability_map": replacement,
        "expected_target_gate_schedule": [[True, False, False, False]],
        "expected_target_gate_mask": [True],
        "baseline_observed_target_gate_schedule": [[True, False, False, False]],
        "baseline_target_gate_mask": [True],
        "intervention_observed_target_gate_schedule": [[True, False, False, False]],
        "intervention_target_gate_mask": [True],
        "baseline_parsed_frame": _frame("p0"),
        "intervention_parsed_frame": _frame("p1" if success else "p0"),
        "missing_donor": False,
        "unchanged_shape": False,
        "intervention_unavailable": False,
        "intervention_unavailable_reason": None,
        "structural_unavailable_reason": None,
    }


def _swap():
    return {
        "baseline_parse_failed": False,
        "changed_parse_failed": False,
        "target_changed": True,
        "non_target_preserved": True,
        "exact_target_text": False,
        "target_change_correct": True,
        "target_factor": "participant",
        "support_group": "unseen_pair",
    }


def _run(arm: str, seed: int):
    success = arm == "H1L1_ANCHOR"
    controls = [
        _control(control, success=(success and control == "alternate_class_one_hot"))
        for control in ("same_class_soft_shape", "alternate_class_one_hot", "alternate_class_donor_soft")
        for _ in range(8)
    ]
    ordinary = _ordinary(0.50 if arm == "H1L1" else 0.48, 0.90, 0.40, 0.0, 0.20)
    return {
        "status": "complete",
        "arm": arm,
        "seed": seed,
        "training": {"optimizer_updates": 600, "wall_seconds": 1.0},
        "ordinary_confirmation": ordinary,
        "constant_source_diagnostic": _ordinary(0.48, 0.0, 0.0, 0.0, 0.0) if arm == "H1L1" else None,
        "interventions": {"controls": controls, "source_swaps": [_swap() for _ in range(8)]},
    }


def test_incomplete_set_has_no_ranking_or_claim() -> None:
    result = aggregate_completed_runs([], load_protocol())
    assert result["status"] == "incomplete"
    assert result["ranking"] is None


def test_complete_set_uses_frozen_denominators_and_decision_rules() -> None:
    runs = [
        _run("H1L1", 7), _run("H1L1_ANCHOR", 7),
        _run("H1L1", 17), _run("H1L1_ANCHOR", 17),
        _run("H1L1", 29), _run("H1L1_ANCHOR", 29),
    ]
    result = aggregate_completed_runs(copy.deepcopy(runs), load_protocol())

    assert result["status"] == "complete"
    assert result["optimizer_updates"] == 3600
    assert result["arms"]["H1L1_ANCHOR"]["controls"]["alternate_class_one_hot"]["scheduled_count"] == 24
    assert result["arms"]["H1L1_ANCHOR"]["source_swaps"]["scheduled_count"] == 24
    assert result["decisions"]["preservation_rule_met"] is True
    assert result["decisions"]["improvement_rule_met"] is True
    assert result["decisions"]["h0_bypass_supported"] is True
    assert result["decisions"]["fixture_specific_compositional_failure"] is True


def test_aggregation_without_executor_wall_time_keeps_budget_status_unavailable() -> None:
    runs = [
        _run("H1L1", 7), _run("H1L1_ANCHOR", 7),
        _run("H1L1", 17), _run("H1L1_ANCHOR", 17),
        _run("H1L1", 29), _run("H1L1_ANCHOR", 29),
    ]

    result = aggregate_completed_runs(copy.deepcopy(runs), load_protocol())

    assert result["budget"]["wall_budget_breach_detected"] is None
    assert result["budget"]["wall_budget_breach_reason"] == WALL_TIME_UNAVAILABLE_REASON


def test_new_executor_measurement_drives_wall_budget_boolean() -> None:
    runs = [
        _run("H1L1", 7), _run("H1L1_ANCHOR", 7),
        _run("H1L1", 17), _run("H1L1_ANCHOR", 17),
        _run("H1L1", 29), _run("H1L1_ANCHOR", 29),
    ]

    within_budget = aggregate_completed_runs(
        copy.deepcopy(runs),
        load_protocol(),
        total_wall_seconds=MAX_WALL_SECONDS - 0.001,
    )
    over_budget = aggregate_completed_runs(
        copy.deepcopy(runs),
        load_protocol(),
        total_wall_seconds=MAX_WALL_SECONDS,
    )

    assert within_budget["budget"]["wall_budget_breach_detected"] is False
    assert within_budget["budget"]["wall_budget_breach_reason"] is None
    assert over_budget["budget"]["wall_budget_breach_detected"] is True
    assert over_budget["budget"]["wall_budget_breach_reason"] is None


def test_finalize_existing_runs_keeps_wall_budget_status_unavailable(tmp_path) -> None:
    runs = [
        _run("H1L1", 7), _run("H1L1_ANCHOR", 7),
        _run("H1L1", 17), _run("H1L1_ANCHOR", 17),
        _run("H1L1", 29), _run("H1L1_ANCHOR", 29),
    ]
    for index, run in enumerate(runs, start=1):
        record = {
            "schema": RUN_SCHEMA,
            "descriptor_sha256": PROTOCOL_SHA256,
            "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256,
            **run,
        }
        path = tmp_path / f"run-{index:02d}-{run['arm'].lower()}-seed{run['seed']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")

    result = finalize_existing_runs(tmp_path)

    assert result["summary"]["total_wall_seconds"] is None
    assert result["summary"]["total_wall_seconds_reason"] == WALL_TIME_UNAVAILABLE_REASON
    assert result["summary"]["budget"]["wall_budget_breach_detected"] is None
    assert result["summary"]["budget"]["wall_budget_breach_reason"] == WALL_TIME_UNAVAILABLE_REASON
    readme = _markdown(result["summary"])
    assert "total-wall budget compliance is unavailable" in readme
    assert "No training or inference was rerun" in readme
    assert "`head-factor-audit.json`" in readme
    assert "does not change the frozen decision" in readme
