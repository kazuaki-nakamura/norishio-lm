"""Contract tests for the preregistered Issue #36 Phase 1 tournament."""
from __future__ import annotations

import copy

import pytest

from norishio_lm.benchmark_v3_tournament import (
    ARM_COUNTS,
    BENCHMARK_DIGEST,
    SEEDS,
    calculated_parameter_count,
    calculated_parameter_counts,
    config_sha256,
    load_tournament,
    required_run_keys,
    validate_terminal_runs,
    validate_tournament,
)


def test_frozen_budget_and_six_arm_run_grid() -> None:
    config = load_tournament()
    report = validate_tournament(config)
    assert report["arms"] == 6 and report["runs"] == 18
    assert report["maximum_relative_deviation"] < .03
    assert calculated_parameter_counts() == ARM_COUNTS
    assert all(ARM_COUNTS[arm] == calculated_parameter_count() for arm in ("H0L0", "H0L1", "H1L0", "H1L1", "NO_INPUT"))
    assert ARM_COUNTS["D_AUX"] < calculated_parameter_count()
    assert len(config_sha256(config)) == 64
    assert len(required_run_keys(config)) == 18
    assert tuple(config["randomness"]["run_seeds"]) == SEEDS
    assert config["input_boundary"]["no_input_source"]["token_ids"] == [1, 3]


def test_main_four_differ_only_by_activation_and_gate() -> None:
    arms = {arm["id"]: arm for arm in load_tournament()["arms"]}
    common = {key: value for key, value in arms["H0L0"].items()
              if key not in {"id", "name", "activation", "gate_mode"}}
    for arm_id in ("H0L1", "H1L0", "H1L1"):
        assert {key: value for key, value in arms[arm_id].items()
                if key not in {"id", "name", "activation", "gate_mode"}} == common
    assert {arms[arm_id]["activation"] for arm_id in ("H0L0", "H0L1")} == {"linear"}
    assert {arms[arm_id]["activation"] for arm_id in ("H1L0", "H1L1")} == {"tanh"}
    assert {arms[arm_id]["gate_mode"] for arm_id in ("H0L0", "H1L0")} == {"global"}
    assert {arms[arm_id]["gate_mode"] for arm_id in ("H0L1", "H1L1")} == {"prefix-local"}
    assert arms["NO_INPUT"]["source_input"] == "constant_bos_sep_source_tensor"


@pytest.mark.parametrize("mutation", ["seed", "final", "feature", "token_ids", "parameters", "training", "sampling"])
def test_frozen_fields_reject_changes(mutation: str) -> None:
    config = load_tournament()
    if mutation == "seed":
        config["randomness"]["run_seeds"] = [7]
    elif mutation == "final":
        config["benchmark"]["final_flag"] = "--test"
    elif mutation == "feature":
        config["input_boundary"]["forbidden_features"].remove("gold_frame")
    elif mutation == "token_ids":
        config["input_boundary"]["bos_id"], config["input_boundary"]["eos_id"] = 2, 1
    elif mutation == "parameters":
        config["arms"][0]["trainable_parameters"] += 1
    elif mutation == "sampling":
        config["sampling_schedule"]["batch_size"] = 8
    else:
        config["training"]["steps"] += 1
    with pytest.raises(ValueError):
        validate_tournament(config)


def _complete_record(config: dict, arm: str, seed: int) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "status": "complete",
        "tournament_config_sha256": config_sha256(config),
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "trainable_parameters": ARM_COUNTS[arm],
        "initial_state_sha256": "a" * 64,
        "final_state_sha256": "b" * 64,
        "schedule_sha256": "c" * 64,
        "checkpoint_file_sha256": "d" * 64,
    }


def test_checkpoint_gate_requires_all_exact_records_and_keeps_failure_visible() -> None:
    config = load_tournament()
    records = [_complete_record(config, arm, seed) for arm, seed in required_run_keys(config)]
    records[-1] = {
        "arm": "NO_INPUT", "seed": 29, "status": "failed", "phase": "training",
        "error_type": "RuntimeError", "redacted_message": "finite check failed",
        "tournament_config_sha256": config_sha256(config),
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "trainable_parameters": ARM_COUNTS["NO_INPUT"],
    }
    report = validate_terminal_runs(records, config)
    assert report["terminal"] == 18 and report["complete"] == 17 and report["failed"] == 1
    assert report["ready_for_single_final_invocation"] is False
    assert ["NO_INPUT", 29] not in report["final_evaluable"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_terminal_runs(records[:-1], config)
    changed = copy.deepcopy(records)
    changed[0]["trainable_parameters"] += 1
    with pytest.raises(ValueError, match="parameter"):
        validate_terminal_runs(changed, config)


def test_all_failures_are_terminal_and_leave_no_final_evaluable_checkpoint() -> None:
    config = load_tournament()
    records = [{
        "arm": arm, "seed": seed, "status": "failed", "phase": "training",
        "error_type": "RuntimeError", "redacted_message": "recorded failure",
        "tournament_config_sha256": config_sha256(config),
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "trainable_parameters": ARM_COUNTS[arm],
    } for arm, seed in required_run_keys(config)]
    report = validate_terminal_runs(records, config)
    assert report["ready_for_single_final_invocation"] is False
    assert report["complete"] == 0 and report["failed"] == 18
    assert report["final_evaluable"] == []


def test_all_complete_runs_are_required_for_final_readiness() -> None:
    config = load_tournament()
    records = [_complete_record(config, arm, seed) for arm, seed in required_run_keys(config)]
    report = validate_terminal_runs(records, config)
    assert report["ready_for_single_final_invocation"] is True
    assert report["complete"] == 18 and report["failed"] == 0


def test_functional_model_parameter_counts_match_contract() -> None:
    torch = pytest.importorskip("torch")
    del torch
    from norishio_lm.benchmark_v3_model import build_model

    config = load_tournament()
    expected = {arm["id"]: arm["trainable_parameters"] for arm in config["arms"]}
    actual = {arm: build_model(arm).trainable_parameter_count() for arm in expected}
    assert actual == expected
    assert (max(actual.values()) - min(actual.values())) / min(actual.values()) < .03
