import copy

import pytest

from norishio_lm.benchmark_v2_tournament import (
    ARM_COUNTS, BENCHMARK_DIGEST, calculated_parameter_counts, config_sha256, load_tournament,
    required_run_keys, validate_terminal_runs, validate_tournament,
)


def test_frozen_tournament_budget_and_run_grid():
    config = load_tournament()
    report = validate_tournament(config)
    assert report["arms"] == 6 and report["runs"] == 18
    assert report["maximum_relative_deviation"] < .03
    assert calculated_parameter_counts() == ARM_COUNTS
    assert len(config_sha256(config)) == 64
    assert len(required_run_keys(config)) == 18


@pytest.mark.parametrize("mutation", ["seed", "final", "feature", "parameters", "training"])
def test_frozen_fields_reject_changes(mutation):
    config = load_tournament()
    if mutation == "seed":
        config["randomness"]["run_seeds"] = [7]
    elif mutation == "final":
        config["benchmark"]["final_flag"] = "--test"
    elif mutation == "feature":
        config["input_boundary"]["forbidden_features"].remove("gold_frame")
    elif mutation == "parameters":
        config["arms"][0]["trainable_parameters"] += 1
    else:
        config["training"]["steps"] += 1
    with pytest.raises(ValueError):
        validate_tournament(config)


def complete_record(config, arm, seed):
    return {"arm": arm, "seed": seed, "status": "complete",
            "tournament_config_sha256": config_sha256(config),
            "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
            "trainable_parameters": ARM_COUNTS[arm],
            "initial_state_sha256": "a" * 64, "final_state_sha256": "b" * 64,
            "schedule_sha256": "c" * 64, "checkpoint_file_sha256": "d" * 64}


def test_checkpoint_gate_requires_all_exact_records_and_keeps_failure_visible():
    config = load_tournament()
    records = [complete_record(config, arm, seed) for arm, seed in required_run_keys(config)]
    records[-1] = {"arm": "E", "seed": 29, "status": "failed", "phase": "training",
                   "error_type": "RuntimeError", "redacted_message": "finite check failed",
                   "tournament_config_sha256": config_sha256(config),
                   "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
                   "trainable_parameters": ARM_COUNTS["E"]}
    report = validate_terminal_runs(records, config)
    assert report["terminal"] == 18 and report["complete"] == 17 and report["failed"] == 1
    assert ["E", 29] not in report["final_evaluable"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_terminal_runs(records[:-1], config)
    changed = copy.deepcopy(records)
    changed[0]["trainable_parameters"] += 1
    with pytest.raises(ValueError, match="parameter"):
        validate_terminal_runs(changed, config)


def test_all_failures_are_terminal_and_leave_no_final_evaluable_checkpoint():
    config = load_tournament()
    records = [{"arm": arm, "seed": seed, "status": "failed", "phase": "training",
                "error_type": "RuntimeError", "redacted_message": "recorded failure",
                "tournament_config_sha256": config_sha256(config),
                "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
                "trainable_parameters": ARM_COUNTS[arm]}
               for arm, seed in required_run_keys(config)]
    report = validate_terminal_runs(records, config)
    assert report["ready_for_single_final_invocation"] is True
    assert report["complete"] == 0 and report["failed"] == 18
    assert report["final_evaluable"] == []
