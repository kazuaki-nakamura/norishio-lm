from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_checkpoint import expected_schedule_sha256, state_sha256
from norishio_lm.benchmark_v3_execute import (
    development_path, execute_one, write_development_summary,
)
from norishio_lm.benchmark_v3_model import build_model
from norishio_lm.benchmark_v3_protocol import failed_record, record_path
from norishio_lm.benchmark_v3_runner import TrainingResult
from norishio_lm.benchmark_v3_tournament import ARM_COUNTS, required_run_keys, load_tournament


def _untrained_result(arm: str, seed: int) -> TrainingResult:
    model = build_model(arm, seed=seed)
    digest = state_sha256(model.state_dict())
    return TrainingResult(arm, seed, model, digest, digest,
                          expected_schedule_sha256(seed), [2.0] * 600)


def test_execute_one_publishes_checkpoint_report_then_terminal(tmp_path):
    record = execute_one(
        tmp_path, "H0L0", 7, trainer=_untrained_result,
        evaluator=lambda _model: {"fixture": True},
    )
    assert record["status"] == "complete"
    report = json.loads(development_path(tmp_path, "H0L0", 7).read_text("utf-8"))
    assert report["loss"]["updates"] == 600
    assert report["metrics"] == {"fixture": True}
    assert json.loads(record_path(tmp_path, "H0L0", 7).read_text("utf-8")) == record
    with pytest.raises(FileExistsError):
        execute_one(tmp_path, "H0L0", 7, trainer=_untrained_result,
                    evaluator=lambda _model: {})


def test_execute_one_records_visible_failure(tmp_path):
    def fail(_arm, _seed):
        raise RuntimeError("D:/private/value")

    with pytest.raises(RuntimeError):
        execute_one(tmp_path, "H0L0", 7, trainer=fail)
    record = json.loads(record_path(tmp_path, "H0L0", 7).read_text("utf-8"))
    assert record["status"] == "failed"
    assert record["phase"] == "training"
    assert "private" not in record["redacted_message"]


SUMMARY_METRICS = {
    "free_generation_exact": 1.0,
    "triple_exact": 1.0,
    "pair_exact": 1.0,
    "mean_balanced_atomic_accuracy": 1.0,
    "exact_target_text": 1.0,
    "teacher_forced_byte_match": 1.0,
    "source_swap_non_target_preservation": 1.0,
    "intermediate_non_target_preservation": 1.0,
    "intermediate_target_change": 1.0,
}


def _failed_records():
    config = load_tournament()
    return [failed_record(
        arm=arm, seed=seed, phase="training", error_type="FixtureFailure",
        redacted_message="fixture failure",
    ) for arm, seed in required_run_keys(config)]


def test_development_summary_keeps_fixed_null_metrics_for_all_failed_arms(tmp_path):
    summary_path = write_development_summary(tmp_path, _failed_records())
    summary = json.loads(summary_path.read_text("utf-8"))
    expected = set(SUMMARY_METRICS)
    assert set(summary["arms"]) == set(ARM_COUNTS)
    for arm in ARM_COUNTS:
        means = summary["arms"][arm]["three_seed_mean"]
        assert set(means) == expected
        assert all(value is None for value in means.values())


def test_development_summary_keeps_fixed_null_metrics_for_partial_arm_failure(tmp_path):
    artifact = development_path(tmp_path, "H0L0", 7)
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"metrics": {
        "all": {
            "atomic_balanced_accuracy": {field: 1.0 for field in ("participant", "time", "event", "operator")},
            "free_generation_exact": {"accuracy": 1.0},
            "triple_exact": {"accuracy": 1.0},
            "pair_exact": {"accuracy": 1.0},
            "exact_target_text": {"accuracy": 1.0},
        },
        "teacher_forced": {"byte_match_rate": 1.0},
        "source_side_swap": {"by_factor": {
            field: {"non_target_preserved": {"accuracy": 1.0}}
            for field in ("participant", "time", "event", "operator")
        }},
        "intermediate_probability_intervention": {"by_factor": {
            field: {"non_target_preserved": {"accuracy": 1.0},
                    "target_changed": {"accuracy": 1.0}}
            for field in ("participant", "time", "event", "operator")
        }},
    }}), encoding="utf-8")
    records = _failed_records()
    records[0] = {"arm": "H0L0", "seed": 7, "status": "complete"}
    summary = json.loads(write_development_summary(tmp_path, records).read_text("utf-8"))
    means = summary["arms"]["H0L0"]["three_seed_mean"]
    assert set(means) == set(SUMMARY_METRICS)
    assert all(value is None for value in means.values())


def test_factorial_summary_reports_preregistered_effects_by_seed_and_mean(tmp_path):
    records = _failed_records()
    main = ("H0L0", "H0L1", "H1L0", "H1L1")
    values = {"H0L0": 0.1, "H0L1": 0.2, "H1L0": 0.4, "H1L1": 0.8}
    for index, record in enumerate(records):
        if record["arm"] not in main:
            continue
        metric = values[record["arm"]]
        path = development_path(tmp_path, record["arm"], record["seed"])
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": {
            "all": {
                "atomic_balanced_accuracy": {field: metric for field in ("participant", "time", "event", "operator")},
                "free_generation_exact": {"accuracy": metric},
                "triple_exact": {"accuracy": metric}, "pair_exact": {"accuracy": metric},
                "exact_target_text": {"accuracy": metric},
            },
            "teacher_forced": {"byte_match_rate": metric},
            "source_side_swap": {"by_factor": {field: {"non_target_preserved": {"accuracy": metric}} for field in ("participant", "time", "event", "operator")}},
            "intermediate_probability_intervention": {"by_factor": {field: {"non_target_preserved": {"accuracy": metric}, "target_changed": {"accuracy": metric}} for field in ("participant", "time", "event", "operator")}},
        }}
        path.write_text(json.dumps(payload), encoding="utf-8")
        records[index] = {"arm": record["arm"], "seed": record["seed"], "status": "complete"}
    summary = json.loads(write_development_summary(tmp_path, records).read_text("utf-8"))
    effect = summary["factorial_effects"]["by_seed"]["7"]["free_generation_exact"]
    assert effect == pytest.approx({
        "activation_main_effect": 0.45,
        "locality_main_effect": 0.25,
        "interaction": 0.3,
    })
