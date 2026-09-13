from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v2_checkpoint import expected_schedule_sha256, state_sha256
from norishio_lm.benchmark_v2_execute import (
    development_path, execute_one, write_development_summary,
)
from norishio_lm.benchmark_v2_model import build_model
from norishio_lm.benchmark_v2_protocol import failed_record, record_path
from norishio_lm.benchmark_v2_runner import TrainingResult
from norishio_lm.benchmark_v2_tournament import ARM_COUNTS, required_run_keys, load_tournament


def _untrained_result(arm: str, seed: int) -> TrainingResult:
    model = build_model(arm, seed=seed)
    digest = state_sha256(model.state_dict())
    return TrainingResult(arm, seed, model, digest, digest,
                          expected_schedule_sha256(seed), [2.0] * 600)


def test_execute_one_publishes_checkpoint_report_then_terminal(tmp_path):
    record = execute_one(
        tmp_path, "A_G0", 7, trainer=_untrained_result,
        evaluator=lambda _model: {"fixture": True},
    )
    assert record["status"] == "complete"
    report = json.loads(development_path(tmp_path, "A_G0", 7).read_text("utf-8"))
    assert report["loss"]["updates"] == 600
    assert report["metrics"] == {"fixture": True}
    assert json.loads(record_path(tmp_path, "A_G0", 7).read_text("utf-8")) == record
    with pytest.raises(FileExistsError):
        execute_one(tmp_path, "A_G0", 7, trainer=_untrained_result,
                    evaluator=lambda _model: {})


def test_execute_one_records_visible_failure(tmp_path):
    def fail(_arm, _seed):
        raise RuntimeError("D:/private/value")

    with pytest.raises(RuntimeError):
        execute_one(tmp_path, "A_G0", 7, trainer=fail)
    record = json.loads(record_path(tmp_path, "A_G0", 7).read_text("utf-8"))
    assert record["status"] == "failed"
    assert record["phase"] == "training"
    assert "private" not in record["redacted_message"]


SUMMARY_METRICS = {
    "free_generation_exact": 1.0,
    "generation_frame_exact": 1.0,
    "triple_exact": 1.0,
    "mean_balanced_atomic_accuracy": 1.0,
    "teacher_forced_byte_match": 1.0,
    "intervention_non_target_preservation": 1.0,
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
    artifact = development_path(tmp_path, "A_G0", 7)
    artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({"metrics": {
        "all": {
            "balanced_accuracy": {field: 1.0 for field in ("participant", "time", "event", "operator")},
            "free_generation_exact": {"accuracy": 1.0},
            "generation_frame_exact": {"accuracy": 1.0},
            "participant_time_event_triple_exact": {"accuracy": 1.0},
        },
        "teacher_forced_bytes": {"byte_match_rate": 1.0},
        "intervention_locality": {"by_factor": {
            field: {"preservation_rate": 1.0}
            for field in ("participant", "time", "event", "operator")
        }},
    }}), encoding="utf-8")
    records = _failed_records()
    records[0] = {"arm": "A_G0", "seed": 7, "status": "complete"}
    summary = json.loads(write_development_summary(tmp_path, records).read_text("utf-8"))
    means = summary["arms"]["A_G0"]["three_seed_mean"]
    assert set(means) == set(SUMMARY_METRICS)
    assert all(value is None for value in means.values())
