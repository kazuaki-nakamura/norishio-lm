from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v2_checkpoint import expected_schedule_sha256, state_sha256
from norishio_lm.benchmark_v2_execute import development_path, execute_one
from norishio_lm.benchmark_v2_model import build_model
from norishio_lm.benchmark_v2_protocol import record_path
from norishio_lm.benchmark_v2_runner import TrainingResult


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
