from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v2_checkpoint import (
    expected_schedule_sha256, save_tournament_checkpoint, state_sha256,
)
from norishio_lm.benchmark_v2_final import checkpoint_path, evaluate_final_once, main
from norishio_lm.benchmark_v2_model import build_model
from norishio_lm.benchmark_v2_protocol import (
    complete_record, failed_record, write_terminal_record,
)
from norishio_lm.benchmark_v2_tournament import ARM_COUNTS, SEEDS


def _write_all_failed(root) -> None:
    for arm in ARM_COUNTS:
        for seed in SEEDS:
            write_terminal_record(root, failed_record(
                arm=arm, seed=seed, phase="training", error_type="FixtureFailure",
                redacted_message="fixture terminal failure",
            ))


def test_final_requires_explicit_flag_before_record_access(tmp_path):
    with pytest.raises(PermissionError, match="explicit"):
        evaluate_final_once(tmp_path, evaluate_final=False, evaluator=lambda *_: {})
    assert not (tmp_path / "final-invocation").exists()
    with pytest.raises(SystemExit):
        main(["--root", str(tmp_path)])


def test_all_failed_set_consumes_one_invocation_and_preserves_failures(tmp_path):
    _write_all_failed(tmp_path)
    called = False

    def evaluator(*_):
        nonlocal called
        called = True

    result = evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    assert result["status"] == "complete"
    assert result["rows"] == 384
    assert result["evaluations"] == []
    assert len(result["failed_runs"]) == 18
    assert called is False
    marker = json.loads((tmp_path / "final-invocation" / "marker.json").read_text("utf-8"))
    assert marker["complete"] == 0 and marker["failed"] == 18
    with pytest.raises(FileExistsError):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)


def test_incomplete_terminal_set_does_not_consume_invocation(tmp_path):
    write_terminal_record(tmp_path, failed_record(
        arm="A_G0", seed=7, phase="training", error_type="FixtureFailure",
        redacted_message="fixture terminal failure",
    ))
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=lambda *_: {})
    assert not (tmp_path / "final-invocation").exists()


def test_complete_checkpoint_is_authenticated_before_callback(tmp_path):
    model = build_model("A_G0", seed=7)
    initial = state_sha256(model.state_dict())
    model.encoder.projection.weight.data.add_(0.01)
    path = checkpoint_path(tmp_path, "A_G0", 7)
    save_tournament_checkpoint(
        path, model, arm="A_G0", seed=7,
        schedule_sha256=expected_schedule_sha256(7), initial_state_sha256=initial,
    )
    write_terminal_record(tmp_path, complete_record(path, arm="A_G0", seed=7))
    for arm in ARM_COUNTS:
        for seed in SEEDS:
            if (arm, seed) == ("A_G0", 7):
                continue
            write_terminal_record(tmp_path, failed_record(
                arm=arm, seed=seed, phase="training", error_type="FixtureFailure",
                redacted_message="fixture terminal failure",
            ))

    def evaluator(record, payload, rows):
        assert record["arm"] == payload["model"].arm == "A_G0"
        assert len(rows) == 384
        return {"authenticated": True}

    result = evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    assert result["evaluations"] == [
        {"arm": "A_G0", "seed": 7, "result": {"authenticated": True}}
    ]
