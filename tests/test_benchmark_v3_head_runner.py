from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_head_runner import (
    PARAMETER_COUNT, evaluate_head_outputs, paired_initialization_preflight,
    run_head_mode, train_head_mode,
)
from norishio_lm.benchmark_v3_model import build_model
from norishio_lm.benchmark_v3_runner import PreparedRow


def _rows(count: int = 16) -> list[PreparedRow]:
    result: list[PreparedRow] = []
    for index in range(count):
        token = 4 + (97 + index) % 26
        result.append(PreparedRow(
            source_ids=(1, token, 3), decoder_input_ids=(1, token), labels=(token, 2),
            factor_targets=(index % 6, (index + 1) % 6, index % 4, (index + 2) % 4),
            target_text=chr(97 + index % 26),
            target_frame={"participant": str(index % 6), "time": str((index + 1) % 6),
                          "event": str(index % 4), "operator": str((index + 2) % 4)},
        ))
    return result


def _schedule() -> list[list[int]]:
    return [list(range(16))]


def test_paired_initialization_is_bitwise_equal_and_parameter_matched() -> None:
    report = paired_initialization_preflight(7)
    assert report["bitwise_equal"] is True
    assert report["trainable_parameters"] == PARAMETER_COUNT == 32120


def test_objectives_have_distinct_gradient_paths() -> None:
    rows = _rows(); descriptor = {"fixture": "tiny-test", "schedule": "one-step"}
    joint = train_head_mode("JOINT", 7, rows, _schedule(), descriptor)
    factor_only = train_head_mode("FACTOR_ONLY", 7, rows, _schedule(), descriptor)
    assert joint.final_state_sha256 != joint.initial_state_sha256
    assert factor_only.final_state_sha256 != factor_only.initial_state_sha256
    fresh = build_model("H1L1", seed=7)
    assert any(not torch.equal(joint.model.state_dict()[key], fresh.state_dict()[key])
               for key in joint.model.state_dict() if key.startswith("decoder."))
    assert all(torch.equal(factor_only.model.state_dict()[key], fresh.state_dict()[key])
               for key in factor_only.model.state_dict() if key.startswith("decoder."))


def test_head_outputs_are_probability_simplexes_with_argmaxes() -> None:
    rows = _rows(); descriptor = {"fixture": "tiny-test"}
    trained = train_head_mode("FACTOR_ONLY", 7, rows, _schedule(), descriptor)
    result = evaluate_head_outputs(trained.model, rows, descriptor, split="confirmation",
                                   objective="FACTOR_ONLY", seed=7)
    assert result["row_count"] == len(rows)
    for row in result["head_records"]:
        for field, values in row["factor_probability_vectors"].items():
            assert all(value >= 0.0 for value in values)
            assert sum(values) == pytest.approx(1.0)
            assert row["factor_argmax"][field] == max(range(len(values)), key=values.__getitem__)


def test_combined_result_retains_train_and_confirmation_head_records() -> None:
    rows = _rows()
    result = run_head_mode("JOINT", 7, rows, rows[:4], _schedule(), {"fixture": "tiny-test"})
    assert result["objective"] == "JOINT"
    assert len(result["train_head_records"]) == 16
    assert len(result["confirmation_head_records"]) == 4
    assert result["optimizer_updates"] == 1
    assert result["training_wall_seconds"] >= 0.0
