from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v2_checkpoint import (
    checkpoint_file_sha256,
    expected_schedule_sha256,
    load_tournament_checkpoint,
    save_tournament_checkpoint,
    state_sha256,
)
from norishio_lm.benchmark_v2_model import FROZEN_DERANGEMENTS, build_model


class SizedModel(torch.nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(count, dtype=torch.float32))


def _sha(char: str) -> str:
    return char * 64


def test_checkpoint_roundtrip_and_exclusive_publish(tmp_path):
    model = build_model("A_G0", seed=7)
    initial = state_sha256(model.state_dict())
    model.encoder.projection.weight.data.add_(1)
    path = tmp_path / "run.pt"
    record = save_tournament_checkpoint(
        path,
        model,
        arm="A_G0",
        seed=7,
        schedule_sha256=expected_schedule_sha256(7),
        initial_state_sha256=initial,
    )
    assert record["checkpoint_file_sha256"] == checkpoint_file_sha256(path)
    loaded = load_tournament_checkpoint(path, arm="A_G0", seed=7)
    assert state_sha256(loaded["state_dict"]) == record["final_state_sha256"]
    with pytest.raises(FileExistsError):
        save_tournament_checkpoint(
            path, model, arm="A_G0", seed=7,
            schedule_sha256=expected_schedule_sha256(7),
            initial_state_sha256=initial,
        )


def test_checkpoint_rejects_wrong_count_and_nonfinite_state(tmp_path):
    with pytest.raises(TypeError, match="frozen BenchmarkV2Model"):
        save_tournament_checkpoint(
            tmp_path / "wrong.pt", SizedModel(2), arm="A_G0", seed=7,
            schedule_sha256=expected_schedule_sha256(7), initial_state_sha256=_sha("b"),
        )
    model = build_model("A_G0", seed=7)
    model.encoder.projection.weight.data[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        save_tournament_checkpoint(
            tmp_path / "nan.pt", model, arm="A_G0", seed=7,
            schedule_sha256=expected_schedule_sha256(7),
            initial_state_sha256=state_sha256(build_model("A_G0", seed=7).state_dict()),
        )


def test_checkpoint_rejects_tampered_state_and_binding(tmp_path):
    model = build_model("A_G0", seed=7)
    path = tmp_path / "run.pt"
    save_tournament_checkpoint(
        path, model, arm="A_G0", seed=7,
        schedule_sha256=expected_schedule_sha256(7),
        initial_state_sha256=state_sha256(model.state_dict()),
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["state_dict"]["encoder.projection.weight"][0, 0] += 1
    torch.save(raw, path)
    with pytest.raises(ValueError, match="state digest"):
        load_tournament_checkpoint(path, arm="A_G0", seed=7)
    with pytest.raises(ValueError, match="metadata"):
        load_tournament_checkpoint(path, arm="A_G1", seed=7)


def test_checkpoint_model_metadata_must_be_json_safe(tmp_path):
    model = build_model("A_G0", seed=7)
    with pytest.raises(ValueError, match="model metadata"):
        save_tournament_checkpoint(
            tmp_path / "bad.pt", model, arm="A_G0", seed=7,
            schedule_sha256=expected_schedule_sha256(7),
            initial_state_sha256=state_sha256(model.state_dict()),
            model_metadata={"bad": {1, 2}},
        )


def test_checkpoint_rejects_wrong_schedule_and_initial_state(tmp_path):
    model = build_model("E", seed=17, derangement_mappings=FROZEN_DERANGEMENTS)
    initial = state_sha256(model.state_dict())
    with pytest.raises(ValueError, match="schedule"):
        save_tournament_checkpoint(
            tmp_path / "schedule.pt", model, arm="E", seed=17,
            schedule_sha256=_sha("a"), initial_state_sha256=initial,
        )
    with pytest.raises(ValueError, match="initial state"):
        save_tournament_checkpoint(
            tmp_path / "initial.pt", model, arm="E", seed=17,
            schedule_sha256=expected_schedule_sha256(17), initial_state_sha256=_sha("b"),
        )
