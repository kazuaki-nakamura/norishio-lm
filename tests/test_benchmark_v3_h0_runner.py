from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_fsm import FrozenLocalPrefixFSM
from norishio_lm.benchmark_v3_h0_fixture import spec_data
from norishio_lm.benchmark_v3_h0_model import build_future_h0_model
from norishio_lm.benchmark_v3_h0_runner import (
    evaluate_h0_constant_source,
    h0_factor_vocabulary,
    load_h0_confirmation_rows,
    load_h0_train_rows,
    paired_initialization_preflight,
    prepare_h0_row,
)
from norishio_lm.benchmark_v3_runner import collate_rows, teacher_forced_gates, training_step


def test_new_fixture_rows_prepare_without_gold_in_source() -> None:
    train = load_h0_train_rows()
    confirmation = load_h0_confirmation_rows()
    prepared = prepare_h0_row(train[0])

    assert len(train) == 384
    assert len(confirmation) == 96
    assert prepared.factor_targets == h0_factor_vocabulary().encode(train[0]["targets"]["frame"])
    source_bytes = bytes(token - 4 for token in prepared.source_ids[1:-1]).decode("utf-8")
    assert train[0]["targets"]["text"] not in source_bytes


@pytest.mark.parametrize("seed", [7, 17, 29])
def test_paired_initialization_preflight(seed: int) -> None:
    report = paired_initialization_preflight(seed)
    assert report["bitwise_equal"] is True
    assert report["trainable_parameters"] == 32120


def test_one_cpu_training_step_uses_new_fixture_fsm() -> None:
    model = build_future_h0_model("H1L1_ANCHOR", seed=7)
    row = prepare_h0_row(load_h0_train_rows()[0])
    batch = collate_rows([row])
    fsm = FrozenLocalPrefixFSM(spec_data())
    gates = teacher_forced_gates(row, fsm).unsqueeze(0)
    assert gates.shape == (1, row.decoder_length, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss = training_step(model, optimizer, batch, fsm=fsm)
    assert loss > 0.0


def test_constant_source_diagnostic_does_not_create_an_anchor_arm_ablation() -> None:
    with pytest.raises(ValueError, match="frozen to the H1L1 arm"):
        evaluate_h0_constant_source(build_future_h0_model("H1L1_ANCHOR", seed=7))
