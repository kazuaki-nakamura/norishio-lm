from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_fsm import FrozenLocalPrefixFSM
from norishio_lm.benchmark_v3_model import ARM_IDS, build_model
import norishio_lm.benchmark_v3_runner as runner
from norishio_lm.benchmark_v3_runner import (
    TRAIN_STEPS,
    collate_rows,
    evaluate_benchmark_v3,
    factor_prediction_metadata,
    factor_vocabulary,
    greedy_generate,
    greedy_generate_batch,
    greedy_generate_intervened,
    load_development_rows,
    load_train_rows,
    prepare_row,
    teacher_forced_gates,
    training_step,
)
from norishio_lm.benchmark_v3_tournament import BENCHMARK_DIGEST


def test_factor_vocabulary_is_derived_from_committed_spec() -> None:
    vocabulary = factor_vocabulary()
    assert {field: len(values) for field, values in vocabulary.values.items()} == {
        "participant": 6, "time": 6, "event": 4, "operator": 4,
    }
    assert vocabulary.encode({
        "participant": "FRIEND", "time": "TODAY", "event": "MEET", "operator": "ASSERT",
    }) == (0, 0, 0, 0)
    metadata = factor_prediction_metadata("D_AUX")
    assert metadata["canonicalization"] == "identity"
    assert metadata["canonical_space"] == "authored_structural_factor_label_space"


def test_loaded_rows_are_manifest_validated_and_defensive_copies() -> None:
    first = load_train_rows()
    first[0]["inputs"]["text"] = "mutated caller copy"
    second = load_train_rows()
    assert second[0]["inputs"]["text"] != "mutated caller copy"
    report = runner._fixture_module().validate(runner._validated_bundle())
    assert report["content_digest_sha256"] == BENCHMARK_DIGEST


def test_prepare_and_collate_keep_source_and_target_channels_separate() -> None:
    raw = load_train_rows()[0]
    prepared = prepare_row(raw)
    assert prepared.source_ids == tuple(runner._fixture_module().source_ids(raw))
    assert prepared.decoder_input_ids[0] == 1
    assert prepared.labels[-1] == 2
    assert raw["targets"]["text"].encode("utf-8") != bytes(
        token - 4 for token in prepared.source_ids if 4 <= token < 260
    )
    batch = collate_rows([prepare_row(row) for row in load_train_rows()[::2][:2]])
    assert torch.all(batch.labels[~batch.decoder_mask] == -100)
    assert torch.all(batch.source_ids[~batch.source_mask] == 0)


def test_local_gates_depend_only_on_consumed_target_prefix() -> None:
    row = prepare_row(load_train_rows()[0])
    gates = teacher_forced_gates(row, FrozenLocalPrefixFSM())
    assert tuple(gates.shape) == (row.decoder_length, 4)
    assert not gates[0].any()
    changed_inputs = list(row.decoder_input_ids)
    changed_inputs[-1] = 4 + ((changed_inputs[-1] - 4 + 1) % 256)
    changed = type(row)(row.source_ids, tuple(changed_inputs), row.labels,
                        row.factor_targets, row.target_text, row.target_frame)
    assert torch.equal(gates[:-1], teacher_forced_gates(changed, FrozenLocalPrefixFSM())[:-1])


@pytest.mark.parametrize("arm", ARM_IDS)
def test_single_step_uses_common_lm_plus_four_factor_loss(arm: str) -> None:
    batch = collate_rows([prepare_row(row) for row in load_train_rows()[:2]])
    model = build_model(arm, seed=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    value = training_step(model, optimizer, batch)
    assert isinstance(value, float) and value == value


def test_generation_and_true_intervention_use_bos_self_history() -> None:
    row = prepare_row(load_train_rows()[0])
    model = build_model("H0L0", seed=7)
    baseline = greedy_generate(model, row.source_ids)
    one_hot = torch.tensor([[0, 1, 0, 0]], dtype=torch.float32)
    changed = greedy_generate_intervened(model, row.source_ids, "event", one_hot)
    assert 0 < len(baseline["tokens"]) <= 96
    assert 0 < len(changed["tokens"]) <= 96
    with pytest.raises(ValueError, match="frozen at 96"):
        greedy_generate(model, row.source_ids, max_new_tokens=95)


def test_batched_generation_matches_one_row_wrapper() -> None:
    rows = [prepare_row(row) for row in load_train_rows()[::2][:2]]
    model = build_model("D_AUX", seed=7)
    batch = greedy_generate_batch(model, [row.source_ids for row in rows])
    singles = [greedy_generate(model, row.source_ids) for row in rows]
    assert batch == singles


def test_development_evaluation_emits_both_intervention_schemas_and_never_final() -> None:
    model = build_model("D_AUX", seed=7)
    report = evaluate_benchmark_v3(model, rows=load_development_rows()[:2])
    assert report["version"] == "benchmark-v3-metrics-1"
    assert report["row_count"] == 2
    assert report["teacher_forced"]["diagnostic_only"] is True
    assert report["source_side_swap"]["rows"] == 4
    assert report["intermediate_probability_intervention"]["rows"] == 4
    final = copy.deepcopy(load_development_rows()[0])
    final["split"] = "final-confirmation"
    with pytest.raises(PermissionError, match="unchanged diagnostic-validation"):
        evaluate_benchmark_v3(model, rows=[final])


def test_frozen_step_constant_is_not_a_test_training_request() -> None:
    assert TRAIN_STEPS == 600
