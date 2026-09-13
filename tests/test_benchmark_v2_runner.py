from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v2_fsm import FrozenLocalPrefixFSM
from norishio_lm.benchmark_v2_model import build_model
from norishio_lm.benchmark_v2_runner import (
    TRAIN_STEPS,
    collate_rows,
    evaluate_benchmark_v2,
    factor_vocabulary,
    greedy_generate,
    greedy_generate_batch,
    load_development_rows,
    load_train_rows,
    prepare_row,
    teacher_forced_gates,
    training_step,
)
from norishio_lm.benchmark_v2_tournament import BENCHMARK_DIGEST, load_tournament


def test_factor_vocabulary_is_derived_from_committed_spec() -> None:
    vocabulary = factor_vocabulary()
    assert vocabulary.values == {
        "participant": ("FRIEND", "COLLEAGUE", "SENIOR", "TEACHER", "FAMILY", "JUNIOR"),
        "time": ("TODAY", "TOMORROW", "WEEKEND", "NEXT_WEEK", "MORNING", "NIGHT"),
        "event": ("MEET", "TALK", "VISIT", "HELP"),
        "operator": ("ASSERT", "WANT", "PLAN", "NEGATE"),
    }
    assert vocabulary.encode({
        "participant": "FRIEND", "time": "TODAY", "event": "MEET", "operator": "ASSERT",
    }) == (0, 0, 0, 0)


def test_loaded_rows_are_manifest_validated_and_defensive_copies() -> None:
    first = load_train_rows()
    first[0]["inputs"]["text"] = "mutated caller copy"
    second = load_train_rows()
    assert second[0]["inputs"]["text"] != "mutated caller copy"
    benchmark = __import__("norishio_lm.benchmark_v2_runner", fromlist=["_fixture_module"])
    report = benchmark._fixture_module().validate(benchmark._validated_bundle())
    assert report["content_digest_sha256"] == BENCHMARK_DIGEST


def test_collation_has_strict_source_and_decoder_padding() -> None:
    rows = [prepare_row(row) for row in load_train_rows()[::2][:2]]
    # These two authored variants have different source lengths, so this
    # checks both sides of the padding contract without constructing data.
    assert rows[0].source_length != rows[1].source_length
    batch = collate_rows(rows)
    assert batch.source_ids.dtype is torch.long
    assert batch.decoder_input_ids.dtype is torch.long
    assert batch.source_mask.dtype is torch.bool
    assert batch.decoder_mask.dtype is torch.bool
    assert torch.all(batch.source_ids[~batch.source_mask] == 0)
    assert torch.all(batch.decoder_input_ids[~batch.decoder_mask] == 0)
    assert torch.all(batch.labels[~batch.decoder_mask] == -100)
    assert torch.all(batch.source_ids[batch.source_mask] != 0)


def test_b_gates_use_consumed_target_prefix_and_zero_source_padding() -> None:
    row = prepare_row(load_train_rows()[0])
    gates = teacher_forced_gates(row, FrozenLocalPrefixFSM())
    assert tuple(gates.shape) == (row.decoder_length, 4)
    assert row.decoder_input_ids[0] == 1  # BOS starts the target-only decoder history.
    assert row.labels[-1] == 2  # EOS is the final next-token target.
    assert not gates[0].any()
    # Changing a future target byte cannot change earlier gates.
    changed = copy.copy(row)
    changed_inputs = list(row.decoder_input_ids)
    changed_inputs[-1] = 4 + ((changed_inputs[-1] - 4 + 1) % 256)
    changed = type(row)(
        row.source_ids, tuple(changed_inputs), row.labels, row.factor_targets,
        row.target_text, row.target_frame,
    )
    changed_gates = teacher_forced_gates(changed, FrozenLocalPrefixFSM())
    assert torch.equal(gates[:-1], changed_gates[:-1])


def test_single_step_helper_supports_focused_cpu_checks_without_training_budget() -> None:
    rows = [prepare_row(row) for row in load_train_rows()[:2]]
    batch = collate_rows(rows)
    model = build_model("B", seed=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    value = training_step(model, optimizer, batch)
    assert isinstance(value, float)
    assert value == value


def test_public_generation_is_frozen_to_bos_self_history() -> None:
    row = prepare_row(load_train_rows()[0])
    model = build_model("D", seed=7)
    result = greedy_generate(model, row.source_ids)
    assert len(result["tokens"]) <= 96
    assert result["tokens"]
    with pytest.raises(ValueError, match="frozen at 96"):
        greedy_generate(model, row.source_ids, max_new_tokens=95)


def test_batched_generation_matches_one_row_wrapper() -> None:
    rows = [prepare_row(row) for row in load_train_rows()[::2][:2]]
    model = build_model("D", seed=7)
    batch = greedy_generate_batch(model, [row.source_ids for row in rows])
    singles = [greedy_generate(model, row.source_ids) for row in rows]
    assert batch == singles


def test_batched_generation_compacts_active_histories_after_early_eos() -> None:
    class Stub:
        arm = "D"

        def eval(self):
            return self

        def __call__(self, source, decoder, **kwargs):
            logits = torch.full((source.shape[0], decoder.shape[1], 260), -1.0)
            for index in range(source.shape[0]):
                token = 2 if int(source[index, 0]) == 4 else 5
                logits[index, -1, token] = 1.0
            return {"logits": logits}

    output = greedy_generate_batch(Stub(), [[4], [5]])
    assert output[0]["ended_eos"] is True
    assert len(output[0]["tokens"]) == 1
    assert output[1]["ended_eos"] is False
    assert len(output[1]["tokens"]) == 96


def test_development_evaluation_is_scorer_compatible_and_never_final() -> None:
    model = build_model("D", seed=7)
    report = evaluate_benchmark_v2(model, rows=load_development_rows()[:2])
    assert report["version"] == "benchmark-v2-metrics-1"
    assert report["row_count"] == 2
    assert report["teacher_forced_bytes"]["diagnostic_only"] is True
    assert report["intervention_locality"]["rows"] == 4
    assert set(load_tournament()["metrics"]["required"]) <= set(report)
    final = copy.deepcopy(load_development_rows()[0])
    final["split"] = "final-holdout"
    with pytest.raises(PermissionError, match="unchanged diagnostic-validation"):
        evaluate_benchmark_v2(model, rows=[final])


def test_frozen_step_constant_is_not_a_test_training_request() -> None:
    assert TRAIN_STEPS == 600
