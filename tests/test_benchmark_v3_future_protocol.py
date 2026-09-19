from __future__ import annotations

import copy
import json

import pytest

from norishio_lm.benchmark_v3_future_protocol import (
    DIGEST_FIELD,
    FUTURE_PROTOCOL_DESCRIPTOR,
    FUTURE_PROTOCOL_SHA256,
    PROTOCOL_PATH,
    USAGE_PHASES,
    bind_future_run_metadata,
    load_future_protocol,
    protocol_digest,
    run_future_development,
    run_future_final,
    run_validated_future_phase,
    validate_future_protocol,
)


def _bound(**values):
    return {DIGEST_FIELD: FUTURE_PROTOCOL_SHA256, **values}


def test_published_descriptor_digest_and_scope_are_frozen() -> None:
    descriptor = validate_future_protocol()

    assert protocol_digest(descriptor) == FUTURE_PROTOCOL_SHA256
    assert FUTURE_PROTOCOL_SHA256 == (
        "bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c"
    )
    assert descriptor["scope"] == {
        "applies_to": "future_benchmark_v3_runs_only",
        "historical_issue36_artifacts": "excluded_immutable",
    }
    assert descriptor["execution_binding"]["required_before"] == list(USAGE_PHASES)


def test_tracked_protocol_file_matches_public_descriptor() -> None:
    assert PROTOCOL_PATH.is_file()
    assert load_future_protocol() == FUTURE_PROTOCOL_DESCRIPTOR
    assert protocol_digest(load_future_protocol()) == FUTURE_PROTOCOL_SHA256


def test_tracked_protocol_file_rejects_descriptor_drift(tmp_path) -> None:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["descriptor"]["selection"]["primary"] = (
        "all.free_generation_exact.accuracy"
    )
    candidate = tmp_path / "future-protocol-v1.json"
    candidate.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="descriptor digest mismatch"):
        load_future_protocol(candidate)


def test_descriptor_freezes_selection_and_intervention_contracts() -> None:
    descriptor = validate_future_protocol()
    selection = descriptor["selection"]

    assert selection["primary"] == "all.generation_frame_exact.accuracy"
    assert selection["tie_breakers"] == [
        "all.triple_exact.accuracy",
        "all.pair_exact.accuracy",
        "all.atomic_balanced_accuracy.mean",
        "all.exact_target_text.accuracy",
        "arm_id_ascending",
    ]
    assert selection["derived_metrics"]["all.atomic_balanced_accuracy.mean"] == {
        "source_paths": [
            "all.atomic_balanced_accuracy.participant",
            "all.atomic_balanced_accuracy.time",
            "all.atomic_balanced_accuracy.event",
            "all.atomic_balanced_accuracy.operator",
        ],
        "reducer": "unweighted_arithmetic_mean",
    }
    assert selection["aggregation"] == "unweighted_three_seed_mean"
    assert selection["direction"] == "maximize"
    assert descriptor["intervention"]["rule"] == (
        "alternate_after_baseline_argmax_mod_width"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["selection"].__setitem__(
            "primary", "all.free_generation_exact.accuracy"
        ),
        lambda value: value["selection"]["tie_breakers"].__setitem__(
            0, "all.pair_exact.accuracy"
        ),
        lambda value: value["selection"]["derived_metrics"][
            "all.atomic_balanced_accuracy.mean"
        ]["source_paths"].pop(),
        lambda value: value["selection"]["derived_metrics"][
            "all.atomic_balanced_accuracy.mean"
        ].__setitem__("reducer", "weighted_mean"),
        lambda value: value["selection"].__setitem__(
            "aggregation", "weighted_three_seed_mean"
        ),
        lambda value: value["selection"].__setitem__("direction", "minimize"),
        lambda value: value["execution_binding"]["required_before"].pop(),
        lambda value: value["intervention"].__setitem__("rule", "fixed_class_zero"),
    ],
)
def test_descriptor_drift_is_rejected_before_callback(mutate) -> None:
    descriptor = copy.deepcopy(FUTURE_PROTOCOL_DESCRIPTOR)
    mutate(descriptor)
    metadata = bind_future_run_metadata("fixture-run")
    calls: list[str] = []

    with pytest.raises(ValueError, match="frozen payload"):
        run_validated_future_phase(
            "future_training",
            lambda _: calls.append("training"),
            run_metadata=metadata,
            descriptor=descriptor,
        )

    assert calls == []


def test_run_metadata_digest_mismatch_blocks_training() -> None:
    metadata = bind_future_run_metadata("fixture-run")
    metadata[DIGEST_FIELD] = "0" * 64
    calls: list[str] = []

    with pytest.raises(ValueError, match="metadata protocol digest"):
        run_validated_future_phase(
            "future_training",
            lambda _: calls.append("training"),
            run_metadata=metadata,
        )

    assert calls == []


def test_trainer_binding_mismatch_blocks_evaluator() -> None:
    metadata = bind_future_run_metadata("fixture-run")
    calls: list[str] = []

    def trainer(_):
        calls.append("trainer")
        return {DIGEST_FIELD: "0" * 64}

    def evaluator(*_):
        calls.append("evaluator")
        return _bound()

    with pytest.raises(ValueError, match="trainer result protocol digest"):
        run_future_development(trainer, evaluator, run_metadata=metadata)

    assert calls == ["trainer"]


def test_valid_future_development_binds_training_and_evaluation() -> None:
    metadata = bind_future_run_metadata("fixture-run")
    result = run_future_development(
        lambda bound: _bound(kind="training", run_id=bound["run_id"]),
        lambda training, bound: _bound(
            kind="development", training_kind=training["kind"], run_id=bound["run_id"]
        ),
        run_metadata=metadata,
    )

    assert result["run_metadata"] == metadata
    assert result["evaluation"]["training_kind"] == "training"


def test_final_descriptor_drift_blocks_marker_rows_and_evaluator() -> None:
    metadata = bind_future_run_metadata("fixture-final")
    descriptor = copy.deepcopy(FUTURE_PROTOCOL_DESCRIPTOR)
    descriptor["selection"]["primary"] = "all.free_generation_exact.accuracy"
    calls: list[str] = []

    def callback(name):
        def inner(*_):
            calls.append(name)
            return _bound(kind=name)
        return inner

    with pytest.raises(ValueError, match="frozen payload"):
        run_future_final(
            callback("marker"), callback("rows"), callback("evaluator"),
            run_metadata=metadata, descriptor=descriptor,
        )

    assert calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["selection"].__setitem__(
            "primary", "all.free_generation_exact.accuracy"
        ),
        lambda value: value["selection"]["tie_breakers"].__setitem__(
            0, "all.pair_exact.accuracy"
        ),
        lambda value: value["selection"]["derived_metrics"][
            "all.atomic_balanced_accuracy.mean"
        ]["source_paths"].pop(),
        lambda value: value["selection"]["derived_metrics"][
            "all.atomic_balanced_accuracy.mean"
        ].__setitem__("reducer", "weighted_mean"),
    ],
)
def test_selection_drift_blocks_every_future_execution_callback(mutate) -> None:
    descriptor = copy.deepcopy(FUTURE_PROTOCOL_DESCRIPTOR)
    mutate(descriptor)
    metadata = bind_future_run_metadata("fixture-all-phases")
    calls: list[str] = []

    def callback(name):
        def inner(*_):
            calls.append(name)
            return _bound(kind=name)
        return inner

    with pytest.raises(ValueError, match="frozen payload"):
        run_future_development(
            callback("trainer"), callback("development-evaluator"),
            run_metadata=metadata, descriptor=descriptor,
        )
    with pytest.raises(ValueError, match="frozen payload"):
        run_future_final(
            callback("marker"), callback("rows"), callback("final-evaluator"),
            run_metadata=metadata, descriptor=descriptor,
        )

    assert calls == []


def test_valid_future_final_orders_marker_rows_and_evaluator() -> None:
    metadata = bind_future_run_metadata("fixture-final")
    calls: list[str] = []

    def marker(_):
        calls.append("marker")
        return _bound(kind="marker")

    def rows(_):
        calls.append("rows")
        return _bound(kind="rows")

    def evaluator(row_bundle, _):
        calls.append("evaluator")
        return _bound(kind="result", source=row_bundle["kind"])

    result = run_future_final(
        marker, rows, evaluator, run_metadata=metadata,
    )

    assert calls == ["marker", "rows", "evaluator"]
    assert result["result"]["source"] == "rows"
