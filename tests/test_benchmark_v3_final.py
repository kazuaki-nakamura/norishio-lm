from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_checkpoint import (
    expected_schedule_sha256, save_tournament_checkpoint, state_sha256,
)
import norishio_lm.benchmark_v3_final as final_gate
from norishio_lm.benchmark_v3_final import checkpoint_path, evaluate_final_once, main
from norishio_lm.benchmark_v3_model import build_model
from norishio_lm.benchmark_v3_protocol import (
    complete_record, failed_record, write_terminal_record,
)
from norishio_lm.benchmark_v3_tournament import ARM_COUNTS, SEEDS


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


def test_all_failed_set_refuses_without_consuming_invocation(tmp_path):
    _write_all_failed(tmp_path)
    called = False

    def evaluator(*_):
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="all 18"):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    assert called is False
    assert not (tmp_path / "final-invocation").exists()


def test_result_attestation_binds_result_and_frozen_digests(tmp_path, monkeypatch):
    records = [{"arm": arm, "seed": seed, "status": "complete"}
               for arm in ARM_COUNTS for seed in SEEDS]
    complete = [(record, {"model": object()}) for record in records]
    monkeypatch.setattr(final_gate, "collect_terminal_records", lambda _root: records)
    monkeypatch.setattr(final_gate, "_validate_complete_checkpoints", lambda *_: complete)
    monkeypatch.setattr(final_gate, "_fixture_module", lambda: SimpleNamespace(
        build=lambda: {}, evaluation_rows=lambda *_args, **_kwargs: [{}],
    ))
    evaluate_final_once(tmp_path, evaluate_final=True, evaluator=lambda *_: {})

    result_path = tmp_path / "final-invocation" / "result.json"
    result = json.loads(result_path.read_text("utf-8"))
    attestation = json.loads(
        (tmp_path / "final-invocation" / "result-attestation.json").read_text("utf-8")
    )
    assert attestation == {
        "schema": "norishio.issue36.result-attestation.v1",
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "result_schema": result["schema"],
        "terminal_records_sha256": result["terminal_records_sha256"],
        "benchmark_content_digest_sha256": result["benchmark_content_digest_sha256"],
    }


def test_second_invocation_is_rejected_before_evaluator(tmp_path, monkeypatch):
    records = [{"arm": arm, "seed": seed, "status": "complete"}
               for arm in ARM_COUNTS for seed in SEEDS]
    complete = [(record, {"model": object()}) for record in records]
    monkeypatch.setattr(final_gate, "collect_terminal_records", lambda _root: records)
    monkeypatch.setattr(final_gate, "_validate_complete_checkpoints", lambda *_: complete)
    monkeypatch.setattr(final_gate, "_fixture_module", lambda: SimpleNamespace(
        build=lambda: {}, evaluation_rows=lambda *_args, **_kwargs: [{}],
    ))
    calls = 0

    def evaluator(*_args):
        nonlocal calls
        calls += 1
        return {}

    evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    with pytest.raises(FileExistsError):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    assert calls == len(ARM_COUNTS) * len(SEEDS)


def test_attestation_write_failure_publishes_only_a_complete_failure_pair(
    tmp_path, monkeypatch
):
    records = [{"arm": arm, "seed": seed, "status": "complete"}
               for arm in ARM_COUNTS for seed in SEEDS]
    complete = [(record, {"model": object()}) for record in records]
    monkeypatch.setattr(final_gate, "collect_terminal_records", lambda _root: records)
    monkeypatch.setattr(final_gate, "_validate_complete_checkpoints", lambda *_: complete)
    monkeypatch.setattr(final_gate, "_fixture_module", lambda: SimpleNamespace(
        build=lambda: {}, evaluation_rows=lambda *_args, **_kwargs: [{}],
    ))
    original = final_gate._atomic_exclusive_json
    failed_once = False

    def fail_first_attestation(path: Path, value):
        nonlocal failed_once
        if path.name == "result-attestation.json" and not failed_once:
            failed_once = True
            raise OSError("injected attestation failure")
        return original(path, value)

    monkeypatch.setattr(final_gate, "_atomic_exclusive_json", fail_first_attestation)
    with pytest.raises(OSError, match="injected attestation failure"):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=lambda *_: {})

    result_path = tmp_path / "final-invocation" / "result.json"
    attestation_path = tmp_path / "final-invocation" / "result-attestation.json"
    assert result_path.exists() and attestation_path.exists()
    result = json.loads(result_path.read_text("utf-8"))
    attestation = json.loads(attestation_path.read_text("utf-8"))
    assert result["status"] == "failed"
    assert attestation["result_sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()


def test_selection_summary_reports_three_seed_means_and_tie_break_trace():
    def evaluation(arm, seed, free, pair, triple, balanced):
        return {
            "arm": arm,
            "seed": seed,
            "result": {
                "all": {
                    "free_generation_exact": {"accuracy": free},
                    "pair_exact": {"accuracy": pair},
                    "triple_exact": {"accuracy": triple},
                    "exact_target_text": {"accuracy": free},
                    "atomic_balanced_accuracy": {
                        "participant": balanced,
                        "time": balanced,
                        "event": balanced,
                        "operator": balanced,
                    },
                },
            },
        }

    evaluations = [
        evaluation(arm, seed, free, frame, triple, balanced)
        for arm, free, frame, triple, balanced in (
            ("H0L0", 0.4, 0.2, 0.1, 0.5),
            ("H0L1", 0.4, 0.3, 0.1, 0.5),
            ("H1L0", 0.8, 0.1, 0.1, 0.1),
        )
        for seed in SEEDS
    ]
    summary = final_gate._selection_summary(evaluations)

    assert summary["status"] == "complete"
    assert summary["ranking"] == ["H1L0", "H0L1", "H0L0"]
    assert summary["three_seed_means"]["H0L0"]["three_seed_mean"][
        "free_generation_exact.accuracy"
    ] == pytest.approx(0.4)
    tie = next(item for item in summary["tie_break_trace"] if item["stage"] == "tie_break_2")
    assert tie["candidates"] == ["H0L0", "H0L1"]
    assert tie["survivors"] == ["H0L1"]


def test_default_evaluator_metric_drift_consumes_and_records_failure(tmp_path, monkeypatch):
    records = [{"arm": arm, "seed": seed, "status": "complete"}
               for arm in ARM_COUNTS for seed in SEEDS]
    monkeypatch.setattr(final_gate, "collect_terminal_records", lambda _root: records)
    monkeypatch.setattr(final_gate, "_validate_complete_checkpoints", lambda _root, _records: [
        (record, {"model": object()}) for record in records
    ])
    monkeypatch.setattr(final_gate, "_default_evaluator", lambda *_args: {})
    monkeypatch.setattr(final_gate, "_fixture_module", lambda: SimpleNamespace(
        build=lambda: {},
        evaluation_rows=lambda *_args, **_kwargs: [{}],
    ))

    with pytest.raises(ValueError, match="selection metrics"):
        evaluate_final_once(tmp_path, evaluate_final=True)

    result = json.loads((tmp_path / "final-invocation" / "result.json").read_text("utf-8"))
    assert result["status"] == "failed"
    assert (tmp_path / "final-invocation" / "result-attestation.json").exists()


def test_marker_publish_cleans_temporary_directory_on_failure(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("marker write failed")

    monkeypatch.setattr(final_gate, "_atomic_exclusive_json", fail)
    with pytest.raises(OSError, match="marker write failed"):
        final_gate._publish_invocation_marker(tmp_path, {"schema": "test"})
    assert not (tmp_path / "final-invocation").exists()
    assert not (tmp_path / ".final-invocation.lock").exists()
    assert list(tmp_path.glob(".final-invocation.*")) == []


def test_incomplete_terminal_set_does_not_consume_invocation(tmp_path):
    write_terminal_record(tmp_path, failed_record(
        arm="H0L0", seed=7, phase="training", error_type="FixtureFailure",
        redacted_message="fixture terminal failure",
    ))
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=lambda *_: {})
    assert not (tmp_path / "final-invocation").exists()


def test_one_complete_checkpoint_with_failures_does_not_open_final(tmp_path):
    model = build_model("H0L0", seed=7)
    initial = state_sha256(model.state_dict())
    model.encoder.projection.weight.data.add_(0.01)
    path = checkpoint_path(tmp_path, "H0L0", 7)
    save_tournament_checkpoint(
        path, model, arm="H0L0", seed=7,
        schedule_sha256=expected_schedule_sha256(7), initial_state_sha256=initial,
    )
    write_terminal_record(tmp_path, complete_record(path, arm="H0L0", seed=7))
    for arm in ARM_COUNTS:
        for seed in SEEDS:
            if (arm, seed) == ("H0L0", 7):
                continue
            write_terminal_record(tmp_path, failed_record(
                arm=arm, seed=seed, phase="training", error_type="FixtureFailure",
                redacted_message="fixture terminal failure",
            ))

    called = False
    def evaluator(*_args):
        nonlocal called
        called = True
    with pytest.raises(ValueError, match="all 18"):
        evaluate_final_once(tmp_path, evaluate_final=True, evaluator=evaluator)
    assert called is False
    assert not (tmp_path / "final-invocation").exists()
