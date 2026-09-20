from __future__ import annotations

import hashlib
import json
from pathlib import Path

from norishio_lm.benchmark_v3_h0_execute import _aggregate_swaps
from norishio_lm.benchmark_v3_h0_metrics import CONTROL_TYPES, aggregate_control_records
from norishio_lm.benchmark_v3_h0_protocol import (
    FIXTURE_CONTENT_SHA256,
    PROTOCOL_SHA256,
    load_protocol,
)
ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "docs" / "results" / "benchmark-v3-h0-confirmation"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_manifest_and_six_run_bindings_are_complete() -> None:
    descriptor = load_protocol()
    manifest = _load(RESULT_ROOT / "artifact-manifest.json")
    runs = [_load(path) for path in sorted(RESULT_ROOT.glob("run-*.json"))]

    assert manifest["descriptor_sha256"] == PROTOCOL_SHA256
    assert manifest["fixture_content_digest_sha256"] == FIXTURE_CONTENT_SHA256
    for name, expected in manifest["artifacts"].items():
        path = RESULT_ROOT / name
        assert path.stat().st_size == expected["bytes"]
        assert _sha(path) == expected["sha256"]
    assert len(runs) == 6
    assert [f"{run['arm']} seed{run['seed']}" for run in runs] == descriptor["training"]["run_order"]
    assert all(run["status"] == "complete" for run in runs)
    assert sum(run["training"]["optimizer_updates"] for run in runs) == 3600
    for run in runs:
        seed = str(run["seed"])
        assert run["descriptor_sha256"] == PROTOCOL_SHA256
        assert run["fixture_content_digest_sha256"] == FIXTURE_CONTENT_SHA256
        assert run["training"]["trainable_parameters"] == 32120
        assert run["training"]["initial_state_sha256"] == descriptor["model"]["initial_state_by_seed"][seed][run["arm"]]
        assert run["training"]["schedule_sha256"] == descriptor["training"]["schedule"]["seeded_cpu_torch_schedule_sha256"][seed]
        assert len(run["ordinary_confirmation"]["raw_confirmation_rows"]) == 96
        assert len(run["interventions"]["controls"]) == 24
        assert len(run["interventions"]["source_swaps"]) == 8


def test_saved_ordinary_and_intervention_aggregates_recompute_from_raw() -> None:
    descriptor = load_protocol()
    summary = _load(RESULT_ROOT / "summary.json")
    runs = [_load(path) for path in sorted(RESULT_ROOT.glob("run-*.json"))]
    common = set(descriptor["raw_artifact_schema"]["common_required_fields"])
    controls_required = set(descriptor["raw_artifact_schema"]["internal_control_required_fields"])
    swaps_required = set(descriptor["raw_artifact_schema"]["source_swap_required_fields"])
    aggregate_required = set(descriptor["raw_artifact_schema"]["aggregate_required_fields"])

    for run in runs:
        report = run["ordinary_confirmation"]
        raw = report["raw_confirmation_rows"]
        assert all(common <= set(row) for row in raw)
        for support, report_name in (("unseen_pair", "unseen_pair"), ("seen_pair/unseen_triple", "seen_pair")):
            selected = [row for row in raw if row["support_group"] == support]
            assert len(selected) == 48
            generation_count = sum(row["parsed_frame"] == row["target_frame"] for row in selected)
            assert generation_count == report["groups"][report_name]["generation_frame_exact"]["correct"]
        generation_count = sum(row["parsed_frame"] == row["target_frame"] for row in raw)
        head_count = sum(row["intermediate_frame"] == row["target_frame"] for row in raw)
        text_count = sum(row["generated_text"] == row["target_text"] for row in raw)
        assert generation_count == report["all"]["generation_frame_exact"]["correct"]
        assert head_count == report["all"]["intermediate_frame_exact"]["correct"]
        assert text_count == report["all"]["exact_target_text"]["correct"]
        assert all(controls_required <= set(row) for row in run["interventions"]["controls"])
        assert all(swaps_required <= set(row) for row in run["interventions"]["source_swaps"])

    for arm in ("H1L1", "H1L1_ANCHOR"):
        arm_runs = [run for run in runs if run["arm"] == arm]
        controls = [row for run in arm_runs for row in run["interventions"]["controls"]]
        for control in CONTROL_TYPES:
            recomputed = aggregate_control_records(
                [row for row in controls if row["control_type"] == control],
                expected_scheduled_count=24,
            )
            assert aggregate_required <= set(recomputed)
            assert recomputed == summary["arms"][arm]["controls"][control]
        swaps = [row for run in arm_runs for row in run["interventions"]["source_swaps"]]
        assert _aggregate_swaps(swaps) == summary["arms"][arm]["source_swaps"]
    assert summary["status"] == "complete"
    assert summary["optimizer_updates"] == 3600
    assert summary["decisions"]["claim"] == "decoder_path_inconclusive_due_to_weak_heads"
