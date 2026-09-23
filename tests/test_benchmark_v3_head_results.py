"""Verify the committed Issue #46 result from retained raw artifacts only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from norishio_lm.benchmark_v3_head_execute import aggregate_saved_runs
from norishio_lm.benchmark_v3_head_protocol import load_protocol


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "benchmark-v3-head-learning"


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_six_saved_runs_reaggregate_without_training_or_inference() -> None:
    freeze = load_protocol()
    saved = _load("summary.json")
    recomputed = aggregate_saved_runs()
    assert recomputed == saved
    assert saved["descriptor_sha256"] == freeze["descriptor_sha256"]
    assert saved["attempted_runs"] == 6
    assert saved["optimizer_updates"] == 3600
    assert saved["decision"]["branch"] == "basic_optimization_or_capacity_unresolved"
    assert saved["metrics"]["row_count"] == 6 * (384 + 96)
    assert saved["executor_total_wall_seconds"] >= saved["training_wall_seconds"] > 0


def test_artifact_manifest_preserves_byte_exact_result_files() -> None:
    manifest = _load("artifact-manifest.json")
    assert manifest["descriptor_sha256"] == load_protocol()["descriptor_sha256"]
    assert len(manifest["artifacts"]) == 9
    for name, expected in manifest["artifacts"].items():
        path = RESULTS / name
        assert path.stat().st_size == expected["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
