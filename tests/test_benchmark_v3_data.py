"""Structural and provenance tests for the Issue #36 Phase 0 corpus."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from norishio_lm import benchmark_v3_data as benchmark


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def bundle() -> dict[str, list[dict]]:
    return benchmark.build()


def _frames(rows: list[dict]) -> list[dict]:
    return [row["targets"]["frame"] for row in rows]


def test_deterministic_counts_and_manifest(bundle: dict[str, list[dict]]) -> None:
    assert bundle == benchmark.build()
    report = benchmark.validate(bundle)
    benchmark.check_expected(report)
    assert report["counts"] == {split: 384 for split in benchmark.SPLITS}
    assert report["groups"] == {split: 192 for split in benchmark.SPLITS}
    assert report["support_classes"] == {
        "train": {"train": 384},
        "diagnostic-validation": {"unseen_triple": 192, "unseen_pair": 192},
        "final-confirmation": {"unseen_triple": 192, "unseen_pair": 192},
    }
    assert report["checks"]["v2_rows_not_reused"] is True
    assert report["content_digest_sha256"]


def test_train_covers_atoms_and_evaluations_are_split_disjoint(bundle: dict[str, list[dict]]) -> None:
    train_frames = _frames(bundle["train"])
    support_pairs = {(f["participant"], f["time"]) for f in train_frames}
    support_triples = {(f["participant"], f["time"], f["event"]) for f in train_frames}
    spec = benchmark.spec_data()
    assert {field: {frame[field] for frame in train_frames} for field in benchmark.FIELDS} == {
        "participant": {x["id"] for x in spec["participants"]},
        "time": {x["id"] for x in spec["times"]},
        "event": {x["id"] for x in spec["events"]},
        "operator": set(spec["operators"]),
    }
    for split in benchmark.EVALUATION_SPLITS:
        classes = set()
        for frame in _frames(bundle[split]):
            pair = (frame["participant"], frame["time"])
            triple = (*pair, frame["event"])
            assert triple not in support_triples
            classes.add("unseen_pair" if pair not in support_pairs else "unseen_triple")
        assert classes == {"unseen_pair", "unseen_triple"}


def test_source_and_target_texts_are_distinct_and_globally_disjoint(
    bundle: dict[str, list[dict]],
) -> None:
    source_texts = {
        row["inputs"]["text"] for rows in bundle.values() for row in rows
    }
    target_texts = {
        row["targets"]["text"] for rows in bundle.values() for row in rows
    }
    assert all(row["inputs"]["text"] != row["targets"]["text"]
               for rows in bundle.values() for row in rows)
    assert source_texts.isdisjoint(target_texts)


def test_variants_stay_together_and_each_triple_has_all_operators(bundle: dict[str, list[dict]]) -> None:
    groups: dict[str, list[dict]] = {}
    operators: dict[tuple[str, str, str, str], set[str]] = {}
    for split, rows in bundle.items():
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
            frame = row["targets"]["frame"]
            operators.setdefault((split, frame["participant"], frame["time"], frame["event"]), set()).add(frame["operator"])
    assert all(len(rows) == 2 and len({row["split"] for row in rows}) == 1 for rows in groups.values())
    assert all(values == set(benchmark.spec_data()["operators"]) for values in operators.values())


def test_v3_evaluation_surface_records_do_not_reuse_v2_records(bundle: dict[str, list[dict]]) -> None:
    path = ROOT / "data" / "benchmark_v2" / "benchmark.py"
    module_spec = importlib.util.spec_from_file_location("benchmark_v2_fixture_for_v3_test", path)
    assert module_spec is not None and module_spec.loader is not None
    old = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(old)
    old_surface = {
        (row["inputs"]["text"], row["targets"]["text"], tuple(row["targets"]["frame"].items()))
        for split in ("diagnostic-validation", "final-holdout") for row in old.build()[split]
    }
    current_surface = {
        (row["inputs"]["text"], row["targets"]["text"], tuple(row["targets"]["frame"].items()))
        for split in benchmark.EVALUATION_SPLITS for row in bundle[split]
    }
    assert old_surface.isdisjoint(current_surface)


@pytest.mark.parametrize("field", ["targets", "metadata", "id", "group_id", "split"])
def test_hidden_fields_cannot_change_source_encoding(bundle: dict[str, list[dict]], field: str) -> None:
    row = bundle["train"][0]
    changed = copy.deepcopy(row)
    changed[field] = {"hidden": "MUST_NOT_ENTER_SOURCE"}
    assert benchmark.model_inputs(row) == benchmark.model_inputs(changed)
    assert benchmark.source_ids(row) == benchmark.source_ids(changed)


def test_input_allowlist_and_teacher_forcing_boundary(bundle: dict[str, list[dict]]) -> None:
    changed = copy.deepcopy(bundle["train"][0])
    changed["inputs"]["gold_frame"] = {"participant": "LEAK"}
    with pytest.raises(ValueError, match="only context and text"):
        benchmark.model_inputs(changed)
    packed = benchmark.teacher_forcing(bundle["train"][0])
    source = benchmark.source_ids(bundle["train"][0])
    assert packed["labels"][: len(source) - 1] == [-100] * (len(source) - 1)
    assert packed["input_ids"][: len(source)] == source
    assert len(packed["input_ids"]) == len(packed["labels"])


def test_final_confirmation_requires_manifest_binding(bundle: dict[str, list[dict]]) -> None:
    report = benchmark.validate(bundle)
    assert len(benchmark.evaluation_rows(bundle, "diagnostic-validation")) == 384
    with pytest.raises(PermissionError, match="explicit"):
        benchmark.evaluation_rows(bundle, "final-confirmation")
    with pytest.raises(ValueError, match="digest"):
        benchmark.evaluation_rows(bundle, "final-confirmation", evaluate_final=True, manifest_digest="wrong")
    rows = benchmark.evaluation_rows(bundle, "final-confirmation", evaluate_final=True, manifest_digest=report["content_digest_sha256"])
    assert rows == bundle["final-confirmation"] and rows is not bundle["final-confirmation"]


def test_committed_manifest_rejects_spec_generator_and_bundle_changes(
    bundle: dict[str, list[dict]], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = benchmark.validate(bundle)
    benchmark.check_expected(report)

    changed_spec = benchmark.spec_data()
    changed_spec["split_seed"] += 1
    with pytest.raises(ValueError, match="expected-manifest"):
        benchmark.check_expected(benchmark.validate(benchmark.build(changed_spec), changed_spec))

    monkeypatch.setattr(benchmark, "normalized_source_bytes", lambda path=None: b"changed-generator")
    with pytest.raises(ValueError, match="expected-manifest"):
        benchmark.check_expected(benchmark.validate(bundle))

    changed_bundle = copy.deepcopy(bundle)
    changed_bundle["train"][0]["targets"]["text"] += "改変"
    with pytest.raises(ValueError, match="frozen v3 generator"):
        benchmark.validate(changed_bundle)

    manifest = benchmark.expected_manifest()
    manifest["counts"]["train"] += 1
    monkeypatch.setattr(benchmark, "expected_manifest", lambda: manifest)
    with pytest.raises(ValueError, match="expected-manifest"):
        benchmark.check_expected(report)


def test_export_is_byte_identical_and_exclusive(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    assert benchmark.write_bundle(first) == benchmark.write_bundle(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    with pytest.raises(FileExistsError):
        benchmark.write_bundle(first)
