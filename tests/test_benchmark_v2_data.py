"""Protocol tests only; these do not measure model performance."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import socket

import pytest

from norishio_lm.benchmark_v2_metrics import score_benchmark_v2

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location(
    "benchmark_v2", ROOT / "data/benchmark_v2/benchmark.py"
)
assert module_spec is not None and module_spec.loader is not None
benchmark = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(benchmark)


@pytest.fixture
def bundle():
    return benchmark.build()


def frames(rows):
    return [row["targets"]["frame"] for row in rows]


def test_expected_manifest_and_balanced_counts(bundle):
    report = benchmark.validate(bundle)
    benchmark.check_expected(report)
    assert report["counts"] == {split: 384 for split in benchmark.SPLITS}
    assert report["groups"] == {split: 192 for split in benchmark.SPLITS}
    assert report["support_classes"] == {
        "train": {"train": 384},
        "diagnostic-validation": {"unseen_triple": 192, "unseen_pair": 192},
        "final-holdout": {"unseen_triple": 192, "unseen_pair": 192},
    }


def test_atomic_coverage_and_required_holdouts(bundle):
    train = frames(bundle["train"])
    pair_support = {(row["participant"], row["time"]) for row in train}
    triple_support = {(row["participant"], row["time"], row["event"]) for row in train}
    spec = benchmark.spec_data()
    expected_atoms = {
        "participant": {value["id"] for value in spec["participants"]},
        "time": {value["id"] for value in spec["times"]},
        "event": {value["id"] for value in spec["events"]},
        "operator": set(spec["operators"]),
    }
    assert {field: {row[field] for row in train} for field in benchmark.FIELDS} == expected_atoms
    for split in ("diagnostic-validation", "final-holdout"):
        kinds = set()
        for row in bundle[split]:
            frame = row["targets"]["frame"]
            pair = (frame["participant"], frame["time"])
            triple = (*pair, frame["event"])
            assert triple not in triple_support
            expected = "unseen_pair" if pair not in pair_support else "unseen_triple"
            assert row["metadata"]["support_class"] == expected
            kinds.add(expected)
        assert kinds == {"unseen_pair", "unseen_triple"}


def test_variants_stay_together_and_all_operators_cover_each_triple(bundle):
    groups = {}
    triples = {}
    for split, rows in bundle.items():
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
            frame = row["targets"]["frame"]
            key = (split, frame["participant"], frame["time"], frame["event"])
            triples.setdefault(key, set()).add(frame["operator"])
    assert all(len(rows) == 2 and len({row["split"] for row in rows}) == 1
               for rows in groups.values())
    assert all(values == set(benchmark.spec_data()["operators"]) for values in triples.values())


@pytest.mark.parametrize("field", ["targets", "metadata", "id", "group_id", "split"])
def test_hidden_fields_cannot_change_source_encoding(bundle, field):
    row = bundle["train"][0]
    changed = copy.deepcopy(row)
    changed[field] = {"hidden": "MUST_NOT_ENTER_SOURCE"}
    assert benchmark.model_inputs(row) == benchmark.model_inputs(changed)
    assert benchmark.source_ids(row) == benchmark.source_ids(changed)


def test_input_allowlist_and_teacher_forcing_boundary(bundle):
    row = copy.deepcopy(bundle["train"][0])
    row["inputs"]["gold_frame"] = {"participant": "LEAK"}
    with pytest.raises(ValueError, match="only context and text"):
        benchmark.model_inputs(row)
    example = benchmark.teacher_forcing(bundle["train"][0])
    source = benchmark.source_ids(bundle["train"][0])
    assert example["labels"][: len(source) - 1] == [-100] * (len(source) - 1)
    assert example["input_ids"][: len(source)] == source
    assert len(example["input_ids"]) == len(example["labels"])


def test_frozen_target_parser_is_strict_and_recovers_all_factors(bundle):
    for split in benchmark.SPLITS:
        for row in bundle[split]:
            assert benchmark.parse_target_text(row["targets"]["text"]) == row["targets"]["frame"]
    target = bundle["train"][0]["targets"]["text"]
    assert benchmark.parse_target_text(target + "余分") is None


def test_common_scorer_accepts_frozen_rows_and_train_support(bundle):
    rows = []
    for frozen in bundle["diagnostic-validation"][:8]:
        row = copy.deepcopy(frozen)
        row["generation"] = {"text": row["targets"]["text"], "ended_eos": True,
                             "valid_utf8": True, "invalid_special_tokens": []}
        rows.append(row)
    report = score_benchmark_v2(
        rows, benchmark.parse_target_text, train_support=benchmark.train_support(bundle)
    )
    assert report["all"]["generation_frame_exact"]["accuracy"] == 1
    assert report["all"]["target_text_exact"]["accuracy"] == 1
    assert report["teacher_forced_bytes"] is None


def test_factor_shuffle_plan_is_frozen_and_has_no_fixed_points():
    first = benchmark.factor_shuffle_plan()
    second = benchmark.factor_shuffle_plan()
    assert first == second == benchmark.expected_manifest()["factor_shuffle_plan"]
    for mapping in first.values():
        assert set(mapping) == set(mapping.values())
        assert all(source != target for source, target in mapping.items())


def test_final_requires_explicit_flag_and_matching_manifest(bundle):
    report = benchmark.validate(bundle)
    assert len(benchmark.evaluation_rows(bundle, "diagnostic-validation")) == 384
    with pytest.raises(PermissionError, match="explicit"):
        benchmark.evaluation_rows(bundle, "final-holdout")
    with pytest.raises(ValueError, match="digest"):
        benchmark.evaluation_rows(bundle, "final-holdout", evaluate_final=True,
                                  manifest_digest="wrong")
    rows = benchmark.evaluation_rows(
        bundle, "final-holdout", evaluate_final=True,
        manifest_digest=report["content_digest_sha256"],
    )
    assert rows == bundle["final-holdout"] and rows is not bundle["final-holdout"]


def test_validator_rejects_any_post_freeze_change(bundle):
    changed = copy.deepcopy(bundle)
    changed["train"][0]["targets"]["text"] += "改変"
    with pytest.raises(ValueError, match="frozen generator"):
        benchmark.validate(changed)


def test_hashes_are_line_ending_independent_and_custom_spec_is_attributed(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"one\r\ntwo\r\n")
    assert benchmark.normalized_source_bytes(source) == b"one\ntwo\n"
    spec = benchmark.spec_data()
    spec["split_seed"] += 1
    custom = benchmark.validate(benchmark.build(spec), spec)
    default = benchmark.validate(benchmark.build())
    assert custom["spec_sha256"] != default["spec_sha256"]


def test_offline_export_is_deterministic_and_exclusive(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", forbidden)
    first, second = tmp_path / "first", tmp_path / "second"
    assert benchmark.write_bundle(first) == benchmark.write_bundle(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    with pytest.raises(FileExistsError):
        benchmark.write_bundle(first)
