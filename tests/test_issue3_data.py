"""Dataset checks only; these are not model performance tests."""
import copy
import importlib.util
from pathlib import Path
import socket

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("issue3_toy_data", ROOT / "data/issue3/toy_corpus.py")
assert spec is not None and spec.loader is not None
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)


@pytest.fixture
def bundle():
    return toy.build()


def test_expected_counts_and_groups(bundle):
    report = toy.validate(bundle)
    assert report["counts"] == {"train": 450, "validation": 150, "test": 150, "diagnostic": 24}
    assert report["groups"] == {"train": 15, "validation": 5, "test": 5, "diagnostic": 10}


def test_every_contrast_and_variant_stays_in_one_situation(bundle):
    groups = {}
    for split in toy.SPLITS:
        for row in bundle[split]:
            groups.setdefault(row["group_id"], []).append(row)
    for rows in groups.values():
        assert len(rows) == 30
        assert len({r["split"] for r in rows}) == 1
        assert len({r["metadata"]["condition"] for r in rows}) == 10


def test_slot_values_seen_but_evaluation_combinations_held_out(bundle):
    train_frames = [r["targets"]["concept"] for r in bundle["train"]]
    people = {f["participant"] for f in train_frames}
    times = {f["time"] for f in train_frames}
    combinations = {(f["participant"], f["time"]) for f in train_frames}
    for split in ("validation", "test"):
        for row in bundle[split]:
            f = row["targets"]["concept"]
            assert f["participant"] in people and f["time"] in times
            assert (f["participant"], f["time"]) not in combinations


def test_negation_scope_is_not_collapsed(bundle):
    rows = {r["metadata"]["condition"]: r for r in bundle["train"][:30]}
    assert rows["meet_not_want"]["targets"]["concept"]["operators"] == ["NOT", "WANT"]
    assert rows["meet_want_not"]["targets"]["concept"]["operators"] == ["WANT", "NOT"]
    assert rows["separate_want_not"]["targets"]["concept"]["operators"] != rows["separate_plan_not"]["targets"]["concept"]["operators"]
    assert rows["meet_repeat_want"]["targets"]["concept"]["repeat_marked"]
    assert not rows["meet_want"]["targets"]["concept"]["repeat_marked"]


@pytest.mark.parametrize("field", ["targets", "metadata", "id", "group_id", "split"])
def test_gold_or_metadata_change_cannot_change_source_inputs(bundle, field):
    row = bundle["train"][0]
    changed = copy.deepcopy(row)
    changed[field] = {"leak": "DO_NOT_READ_THIS"}
    assert toy.model_inputs(row) == toy.model_inputs(changed)
    assert toy.source_ids(row) == toy.source_ids(changed)


def test_input_allowlist_rejects_extra_fields(bundle):
    row = copy.deepcopy(bundle["train"][0])
    row["inputs"]["gold_concept"] = "WANT"
    with pytest.raises(ValueError):
        toy.model_inputs(row)
    row["inputs"] = {"context": "", "text": 123}
    with pytest.raises(ValueError):
        toy.model_inputs(row)


@pytest.mark.parametrize("split", ["train", "validation", "test", "diagnostic"])
def test_teacher_forcing_has_shifted_target_and_masked_source(bundle, split):
    for row in bundle[split]:
        example = toy.teacher_forcing(row)
        source = toy.source_ids(row)
        answer = [*(b + toy.BYTE_OFFSET for b in row["targets"]["text"].encode("utf-8")), toy.EOS]
        assert example["source_ids"] == source
        assert example["input_ids"] == (source + answer)[:-1]
        assert example["labels"] == [-100] * (len(source) - 1) + answer
        assert example["concept_position"] == len(source) - 1
        assert max(example["input_ids"]) < toy.VOCAB_SIZE
        # At a supervised position k, the target byte is at k+1, not k.
        full = source + answer
        for k, label in enumerate(example["labels"]):
            assert label == (-100 if k < len(source) - 1 else full[k + 1])


def test_unlabeled_auxiliary_targets_keep_lm_target(bundle):
    row = bundle["train"][0]
    result = toy.supervised_targets(row, ("sense", "sememes", "concept"))
    assert result["text"] == row["targets"]["text"]
    assert all(result[k] is None for k in ("sense", "sememes", "concept"))
    assert all(row["targets"][k] is not None for k in ("sense", "sememes", "concept"))
    with pytest.raises(ValueError):
        toy.supervised_targets(row, "sense")
    with pytest.raises(ValueError):
        toy.supervised_targets(row, ["text"])


def test_same_utterance_different_context_can_have_different_references(bundle):
    rows = [r for r in bundle["diagnostic"] if r["inputs"]["text"] == "もういい。"]
    assert len(rows) == 3
    assert len({toy.canonical(r["inputs"]) for r in rows}) == 3
    assert len({r["targets"]["interpretation"] for r in rows}) == 3


def test_false_glyph_fixture_is_not_presented_as_etymological_fact(bundle):
    rows = [r for r in bundle["diagnostic"] if r["group_id"] == "diagnostic-glyph_adversarial"]
    assert len(rows) == 2
    assert toy.model_inputs(rows[0]) == toy.model_inputs(rows[1])
    assert rows[0]["targets"] == rows[1]["targets"]
    assert rows[0]["metadata"]["perturbation"]["kind"] == "intentionally_false_fixture"
    assert rows[1]["metadata"]["perturbation"] is None


def test_validator_rejects_sibling_group_leakage(bundle):
    row = copy.deepcopy(bundle["train"][0])
    row["id"] += "-moved"
    row["split"] = "test"
    bundle["test"].append(row)
    with pytest.raises(ValueError, match="group"):
        toy.validate(bundle)


def test_validator_rejects_duplicate_source(bundle):
    row = copy.deepcopy(bundle["train"][0])
    row["id"] += "-duplicate"
    bundle["train"].append(row)
    with pytest.raises(ValueError, match="source"):
        toy.validate(bundle)


def test_build_and_export_are_offline_deterministic_and_do_not_overwrite(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")
    monkeypatch.setattr(socket, "socket", forbidden)
    first, second = tmp_path / "first", tmp_path / "second"
    assert toy.write_bundle(first) == toy.write_bundle(second)
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()
    with pytest.raises(FileExistsError):
        toy.write_bundle(first)
