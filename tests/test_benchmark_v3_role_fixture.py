"""Pre-training regression guards for the paired authored role fixture."""
from collections import Counter

import pytest

from norishio_lm import benchmark_v3_role_fixture as fixture


def test_paired_rows_have_exact_new_split_and_four_changed_value_bytes():
    bundle = fixture.build()
    assert {arm: {split: len(bundle[arm][split]) for split in fixture.SPLITS}
            for arm in fixture.ARMS} == {
        arm: {"train": 384, "confirmation": 96} for arm in fixture.ARMS
    }
    assert fixture.paired_correspondence(bundle)["only_four_value_bytes_changed"]
    for arm in fixture.ARMS:
        train = bundle[arm]["train"]
        confirmation = bundle[arm]["confirmation"]
        assert {(int(row["targets"]["frame"]["time"][1:]) -
                 int(row["targets"]["frame"]["participant"][1:])) % 6
                for row in train} == {2, 3, 4, 5}
        by_support = {support: [row for row in confirmation
                                if row["metadata"]["support_class"] == support]
                      for support in fixture.CONFIRMATION_SUPPORTS}
        assert {support: len(rows) for support, rows in by_support.items()} == {
            "unseen_pair": 48, "seen_pair/unseen_triple": 48,
        }
        for support, offset, events, operators in (
            ("unseen_pair", 1, {"E1", "E2"}, {"O0", "O1"}),
            ("seen_pair/unseen_triple", 3, {"E0", "E3"}, {"O2", "O3"}),
        ):
            rows = by_support[support]
            assert {(int(row["targets"]["frame"]["time"][1:]) -
                     int(row["targets"]["frame"]["participant"][1:])) % 6
                    for row in rows} == {offset}
            assert {row["targets"]["frame"]["event"] for row in rows} == events
            assert {row["targets"]["frame"]["operator"] for row in rows} == operators


def test_complete_source_signatures_are_orderless_and_rational():
    row = fixture.build()["ROLE_ALIASED"]["train"][0]
    ids = fixture.source_ids(row)
    assert (ids[0], ids[-1]) == (fixture.BOS, fixture.SEP)
    assert fixture.count_signature(row) == tuple(sorted(Counter(ids).items()))
    frequency = fixture.normalized_frequency_signature(row)
    assert all(isinstance(numerator, int) and isinstance(denominator, int)
               for _, numerator, denominator in frequency)
    assert all(numerator / denominator == count / len(ids)
               for (_, count), (_, numerator, denominator) in zip(
                   fixture.count_signature(row), frequency, strict=True))


def test_static_stop_gate_measures_real_aliasing_and_disjoint_identity():
    bundle = fixture.build()
    audit = fixture.static_audit_report(bundle)
    a_train = audit["ROLE_ALIASED"]["train"]
    b_train = audit["ROLE_DISJOINT"]["train"]
    assert (a_train["row_count"], a_train["unique_count_signatures"],
            a_train["collision_rows"]) == (384, 140, 340)
    assert {factor: entry["majority_rows"] for factor, entry in
            a_train["by_factor"].items()} == {
        "participant": 216, "time": 240, "event": 236, "operator": 262,
    }
    assert b_train["unique_count_signatures"] == 384
    assert all(entry["representation_ceiling"] == 1
               for entry in b_train["by_factor"].values())
    for arm in fixture.ARMS:
        assert audit[arm]["union"]["row_count"] == 480
        assert audit[arm]["confirmation_all"]["row_count"] == 96
        assert {support: audit[arm]["confirmation_by_support"][support]["row_count"]
                for support in fixture.CONFIRMATION_SUPPORTS} == {
            "unseen_pair": 48, "seen_pair/unseen_triple": 48,
        }
    assert fixture.preflight(bundle)["aliased_train_ceiling"]["participant"] < 0.8
    tampered = fixture.build()
    tampered["ROLE_DISJOINT"]["train"][0]["inputs"] = dict(
        tampered["ROLE_DISJOINT"]["train"][2]["inputs"])
    with pytest.raises(ValueError):
        fixture.preflight(tampered)


def test_manifest_is_deterministic_and_binds_four_row_files():
    report = fixture.manifest()
    assert len(report["row_files_sha256"]) == 4
    assert fixture.manifest_digest() == fixture.manifest_digest()
