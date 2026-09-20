from __future__ import annotations

import copy

import pytest

from norishio_lm.benchmark_v3_h0_fixture import (
    CONFIRMATION_SUPPORTS,
    FIELDS,
    MANIFEST_PATH,
    SUPPORT_SEEN_UNSEEN_TRIPLE,
    SUPPORT_UNSEEN_PAIR,
    build,
    check_expected,
    expected_manifest,
    probe_identities,
    source_swap_pairs,
    train_support,
    validate,
)


def _frames(bundle, split):
    return [row for row in bundle[split] if row["metadata"]["variant"] == 0]


def test_fixture_is_deterministic_and_manifest_bound() -> None:
    first = build()
    second = build()
    assert first == second
    report = validate(first)
    check_expected(report)
    assert MANIFEST_PATH.is_file()
    assert report["content_digest_sha256"] == expected_manifest()["content_digest_sha256"]


def test_counts_variants_and_support_classes_are_frozen() -> None:
    bundle = build()
    report = validate(bundle)

    assert report["counts"] == {"train": 384, "confirmation": 96}
    assert report["groups"] == {"train": 192, "confirmation": 48}
    assert report["support_classes"] == {
        "train": {"train": 384},
        "confirmation": {
            SUPPORT_UNSEEN_PAIR: 48,
            SUPPORT_SEEN_UNSEEN_TRIPLE: 48,
        },
    }
    assert all(
        sum(row["group_id"] == group for split in bundle for row in bundle[split]) == 2
        for group in {row["group_id"] for split in bundle for row in bundle[split]}
    )


def test_support_semantics_and_train_atom_coverage() -> None:
    bundle = build()
    support = train_support(bundle)
    assert len(support["participant_time"]) == 24
    assert len(support["participant_time_event"]) == 48

    train_atoms = {field: set() for field in FIELDS}
    for row in _frames(bundle, "train"):
        for field in FIELDS:
            train_atoms[field].add(row["targets"]["frame"][field])
    assert train_atoms == {
        "participant": {f"P{index}" for index in range(6)},
        "time": {f"T{index}" for index in range(6)},
        "event": {f"E{index}" for index in range(4)},
        "operator": {f"O{index}" for index in range(4)},
    }

    for row in _frames(bundle, "confirmation"):
        frame = row["targets"]["frame"]
        pair = (frame["participant"], frame["time"])
        triple = (*pair, frame["event"])
        if row["metadata"]["support_class"] == SUPPORT_UNSEEN_PAIR:
            assert pair not in support["participant_time"]
            assert triple not in support["participant_time_event"]
        else:
            assert pair in support["participant_time"]
            assert triple not in support["participant_time_event"]


def test_source_and_target_surfaces_are_disjoint_from_prior_namespace() -> None:
    bundle = build()
    sources = {row["inputs"]["text"] for rows in bundle.values() for row in rows}
    targets = {row["targets"]["text"] for rows in bundle.values() for row in rows}
    assert sources.isdisjoint(targets)
    assert all(value.startswith("h0-source-") for value in sources)
    assert all(value.startswith("h0-target-") for value in targets)
    assert expected_manifest()["checks"]["prior_v2_v3_evaluation_surfaces_disjoint"] is True


def test_every_factor_and_support_has_a_deterministic_source_swap_pair() -> None:
    bundle = build()
    pairs = source_swap_pairs(bundle)
    assert set(pairs) == set(CONFIRMATION_SUPPORTS)
    for support in CONFIRMATION_SUPPORTS:
        assert set(pairs[support]) == set(FIELDS)
        for factor, (left_id, right_id) in pairs[support].items():
            assert left_id != right_id
            left = next(row for row in bundle["confirmation"] if row["id"] == left_id)
            right = next(row for row in bundle["confirmation"] if row["id"] == right_id)
            left_frame, right_frame = left["targets"]["frame"], right["targets"]["frame"]
            assert left["metadata"]["support_class"] == support
            assert right["metadata"]["support_class"] == support
            assert [name for name in FIELDS if left_frame[name] != right_frame[name]] == [factor]
    assert source_swap_pairs(bundle) == pairs


def test_eight_probe_identities_are_deterministic_and_manifest_bound() -> None:
    bundle = build()
    probes = probe_identities(bundle)
    assert set(probes) == set(CONFIRMATION_SUPPORTS)
    assert all(set(probes[support]) == set(FIELDS) for support in CONFIRMATION_SUPPORTS)
    assert len({row_id for values in probes.values() for row_id in values.values()}) == 8
    assert probes == probe_identities(bundle)
    assert probes == expected_manifest()["selection"]["probe_identities"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle["train"][0]["inputs"].__setitem__("text", "tampered"),
        lambda bundle: bundle["confirmation"][0]["metadata"].__setitem__(
            "support_class", SUPPORT_SEEN_UNSEEN_TRIPLE
        ),
        lambda bundle: bundle["confirmation"][0]["id"].__setitem__(0, "x")
        if isinstance(bundle["confirmation"][0]["id"], list)
        else bundle["confirmation"][0].__setitem__("id", "reseated"),
    ],
)
def test_tampered_rows_are_rejected(mutate) -> None:
    bundle = copy.deepcopy(build())
    mutate(bundle)
    with pytest.raises(ValueError, match="bundle differs|support|source"):
        validate(bundle)


def test_confirmation_access_requires_exact_manifest_digest() -> None:
    from norishio_lm.benchmark_v3_h0_fixture import confirmation_rows

    with pytest.raises(ValueError, match="manifest digest"):
        confirmation_rows(manifest_digest="0" * 64)
