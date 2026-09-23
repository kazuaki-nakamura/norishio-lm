from __future__ import annotations

import copy

import pytest

from norishio_lm.benchmark_v3_head_fixture import (
    CONFIRMATION_SUPPORTS,
    FIELDS,
    MANIFEST_PATH,
    SUPPORT_SEEN_UNSEEN_TRIPLE,
    SUPPORT_UNSEEN_PAIR,
    build,
    canonical,
    check_expected,
    expected_manifest,
    load,
    manifest,
    manifest_digest,
    parse_target_text,
    source_ids,
    spec_data,
    train_support,
    validate,
)


def _frames(bundle: dict[str, list[dict]], split: str) -> list[dict]:
    return [row for row in bundle[split] if row["metadata"]["variant"] == 0]


def test_fixture_is_deterministic_and_manifest_bound() -> None:
    first = build()
    second = build()
    assert first == second
    report = validate(first)
    check_expected(report)
    assert MANIFEST_PATH.is_file()
    assert report["content_digest_sha256"] == expected_manifest()["content_digest_sha256"]
    assert manifest_digest(first) == report["content_digest_sha256"]
    assert load() == first
    assert manifest(first) == report


def test_counts_graph_and_confirmation_supports_are_frozen() -> None:
    bundle = build()
    report = validate(bundle)
    assert report["counts"] == {"train": 384, "confirmation": 96}
    assert report["groups"] == {"train": 192, "confirmation": 48}
    assert report["support_classes"] == {
        "train": {"train": 384},
        "confirmation": {SUPPORT_UNSEEN_PAIR: 48, SUPPORT_SEEN_UNSEEN_TRIPLE: 48},
    }
    assert report["train_pair_offset_counts"] == {"2": 48, "3": 48, "4": 48, "5": 48}
    assert set(report["support_classes"]["confirmation"]) == set(CONFIRMATION_SUPPORTS)


def test_every_atom_is_train_supported_and_predicates_are_counterbalanced() -> None:
    bundle = build()
    spec = spec_data()
    frames = [row["targets"]["frame"] for row in _frames(bundle, "train")]
    assert {field: {frame[field] for frame in frames} for field in FIELDS} == {
        "participant": {value["id"] for value in spec["participants"]},
        "time": {value["id"] for value in spec["times"]},
        "event": {value["id"] for value in spec["events"]},
        "operator": set(spec["operators"]),
    }
    # Each participant/time sees each event eight times, and each event/operator
    # predicate twice.  Thus no single predicate identifies a participant/time.
    report = validate(bundle)
    assert report["balance"] == {
        "participant_event_count": 8,
        "time_event_count": 8,
        "participant_predicate_count": 2,
        "time_predicate_count": 2,
        "all_equal": True,
    }
    assert len(train_support(bundle)["participant_time"]) == 24
    assert len(train_support(bundle)["participant_time_event"]) == 48


def test_confirmation_contains_unseen_pairs_and_seen_pairs_with_unseen_triples() -> None:
    bundle = build()
    support = train_support(bundle)
    classes = {support: [] for support in CONFIRMATION_SUPPORTS}
    for row in _frames(bundle, "confirmation"):
        frame = row["targets"]["frame"]
        pair = (frame["participant"], frame["time"])
        triple = (*pair, frame["event"])
        support_class = row["metadata"]["support_class"]
        classes[support_class].append((pair, triple))
        if support_class == SUPPORT_UNSEEN_PAIR:
            assert pair not in support["participant_time"]
        else:
            assert pair in support["participant_time"]
            assert triple not in support["participant_time_event"]
    assert {len(values) for values in classes.values()} == {24}


def test_variants_are_grouped_and_namespaces_are_disjoint_from_prior() -> None:
    bundle = build()
    groups: dict[str, list[dict]] = {}
    sources = set()
    targets = set()
    for rows in bundle.values():
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
            sources.add(row["inputs"]["text"])
            targets.add(row["targets"]["text"])
    assert all(len(rows) == 2 and {row["metadata"]["variant"] for row in rows} == {0, 1} for rows in groups.values())
    assert sources.isdisjoint(targets)
    assert all(value.startswith("hl-source-") for value in sources)
    assert all(value.startswith("hl-target-") for value in targets)


def test_target_parser_and_source_encoding_are_bound_to_declared_fields() -> None:
    bundle = build()
    row = bundle["train"][0]
    assert parse_target_text(row["targets"]["text"]) == row["targets"]["frame"]
    assert source_ids(row)[0] == 1 and source_ids(row)[-1] == 3
    altered = copy.deepcopy(row)
    altered["targets"]["frame"]["participant"] = "LEAK"
    assert source_ids(row) == source_ids(altered)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda bundle: bundle["train"][0]["id"].__class__ and bundle["train"][0].__setitem__("id", "tampered"),
        lambda bundle: bundle["confirmation"][0]["metadata"].__setitem__("support_class", SUPPORT_SEEN_UNSEEN_TRIPLE),
        lambda bundle: bundle["train"][0]["targets"].__setitem__("text", "tampered-target"),
    ],
)
def test_tampered_rows_are_rejected(mutate) -> None:
    bundle = copy.deepcopy(build())
    mutate(bundle)
    with pytest.raises(ValueError, match="bundle differs|support|target"):
        validate(bundle)


def test_canonical_manifest_is_stable() -> None:
    assert canonical(expected_manifest()) == canonical(expected_manifest())
