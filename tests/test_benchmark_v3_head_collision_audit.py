import json

from norishio_lm.benchmark_v3_head_collision_audit import (
    audit_bundle,
    audit_fixture,
    token_count_signature,
    write_json,
)


def row(text, *, support="train", labels=None):
    labels = labels or {"participant": "p0", "time": "t0", "event": "e0", "operator": "o0"}
    return {
        "inputs": {"context": "", "text": text},
        "targets": {"text": "target", "frame": labels},
        "metadata": {"support_class": support},
    }


def test_token_count_signature_is_order_insensitive_and_keeps_wrappers():
    assert token_count_signature([1, 8, 3, 8]) == ((1, 1), (3, 1), (8, 2))
    assert token_count_signature([8, 1, 8, 3]) == token_count_signature([1, 8, 3, 8])


def test_exact_collision_and_majority_ceiling_are_reported():
    labels0 = {"participant": "p0", "time": "t0", "event": "e0", "operator": "o0"}
    labels1 = {"participant": "p1", "time": "t0", "event": "e0", "operator": "o0"}
    # The source count signatures collide because the two JSON payloads have
    # the same byte counts, while the gold participant labels differ.
    report = audit_bundle({
        "train": [row("ab", labels=labels0), row("ba", labels=labels1), row("ab", labels=labels0)],
        "confirmation": [],
    })["scopes"]["train"]
    assert report["row_count"] == 3
    assert report["collision_group_count"] == 1
    assert report["collision_rows"] == 3
    assert report["per_factor_conflicting_gold_signature_counts"]["participant"] == 1
    assert report["per_factor_representation_ceiling"]["participant"] == 2 / 3
    assert report["per_factor_representation_ceiling_detail"]["participant"] == {
        "majority_label_rows": 2, "scheduled_rows": 3,
    }


def test_confirmation_is_grouped_by_all_and_support_stratum():
    bundle = {
        "train": [],
        "confirmation": [
            row("aa", support="unseen_pair"),
            row("aa", support="seen_pair/unseen_triple"),
        ],
    }
    scopes = audit_bundle(bundle)["scopes"]
    assert scopes["confirmation_all"]["row_count"] == 2
    assert scopes["confirmation_by_support"]["unseen_pair"]["row_count"] == 1
    assert scopes["confirmation_by_support"]["seen_pair/unseen_triple"]["row_count"] == 1


def test_deterministic_json_artifact(tmp_path):
    report = audit_bundle({"train": [row("ab")], "confirmation": []})
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_json(first, report)
    write_json(second, audit_bundle({"train": [row("ab")], "confirmation": []}))
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == report


def test_frozen_fixture_train_ceiling_is_below_strong_head_gate():
    report = audit_fixture()
    assert report["fixture_content_digest_sha256"] == (
        "b587c091e3499348c5e9bcfabff68ac3bb3bbd2c1df60edaac62833cf69cbed5"
    )
    train = report["scopes"]["train"]
    assert (train["row_count"], train["unique_signatures"], train["collision_rows"]) == (384, 140, 340)
    assert {factor: detail["majority_label_rows"] for factor, detail in
            train["per_factor_representation_ceiling_detail"].items()} == {
        "participant": 216, "time": 240, "event": 236, "operator": 262,
    }
    assert all(len(set(support.values())) == 1 for support in train["per_factor_gold_support"].values())
    assert all(value < 0.8 for value in train["per_factor_representation_ceiling"].values())
