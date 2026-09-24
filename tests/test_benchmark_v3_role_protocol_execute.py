"""Pre-execution binding tests; no Issue #48 learned run is started."""
from copy import deepcopy

import pytest

from norishio_lm import benchmark_v3_role_fixture as fixture
from norishio_lm.benchmark_v3_role_execute import _raw_rows, _validate_saved_row, _vocabulary
from norishio_lm import benchmark_v3_role_protocol as protocol
from norishio_lm.benchmark_v3_role_protocol import freeze_document
from norishio_lm.benchmark_v3_role_runner import prepare_role_row


def test_descriptor_binds_exact_static_fixture_and_paired_initial_rows():
    frozen = freeze_document()
    descriptor = frozen["descriptor"]
    assert descriptor["fixture"]["manifest_sha256"] == fixture.manifest_digest()
    assert descriptor["fixture"]["static_preflight"]["aliased_train_ceiling"]["participant"] == 216 / 384
    assert descriptor["fixture"]["static_preflight"]["aliased_train_ceiling"]["time"] == 240 / 384
    assert all(record["paired_rows_with_equal_initial_latent"] == 480
               and record["cloned_embedding_rows"] == 20
               and record["trainable_parameters"] == 32_120
               for record in descriptor["arms"]["paired_initialization"].values())
    assert len(descriptor["training"]["schedule_sha256"]) == 3
    assert descriptor["training"]["max_executor_wall_seconds"] == 7200


def test_prepared_row_rejects_a_source_gold_mismatch():
    row = fixture.build()["ROLE_ALIASED"]["train"][0]
    vocab = _vocabulary()
    valid = prepare_role_row(row, fixture.source_ids(row), vocab)
    assert valid.target_frame == row["targets"]["frame"]
    with pytest.raises(ValueError, match="prepared source tokens differ"):
        prepare_role_row(row, fixture.source_ids(row)[:-1], vocab)


def test_saved_raw_source_binding_is_checked_before_aggregation():
    row = fixture.build()["ROLE_DISJOINT"]["train"][0]
    vocab = _vocabulary()
    vectors = {factor: [1.0] + [0.0] * (len(vocab.values[factor]) - 1)
               for factor in vocab.values}
    head = {"row_index": 0, "target_frame": row["targets"]["frame"],
            "factor_probability_vectors": vectors,
            "factor_argmax": {factor: 0 for factor in vocab.values}}
    raw = _raw_rows("ROLE_DISJOINT", 7, "train", [row], [head], vocab)[0]
    _validate_saved_row(raw, row, arm="ROLE_DISJOINT", seed=7, split="train",
                        vocabulary=vocab)
    changed = deepcopy(raw)
    changed["count_signature_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding"):
        _validate_saved_row(changed, row, arm="ROLE_DISJOINT", seed=7,
                            split="train", vocabulary=vocab)


def test_raw_only_protocol_validation_never_recomputes_model_outputs(monkeypatch):
    def forbidden_model_preflight(_bundle):
        raise AssertionError("raw-only validation must not run a model")

    monkeypatch.setattr(protocol, "_paired_initial_preflight", forbidden_model_preflight)
    loaded = protocol.load_protocol(raw_only=True)
    assert loaded["descriptor_sha256"] == protocol.sha256(loaded["descriptor"])
