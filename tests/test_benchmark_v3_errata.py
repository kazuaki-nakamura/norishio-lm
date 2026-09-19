from __future__ import annotations

import hashlib
import json

import pytest

from norishio_lm.benchmark_v3_errata import ARM_IDS, SEEDS, build_protocol_errata_audit


def _metrics(frame: float, *, target_changed: int = 0) -> dict:
    examples = [
        {"factor": factor, "one_hot": [1.0, *([0.0] * (5 if factor in {"participant", "time"} else 3))]}
        for factor in ("participant", "time", "event", "operator")
    ]
    return {
        "all": {
            "generation_frame_exact": {"accuracy": frame},
            "triple_exact": {"accuracy": frame / 2},
            "pair_exact": {"accuracy": frame / 3},
            "atomic_balanced_accuracy": {
                "participant": frame,
                "time": frame,
                "event": frame,
                "operator": frame,
            },
            "exact_target_text": {"accuracy": frame / 4},
        },
        "intermediate_probability_intervention": {
            "rows": 4,
            "examples": examples,
            "by_factor": {
                factor: {
                    "target_changed": {
                        "correct": target_changed if factor == "participant" else 0,
                        "denominator": 1,
                    }
                }
                for factor in ("participant", "time", "event", "operator")
            },
        },
    }


def _write_artifacts(tmp_path):
    development = tmp_path / "development"
    development.mkdir()
    final_evaluations = []
    arm_values = {arm: (index + 1) / 10 for index, arm in enumerate(ARM_IDS)}
    for arm in ARM_IDS:
        for seed in SEEDS:
            metrics = _metrics(arm_values[arm])
            final_evaluations.append({"arm": arm, "seed": seed, "result": metrics})
            (development / f"{arm}-{seed}.json").write_text(json.dumps({
                "schema": "norishio.issue36.development-run.v1",
                "arm": arm,
                "seed": seed,
                "metrics": metrics,
            }), encoding="utf-8")
    final = tmp_path / "result.json"
    payload = json.dumps({
        "schema": "norishio.issue36.final-result.v1",
        "status": "complete",
        "evaluations": final_evaluations,
    }, separators=(",", ":")).encode("utf-8")
    final.write_bytes(payload)
    return final, development, hashlib.sha256(payload).hexdigest()


def test_audit_uses_generation_frame_contract_and_classifies_omitted_vectors(tmp_path):
    final, development, digest = _write_artifacts(tmp_path)

    audit = build_protocol_errata_audit(
        final, development, expected_final_sha256=digest,
    )

    assert audit["validation_issues"] == []
    assert audit["sources"]["final"]["sha256_matches_expected"] is True
    assert audit["historical_selection_implementation"]["primary"] == (
        "all.free_generation_exact.accuracy"
    )
    assert audit["selection_contract"]["primary"] == "all.generation_frame_exact.accuracy"
    assert audit["final"]["ranking"] == list(reversed(ARM_IDS))
    assert audit["development"]["ranking"] == list(reversed(ARM_IDS))
    intervention = audit["final"]["intervention_audit"]
    assert intervention["baseline_argmax_classification"] == "unavailable"
    assert "omits baseline_probabilities" in intervention["reason"]
    assert intervention["main_arms"] == {"runs": 12, "rows": 48, "target_changed": 0}
    assert intervention["all_retained_one_hots_select_class_zero"] is True


def test_audit_reconstructs_three_seed_means_and_factorial_effects(tmp_path):
    final, development, digest = _write_artifacts(tmp_path)

    audit = build_protocol_errata_audit(final, development, expected_final_sha256=digest)
    primary = "all.generation_frame_exact.accuracy"

    assert audit["final"]["arms"]["H0L0"]["three_seed_mean"][primary] == pytest.approx(0.1)
    effects = audit["final"]["factorial_effects"][primary]
    assert effects == pytest.approx({
        "activation_main_effect": 0.2,
        "locality_main_effect": 0.1,
        "interaction": 0.0,
    })


def test_final_hash_mismatch_is_visible_without_mutating_source(tmp_path):
    final, development, _ = _write_artifacts(tmp_path)
    before = final.read_bytes()

    audit = build_protocol_errata_audit(
        final, development, expected_final_sha256="0" * 64,
    )

    assert audit["sources"]["final"]["sha256_matches_expected"] is False
    assert any("SHA-256 mismatch" in issue for issue in audit["validation_issues"])
    assert final.read_bytes() == before


def test_incomplete_intervention_evidence_is_a_validation_issue(tmp_path):
    final, development, _ = _write_artifacts(tmp_path)
    value = json.loads(final.read_text(encoding="utf-8"))
    value["evaluations"][0]["result"]["intermediate_probability_intervention"][
        "examples"
    ].pop()
    final.write_text(json.dumps(value), encoding="utf-8")
    digest = hashlib.sha256(final.read_bytes()).hexdigest()

    audit = build_protocol_errata_audit(
        final, development, expected_final_sha256=digest,
    )

    assert any(
        "must contain exactly four examples" in issue
        for issue in audit["final"]["validation_issues"]
    )
    assert any(
        issue.startswith("final:") and "must contain exactly four examples" in issue
        for issue in audit["validation_issues"]
    )
    intervention = audit["final"]["intervention_audit"]
    assert intervention["main_arms"]["rows"] == 44
    assert intervention["all_retained_one_hots_select_class_zero"] is None
