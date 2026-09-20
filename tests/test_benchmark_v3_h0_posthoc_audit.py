from __future__ import annotations

from copy import deepcopy
import json

import pytest

from norishio_lm.benchmark_v3_h0_posthoc_audit import (
    ARMS,
    AUDIT_PATH,
    CONTROL_TYPES,
    FACTORS,
    RESULT_ROOT,
    SEEDS,
    SUPPORT_GROUPS,
    build_head_factor_audit,
    load_saved_runs,
    regenerate_head_factor_audit,
    render_head_factor_audit,
)


def test_tracked_artifact_is_byte_identical_to_raw_regeneration() -> None:
    expected = render_head_factor_audit(regenerate_head_factor_audit())
    assert AUDIT_PATH.read_bytes() == expected


def test_atomic_support_and_control_denominators_and_order() -> None:
    audit = regenerate_head_factor_audit()
    assert audit["scope"]["run_order"] == [f"{arm} seed{seed}" for seed in SEEDS for arm in ARMS]
    assert audit["scope"]["factors"] == list(FACTORS)
    assert audit["scope"]["support_groups"] == list(SUPPORT_GROUPS)
    for arm in ARMS:
        for seed in SEEDS:
            for factor in FACTORS:
                metric = audit["atomic_head"][arm][str(seed)][factor]
                assert metric["count"] == metric["denominator"] == 96
                assert metric["rate"] == metric["correct"] / 96
        for factor in FACTORS:
            for support_group in SUPPORT_GROUPS:
                metric = audit["support_aggregate"][arm][factor][support_group]
                assert metric["count"] == metric["denominator"] == 144
            for control in CONTROL_TYPES:
                metrics = audit["controls"][arm][factor][control]
                for metric in metrics.values():
                    assert metric["count"] == metric["denominator"] == 6


def test_fixture_and_descriptor_bindings_are_required() -> None:
    runs = load_saved_runs()
    changed = deepcopy(runs)
    changed[0]["descriptor_sha256"] = "tampered"
    with pytest.raises(ValueError, match="descriptor binding"):
        build_head_factor_audit(changed)

    changed = deepcopy(runs)
    changed[0]["fixture_content_digest_sha256"] = "tampered"
    with pytest.raises(ValueError, match="fixture binding"):
        build_head_factor_audit(changed)


def test_raw_row_order_is_checked_before_reaggregation() -> None:
    runs = load_saved_runs()
    changed = deepcopy(runs)
    rows = changed[0]["ordinary_confirmation"]["raw_confirmation_rows"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="raw row order"):
        build_head_factor_audit(changed)


def test_absent_saved_control_is_reported_unavailable() -> None:
    runs = load_saved_runs()
    changed = deepcopy(runs)
    for run in changed:
        run["interventions"]["control_records"] = [
            row
            for row in run["interventions"]["control_records"]
            if not (row["control_type"] == "alternate_class_donor_soft" and row["target_factor"] == "event")
        ]
    audit = build_head_factor_audit(changed)
    metrics = audit["controls"]["H1L1"]["event"]["alternate_class_donor_soft"]
    assert metrics["nontrivial_joint_success"]["available"] is False
    assert metrics["missing"]["available"] is False


def test_artifact_is_valid_json() -> None:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == "norishio.issue44.h0-head-factor-posthoc-audit.v1"
    assert payload["decision_policy"]["changes_frozen_decision"] is False
