from __future__ import annotations

import hashlib
import json

from norishio_lm.benchmark_v2_audit import (
    ARM_IDS,
    SEEDS,
    build_split_audit,
    export_split_audit,
    render_markdown,
)


def _evaluation(arm: str, seed: int, *, omit: tuple[str, str] | None = None) -> dict:
    unseen = {
        "rows": 192,
        "free_generation_exact": {"accuracy": 0.1},
        "generation_frame_exact": {"accuracy": 0.2},
        "triple_exact": {"accuracy": 0.3},
        "parse_coverage": 0.4,
    }
    seen = {
        "rows": 192,
        "free_generation_exact": {"accuracy": 0.5},
        "generation_frame_exact": {"accuracy": 0.6},
        "triple_exact": {"accuracy": 0.7},
        "parse_coverage": 0.8,
    }
    if omit is not None:
        del {"unseen_pair": unseen, "seen_pair": seen}[omit[0]][omit[1]]
    return {
        "arm": arm,
        "seed": seed,
        "result": {"train_support_groups": {"unseen_pair": unseen, "seen_pair": seen}},
    }


def _full_result() -> dict:
    return {
        "schema": "norishio.issue34.final-result.v1",
        "status": "complete",
        "evaluations": [_evaluation(arm, seed) for arm in ARM_IDS for seed in SEEDS],
    }


def test_build_split_audit_covers_all_evaluations_and_records_source_hash(tmp_path):
    source = tmp_path / "result.json"
    payload = json.dumps(_full_result(), separators=(",", ":")).encode("utf-8")
    source.write_bytes(payload)

    audit = build_split_audit(source)

    assert audit["schema"] == "norishio.issue34.split-audit.v1"
    assert audit["source"]["schema"] == "norishio.issue34.final-result.v1"
    assert audit["source"]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert audit["expected_evaluations"] == audit["observed_evaluations"] == 18
    assert audit["validation_issues"] == []
    assert len(audit["evaluations"]) == 18
    assert audit["evaluations"][0]["splits"]["unseen_pair"] == {
        "count": 192,
        "free_exact": 0.1,
        "frame_exact": 0.2,
        "triple_exact": 0.3,
        "parse_coverage": 0.4,
    }
    assert audit["evaluations"][0]["splits"]["seen_pair"]["free_exact"] == 0.5


def test_missing_source_value_is_null_with_reason_and_markdown_is_compact(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps({"schema": "norishio.issue34.final-result.v1", "status": "complete", "evaluations": [_evaluation("A_G0", 7, omit=("unseen_pair", "triple_exact"))]}),
        encoding="utf-8",
    )

    audit = build_split_audit(source)
    unseen = audit["evaluations"][0]["splits"]["unseen_pair"]
    assert unseen["triple_exact"] is None
    assert unseen["missing_reasons"]["triple_exact"] == "missing result field: triple_exact.accuracy"
    assert any(issue == "missing evaluation: A_G0/17" for issue in audit["validation_issues"])
    markdown = render_markdown(audit)
    assert "seen_pair` is the pair-known" in markdown
    assert "| A_G0 | 7 | unseen_pair | 192 | 0.100000 |" in markdown
    assert "| A_G0 | 7 | seen_pair | 192 | 0.500000 |" in markdown


def test_source_schema_and_status_drift_are_validation_issues(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(
        json.dumps({"schema": "other.schema.v1", "status": "failed", "evaluations": []}),
        encoding="utf-8",
    )

    audit = build_split_audit(source)

    assert any("source schema" in issue for issue in audit["validation_issues"])
    assert any("source status" in issue for issue in audit["validation_issues"])


def test_path_separator_variants_produce_identical_audits(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(_full_result()), encoding="utf-8")
    slash_path = str(source).replace("\\", "/")
    backslash_path = str(source).replace("/", "\\")

    slash_audit = build_split_audit(slash_path)
    backslash_audit = build_split_audit(backslash_path)

    assert slash_audit == backslash_audit
    assert slash_audit["source"]["path"] == slash_path


def test_export_writes_json_and_markdown_from_same_audit(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(_full_result()), encoding="utf-8")
    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"

    audit = export_split_audit(source, json_path, markdown_path)

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written == audit
    assert markdown_path.read_text(encoding="utf-8").count("| A_G0 | 7 |") == 2
