"""Deterministic paired source-role fixture and pre-training static audit.

Only authored data and integer token signatures are used here. No model,
checkpoint, training, or inference is loaded by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import gcd
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "data" / "benchmark_v3_role_identifiability" / "spec.json"
MANIFEST_PATH = SPEC_PATH.with_name("expected-manifest.json")
ARMS = ("ROLE_ALIASED", "ROLE_DISJOINT")
SPLITS = ("train", "confirmation")
FACTORS = ("participant", "time", "event", "operator")
CONFIRMATION_SUPPORTS = ("unseen_pair", "seen_pair/unseen_triple")
BOS, SEP, BYTE_OFFSET = 1, 3, 4
SCHEMA = "norishio.issue48.role-fixture.v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def spec_data() -> dict[str, Any]:
    value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    validate_spec(value)
    return value


def validate_spec(spec: Mapping[str, Any]) -> None:
    if (spec.get("version") != "norishio-benchmark-v3-role-identifiability.v1"
            or spec.get("roles") != list(ARMS)
            or len(spec.get("participants", [])) != 6
            or len(spec.get("times", [])) != 6
            or len(spec.get("events", [])) != 4
            or spec.get("operators") != [f"O{i}" for i in range(4)]):
        raise ValueError("role fixture vocabulary or version changed")
    expected_selection = {
        "train_pair_offsets": [2, 3, 4, 5],
        "unseen_pair_offsets": [1], "unused_pair_offsets": [0],
        "train_events_by_offset": {"2": ["E0", "E1"], "3": ["E1", "E2"],
                                   "4": ["E2", "E3"], "5": ["E3", "E0"]},
        "unseen_pair_events": ["E1", "E2"],
        "unseen_pair_operators": ["O0", "O1"],
        "seen_pair_offset": 3,
        "seen_pair_events": ["E0", "E3"],
        "seen_pair_operators": ["O2", "O3"],
        "variant_count": 2,
    }
    if spec.get("selection") != expected_selection:
        raise ValueError("role fixture split or confirmation plan changed")
    expected_codebooks = {
        "ROLE_ALIASED": {"participant": list("012345"), "time": list("012345"),
                         "event": list("0123"), "operator": list("0123")},
        "ROLE_DISJOINT": {"participant": list("ABCDEF"), "time": list("GHIJKL"),
                          "event": list("MNOP"), "operator": list("QRST")},
    }
    if spec.get("codebooks") != expected_codebooks:
        raise ValueError("source role codebook changed")
    for kind in ("source", "target"):
        templates = [f"ri-{kind}-{variant}::ri-participant-{{p}}|ri-time-{{t}}|ri-predicate-e{{e}}-o{{o}}"
                     for variant in ("a", "b")]
        if spec.get(f"{kind}_templates") != templates:
            raise ValueError(f"{kind} surface templates changed")
    for factor, prefix, count in (("participants", "P", 6), ("times", "T", 6)):
        entries = spec[factor]
        if [item.get("id") for item in entries] != [f"{prefix}{i}" for i in range(count)]:
            raise ValueError(f"{factor} order changed")
    if [item.get("id") for item in spec["events"]] != [f"E{i}" for i in range(4)]:
        raise ValueError("event order changed")
    authorship = spec.get("authorship")
    if (not isinstance(authorship, dict) or authorship.get("human_verified") is not False
            or authorship.get("private_data_used") is not False):
        raise ValueError("fixture provenance changed")


def _pair_plan(spec: Mapping[str, Any]):
    selection = spec["selection"]
    for participant in range(6):
        for time in range(6):
            offset = (time - participant) % 6
            if offset in selection["train_pair_offsets"]:
                yield (participant, time, selection["train_events_by_offset"][str(offset)],
                       spec["operators"], "train", "train")
                if offset == selection["seen_pair_offset"]:
                    yield (participant, time, selection["seen_pair_events"],
                           selection["seen_pair_operators"],
                           "confirmation", "seen_pair/unseen_triple")
            elif offset in selection["unseen_pair_offsets"]:
                yield (participant, time, selection["unseen_pair_events"],
                       selection["unseen_pair_operators"],
                       "confirmation", "unseen_pair")


def build(spec: Mapping[str, Any] | None = None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    output: dict[str, dict[str, list[dict[str, Any]]]] = {
        arm: {split: [] for split in SPLITS} for arm in ARMS
    }
    for arm in ARMS:
        codebook = current["codebooks"][arm]
        for pi, ti, events, operators, split, support in _pair_plan(current):
            for event in events:
                ei = int(event[1:])
                for operator in operators:
                    oi = int(operator[1:])
                    frame = {"participant": f"P{pi}", "time": f"T{ti}",
                             "event": event, "operator": operator}
                    group_id = f"ri-{support.replace('/', '-')}-p{pi}-t{ti}-e{ei}-o{oi}"
                    coded = {"p": codebook["participant"][pi], "t": codebook["time"][ti],
                             "e": codebook["event"][ei], "o": codebook["operator"][oi]}
                    gold = {"p": str(pi), "t": str(ti), "e": str(ei), "o": str(oi)}
                    for variant, (source_template, target_template) in enumerate(zip(
                            current["source_templates"], current["target_templates"], strict=True)):
                        output[arm][split].append({
                            "id": f"{group_id}-v{variant}", "group_id": group_id,
                            "split": split,
                            "inputs": {"context": "", "text": source_template.format(**coded)},
                            "targets": {"text": target_template.format(**gold),
                                        "frame": deepcopy(frame)},
                            "metadata": {"variant": variant, "support_class": support,
                                         "generator_seed": current["generator_seed"],
                                         "provenance": deepcopy(current["authorship"])},
                        })
    return output


def model_inputs(row: Mapping[str, Any]) -> dict[str, str]:
    inputs = row.get("inputs")
    if (not isinstance(inputs, Mapping) or set(inputs) != {"context", "text"}
            or any(not isinstance(inputs[field], str) for field in ("context", "text"))):
        raise ValueError("model inputs must contain only context and text")
    return {"context": inputs["context"], "text": inputs["text"]}


def source_ids(row: Mapping[str, Any]) -> list[int]:
    raw = canonical(model_inputs(row)).encode("utf-8")
    return [BOS, *(byte + BYTE_OFFSET for byte in raw), SEP]


def count_signature(row: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(Counter(source_ids(row)).items()))


def normalized_frequency_signature(row: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    counts = count_signature(row)
    length = sum(count for _, count in counts)
    common = length
    for _, count in counts:
        common = gcd(common, count)
    return tuple((token, count // common, length // common) for token, count in counts)


def signature_digest(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical(count_signature(row)).encode("utf-8"))


def _audit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    signatures: dict[tuple[tuple[int, int], ...], list[Mapping[str, Any]]] = defaultdict(list)
    frequencies: dict[tuple[tuple[int, int, int], ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        signatures[count_signature(row)].append(row)
        frequencies[normalized_frequency_signature(row)].append(row)
    if sorted(sorted(row["id"] for row in group) for group in signatures.values()) != sorted(
            sorted(row["id"] for row in group) for group in frequencies.values()):
        raise ValueError("count and rational-frequency partitions differ")
    scheduled = len(rows)
    by_factor: dict[str, Any] = {}
    for factor in FACTORS:
        majority = 0
        conflict_groups = 0
        conflicting_rows = 0
        support: Counter[str] = Counter()
        for group in signatures.values():
            labels = Counter(str(row["targets"]["frame"][factor]) for row in group)
            support.update(labels)
            majority += max(labels.values())
            if len(labels) > 1:
                conflict_groups += 1
                conflicting_rows += len(group)
        by_factor[factor] = {
            "conflicting_gold_signature_count": conflict_groups,
            "conflicting_gold_rows": conflicting_rows,
            "majority_rows": majority,
            "scheduled_rows": scheduled,
            "representation_ceiling": majority / scheduled if scheduled else None,
            "gold_support": dict(sorted(support.items())),
        }
    return {
        "row_count": scheduled,
        "unique_count_signatures": len(signatures),
        "unique_normalized_frequency_signatures": len(frequencies),
        "collision_group_count": sum(len(group) > 1 for group in signatures.values()),
        "collision_rows": sum(len(group) for group in signatures.values() if len(group) > 1),
        "conflicting_frame_signature_count": sum(
            len({canonical(row["targets"]["frame"]) for row in group}) > 1
            for group in signatures.values()
        ),
        "by_factor": by_factor,
    }


def static_audit_report(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> dict[str, Any]:
    current = build() if bundle is None else bundle
    result: dict[str, Any] = {}
    for arm in ARMS:
        train, confirmation = current[arm]["train"], current[arm]["confirmation"]
        result[arm] = {
            "train": _audit_rows(train),
            "confirmation_all": _audit_rows(confirmation),
            "confirmation_by_support": {
                support: _audit_rows([row for row in confirmation
                                      if row["metadata"]["support_class"] == support])
                for support in CONFIRMATION_SUPPORTS
            },
            "union": _audit_rows([*train, *confirmation]),
        }
    return result


def paired_correspondence(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> dict[str, Any]:
    current = build() if bundle is None else bundle
    for split in SPLITS:
        left, right = current[ARMS[0]][split], current[ARMS[1]][split]
        if len(left) != len(right):
            raise ValueError("paired arms have different row counts")
        for a, b in zip(left, right, strict=True):
            if any(a[key] != b[key] for key in ("id", "group_id", "split", "targets", "metadata")):
                raise ValueError("paired row labels or metadata differ")
            left_text, right_text = a["inputs"]["text"], b["inputs"]["text"]
            if (len(left_text) != len(right_text)
                    or sum(a_byte != b_byte for a_byte, b_byte in zip(left_text, right_text, strict=True)) != 4):
                raise ValueError("paired source must differ at exactly four value bytes")
    return {"same_ids_targets_splits_metadata": True, "only_four_value_bytes_changed": True}


def preflight(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> dict[str, Any]:
    current = build() if bundle is None else bundle
    paired = paired_correspondence(current)
    audit = static_audit_report(current)
    for arm in ARMS:
        if len(current[arm]["train"]) != 384 or len(current[arm]["confirmation"]) != 96:
            raise ValueError("fixture row counts differ from frozen plan")
        all_rows = [*current[arm]["train"], *current[arm]["confirmation"]]
        if len({row["id"] for row in all_rows}) != 480:
            raise ValueError("fixture row IDs are not unique")
        if ({row["inputs"]["text"] for row in all_rows}
                & {row["targets"]["text"] for row in all_rows}):
            raise ValueError("source and target surfaces overlap")
        if any(set(model_inputs(row)) != {"context", "text"}
               or model_inputs(row)["context"] != "" for row in all_rows):
            raise ValueError("fixture model inputs include metadata")
        for support in CONFIRMATION_SUPPORTS:
            if audit[arm]["confirmation_by_support"][support]["row_count"] != 48:
                raise ValueError("confirmation support count differs from frozen plan")
    for scope in ("train", "confirmation_all", "union"):
        disjoint = audit["ROLE_DISJOINT"][scope]
        if disjoint["conflicting_frame_signature_count"]:
            raise ValueError("role-disjoint signature has conflicting canonical frames")
        if any(disjoint["by_factor"][factor]["representation_ceiling"] != 1
               for factor in FACTORS):
            raise ValueError("role-disjoint factor ceiling is below one")
    for support in CONFIRMATION_SUPPORTS:
        disjoint = audit["ROLE_DISJOINT"]["confirmation_by_support"][support]
        if disjoint["conflicting_frame_signature_count"]:
            raise ValueError("role-disjoint confirmation signature conflicts")
    aliased_train = audit["ROLE_ALIASED"]["train"]["by_factor"]
    if any(aliased_train[factor]["representation_ceiling"] >= 0.8
           for factor in ("participant", "time")):
        raise ValueError("role-aliased train ceiling did not reproduce the gate")
    for factor, count in (("participant", 6), ("time", 6), ("event", 4), ("operator", 4)):
        if len(aliased_train[factor]["gold_support"]) != count:
            raise ValueError("train lacks a canonical factor class")
        supports = list(aliased_train[factor]["gold_support"].values())
        if len(set(supports)) != 1:
            raise ValueError("train factor class support is not balanced")
    return {"status": "pass", "paired_correspondence": paired,
            "aliased_train_ceiling": {f: aliased_train[f]["representation_ceiling"] for f in FACTORS},
            "disjoint_all_factor_ceilings_one": True}


def manifest(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> dict[str, Any]:
    current = build() if bundle is None else bundle
    gate = preflight(current)
    files = {}
    for arm in ARMS:
        for split in SPLITS:
            payload = ("\n".join(canonical(row) for row in current[arm][split]) + "\n").encode("utf-8")
            files[f"{arm.lower()}-{split}.jsonl"] = sha256_bytes(payload)
    return {"schema": SCHEMA, "counts": {arm: {split: len(current[arm][split]) for split in SPLITS}
                                         for arm in ARMS},
            "spec_sha256": sha256_bytes(SPEC_PATH.read_bytes()),
            "generator_sha256": sha256_bytes(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
            "row_files_sha256": files,
            "static_audit_sha256": sha256_bytes(canonical(static_audit_report(current)).encode("utf-8")),
            "preflight": gate}


def manifest_digest(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> str:
    return sha256_bytes(canonical(manifest(bundle)).encode("utf-8"))


def expected_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def check_expected(bundle: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None) -> dict[str, Any]:
    report = manifest(bundle)
    if report != expected_manifest():
        raise ValueError("role fixture differs from expected-manifest.json")
    return report


__all__ = ["ARMS", "CONFIRMATION_SUPPORTS", "FACTORS", "SPLITS", "build", "canonical",
           "check_expected", "count_signature", "expected_manifest", "manifest",
           "manifest_digest", "model_inputs", "normalized_frequency_signature",
           "paired_correspondence", "preflight", "sha256_bytes", "signature_digest",
           "source_ids", "spec_data", "static_audit_report", "validate_spec"]
