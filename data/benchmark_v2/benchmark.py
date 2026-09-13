"""Frozen benchmark-v2 protocol fixtures; this module does not train a model."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VERSION = "norishio-benchmark-v2.0"
SPLITS = ("train", "diagnostic-validation", "final-holdout")
FIELDS = ("participant", "time", "event", "operator")
PAD, BOS, EOS, SEP, BYTE_OFFSET, VOCAB_SIZE = 0, 1, 2, 3, 4, 260
Row = dict[str, Any]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_source_bytes(path: Path) -> bytes:
    """Hash source consistently across Git LF/CRLF checkout policies."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def spec_data() -> dict[str, Any]:
    spec = json.loads((ROOT / "spec.json").read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("version") != VERSION:
        raise ValueError("unsupported benchmark version")
    expected = {"version", "split_seed", "authorship", "participants", "times", "events",
                "operators", "source_templates", "target_templates"}
    if set(spec) != expected:
        raise ValueError("unexpected spec fields")
    if type(spec["split_seed"]) is not int:
        raise ValueError("split_seed must be an integer")
    sizes = (len(spec["participants"]), len(spec["times"]), len(spec["events"]),
             len(spec["operators"]))
    if sizes != (6, 6, 4, 4):
        raise ValueError("v2 freezes a 6x6x4x4 factor grid")
    if len(spec["source_templates"]) != 2 or len(spec["target_templates"]) != 2:
        raise ValueError("v2 freezes two aligned surface variants")
    for name in ("participants", "times"):
        values = spec[name]
        if any(set(value) != {"id", "surface"} for value in values):
            raise ValueError(f"invalid {name} entry")
        if len({value["id"] for value in values}) != len(values):
            raise ValueError(f"duplicate {name} id")
        if len({value["surface"] for value in values}) != len(values):
            raise ValueError(f"duplicate {name} surface")
    operators = spec["operators"]
    if len(set(operators)) != 4 or not all(isinstance(value, str) and value for value in operators):
        raise ValueError("operators must be four unique strings")
    for event in spec["events"]:
        if set(event) != {"id", "predicates"} or set(event["predicates"]) != set(operators):
            raise ValueError("every event requires one predicate per operator")


def split_plan(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the frozen rule parameters, derived only from the committed seed."""
    spec = spec_data() if spec is None else spec
    size = len(spec["participants"])
    diagnostic_offset = spec["split_seed"] % size
    final_offset = (diagnostic_offset + 2) % size
    return {
        "name": "latin-pair-plus-per-pair-event-v1",
        "seed": spec["split_seed"],
        "diagnostic_pair_offset": diagnostic_offset,
        "final_pair_offset": final_offset,
        "event_rule": "diagnostic=(participant+2*time+seed)%4; final=(diagnostic+1+(participant+time)%3)%4",
    }


def factor_shuffle_plan(spec: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Predetermine one derangement per factor for the broken-semantics control."""
    spec = spec_data() if spec is None else spec
    values = {
        "participant": [value["id"] for value in spec["participants"]],
        "time": [value["id"] for value in spec["times"]],
        "event": [value["id"] for value in spec["events"]],
        "operator": list(spec["operators"]),
    }
    plan: dict[str, dict[str, str]] = {}
    for index, (field, field_values) in enumerate(values.items()):
        shift = 1 + (spec["split_seed"] + index) % (len(field_values) - 1)
        plan[field] = {
            value: field_values[(position + shift) % len(field_values)]
            for position, value in enumerate(field_values)
        }
    return plan


def _split_for(pi: int, ti: int, ei: int, event_count: int,
               plan: dict[str, Any]) -> tuple[str, str]:
    size = 6
    if ti == (pi + plan["diagnostic_pair_offset"]) % size:
        return "diagnostic-validation", "unseen_pair"
    if ti == (pi + plan["final_pair_offset"]) % size:
        return "final-holdout", "unseen_pair"
    diagnostic_event = (pi + 2 * ti + plan["seed"]) % event_count
    final_event = (diagnostic_event + 1 + (pi + ti) % 3) % event_count
    if ei == diagnostic_event:
        return "diagnostic-validation", "unseen_triple"
    if ei == final_event:
        return "final-holdout", "unseen_triple"
    return "train", "train"


def build(spec: dict[str, Any] | None = None) -> dict[str, list[Row]]:
    """Build all rows. Surface variants of one four-factor frame never cross splits."""
    spec = spec_data() if spec is None else deepcopy(spec)
    validate_spec(spec)
    plan = split_plan(spec)
    result: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for pi, participant in enumerate(spec["participants"]):
        for ti, time in enumerate(spec["times"]):
            for ei, event in enumerate(spec["events"]):
                split, support = _split_for(pi, ti, ei, len(spec["events"]), plan)
                for oi, operator in enumerate(spec["operators"]):
                    frame = {"participant": participant["id"], "time": time["id"],
                             "event": event["id"], "operator": operator}
                    predicate = event["predicates"][operator]
                    group = f"frame-{pi}-{ti}-{ei}-{oi}"
                    slots = {"participant": participant["surface"], "time": time["surface"],
                             "predicate": predicate}
                    for variant, (source_template, target_template) in enumerate(zip(
                            spec["source_templates"], spec["target_templates"], strict=True)):
                        result[split].append({
                            "id": f"{group}-v{variant}",
                            "group_id": group,
                            "split": split,
                            "inputs": {"context": "", "text": source_template.format(**slots)},
                            "targets": {"text": target_template.format(**slots), "frame": deepcopy(frame)},
                            "metadata": {"variant": variant, "support_class": support,
                                         "provenance": deepcopy(spec["authorship"])},
                        })
    return result


def model_inputs(row: Row) -> dict[str, str]:
    """Allow only source text. IDs, split, templates, provenance and gold stay outside."""
    inputs = row.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"context", "text"}:
        raise ValueError("inputs must contain only context and text")
    if not all(isinstance(value, str) for value in inputs.values()):
        raise ValueError("source context/text must be strings")
    return dict(inputs)


def source_ids(row: Row) -> list[int]:
    """Encode raw UTF-8 bytes; this creates no categorical full-surface feature ID."""
    raw = canonical(model_inputs(row)).encode("utf-8")
    return [BOS, *(byte + BYTE_OFFSET for byte in raw), SEP]


def teacher_forcing(row: Row) -> dict[str, Any]:
    source = source_ids(row)
    target = row.get("targets", {}).get("text")
    if not isinstance(target, str) or not target:
        raise ValueError("a nonempty reference text is required")
    continuation = [*(byte + BYTE_OFFSET for byte in target.encode("utf-8")), EOS]
    full = source + continuation
    return {"source_ids": source, "input_ids": full[:-1],
            "labels": [-100] * (len(source) - 1) + continuation,
            "frame_position": len(source) - 1}


def _target_lookup(spec: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for participant in spec["participants"]:
        for time in spec["times"]:
            for event in spec["events"]:
                for operator in spec["operators"]:
                    frame = {"participant": participant["id"], "time": time["id"],
                             "event": event["id"], "operator": operator}
                    slots = {"participant": participant["surface"], "time": time["surface"],
                             "predicate": event["predicates"][operator]}
                    for template in spec["target_templates"]:
                        rendered = template.format(**slots)
                        if rendered in lookup:
                            raise ValueError("frozen target grammar is ambiguous")
                        lookup[rendered] = frame
    return lookup


def parse_target_text(text: str, spec: dict[str, Any] | None = None) -> dict[str, str] | None:
    """Parse only the frozen target grammar; near matches remain parse failures."""
    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    spec = spec_data() if spec is None else spec
    validate_spec(spec)
    result = _target_lookup(spec).get(text)
    return deepcopy(result) if result is not None else None


def train_support(bundle: dict[str, list[Row]]) -> dict[str, set[tuple[str, ...]]]:
    frames = [row["targets"]["frame"] for row in bundle["train"]]
    return {
        "participant_time": {(frame["participant"], frame["time"]) for frame in frames},
        "participant_time_event": {
            (frame["participant"], frame["time"], frame["event"]) for frame in frames
        },
    }


def _payload(rows: list[Row]) -> bytes:
    return ("\n".join(canonical(row) for row in rows) + "\n").encode("utf-8")


def validate(bundle: dict[str, list[Row]], spec: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = spec_data() if spec is None else deepcopy(spec)
    validate_spec(spec)
    if set(bundle) != set(SPLITS):
        raise ValueError("unexpected split set")
    expected = build(spec)
    if canonical(bundle) != canonical(expected):
        raise ValueError("bundle differs from the frozen generator result")
    ids: set[str] = set()
    groups: dict[str, str] = {}
    sources: set[str] = set()
    references: dict[str, str] = {}
    atoms = {field: set() for field in FIELDS}
    counts = Counter()
    support_classes: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_lookup = _target_lookup(spec)
    for split in SPLITS:
        for row in bundle[split]:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate id or split mismatch")
            ids.add(row["id"])
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError("surface variant group crosses splits")
            source = canonical(model_inputs(row))
            if source in sources:
                raise ValueError("duplicate source")
            sources.add(source)
            reference = row["targets"]["text"]
            if references.setdefault(reference, split) != split:
                raise ValueError("reference text crosses splits")
            frame = row["targets"]["frame"]
            if set(frame) != set(FIELDS) or not all(isinstance(frame[field], str) for field in FIELDS):
                raise ValueError("invalid frame")
            if target_lookup.get(reference) != frame:
                raise ValueError("reference does not match frozen target grammar")
            if split == "train":
                for field in FIELDS:
                    atoms[field].add(frame[field])
            packed = teacher_forcing(row)
            if len(packed["input_ids"]) != len(packed["labels"]):
                raise ValueError("teacher forcing alignment error")
            counts[split] += 1
            support_classes[split][row["metadata"]["support_class"]] += 1
    expected_atoms = {
        "participant": {value["id"] for value in spec["participants"]},
        "time": {value["id"] for value in spec["times"]},
        "event": {value["id"] for value in spec["events"]},
        "operator": set(spec["operators"]),
    }
    if atoms != expected_atoms:
        raise ValueError("not every atomic value is supported by train")
    support = train_support(bundle)
    for split in ("diagnostic-validation", "final-holdout"):
        for row in bundle[split]:
            frame = row["targets"]["frame"]
            pair = (frame["participant"], frame["time"])
            triple = (*pair, frame["event"])
            declared = row["metadata"]["support_class"]
            actual = "unseen_pair" if pair not in support["participant_time"] else "unseen_triple"
            if declared != actual or triple in support["participant_time_event"]:
                raise ValueError("incorrect evaluation support class")
    hashes = {f"{split}.jsonl": sha256_bytes(_payload(bundle[split])) for split in SPLITS}
    report = {
        "version": VERSION,
        "counts": dict(counts),
        "groups": dict(Counter(groups.values())),
        "support_classes": {split: dict(support_classes[split]) for split in SPLITS},
        "train_support": {name: len(values) for name, values in support.items()},
        "split_plan": split_plan(spec),
        "factor_shuffle_plan": factor_shuffle_plan(spec),
        "spec_sha256": sha256_bytes((canonical(spec) + "\n").encode("utf-8")),
        "generator_sha256": sha256_bytes(normalized_source_bytes(Path(__file__))),
        "sha256": hashes,
        "checks": {"unique_ids": True, "group_disjoint": True, "unique_sources": True,
                   "references_split_disjoint": True, "train_atomic_coverage": True,
                   "pair_and_triple_holdouts": True, "teacher_forcing_alignment": True,
                   "source_allowlist": True, "target_grammar_unambiguous": True,
                   "no_full_surface_categorical_id": True},
        "scope": "frozen authored structural benchmark; no trained-model or linguistic-quality claim",
    }
    digest_source = {key: report[key] for key in (
        "version", "counts", "groups", "support_classes", "train_support", "split_plan",
        "factor_shuffle_plan",
        "spec_sha256", "generator_sha256", "sha256",
    )}
    report["content_digest_sha256"] = sha256_bytes(canonical(digest_source).encode("utf-8"))
    return report


def expected_manifest() -> dict[str, Any]:
    return json.loads((ROOT / "expected-manifest.json").read_text(encoding="utf-8"))


def check_expected(report: dict[str, Any]) -> None:
    if canonical(report) != canonical(expected_manifest()):
        raise ValueError("generated benchmark does not match expected-manifest.json")


def evaluation_rows(bundle: dict[str, list[Row]], split: str, *, evaluate_final: bool = False,
                    manifest_digest: str | None = None) -> list[Row]:
    """Expose final rows only through an explicit, manifest-bound evaluation call."""
    if split not in {"diagnostic-validation", "final-holdout"}:
        raise ValueError("evaluation split must be diagnostic-validation or final-holdout")
    report = validate(bundle)
    check_expected(report)
    if split == "final-holdout":
        if not evaluate_final:
            raise PermissionError("final-holdout requires explicit evaluate_final=True")
        if manifest_digest != report["content_digest_sha256"]:
            raise ValueError("final-holdout manifest digest mismatch")
    return deepcopy(bundle[split])


def write_bundle(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    bundle = build()
    report = validate(bundle)
    check_expected(report)
    out.mkdir(parents=True)
    for split in SPLITS:
        (out / f"{split}.jsonl").write_bytes(_payload(bundle[split]))
    (out / "manifest.json").write_bytes((canonical(report) + "\n").encode("utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("codex/work_output/norishio-benchmark-v2"))
    parser.add_argument("--check", action="store_true", help="validate and compare the frozen manifest")
    args = parser.parse_args()
    try:
        report = validate(build())
        check_expected(report)
        if not args.check:
            report = write_bundle(args.out)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"benchmark error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
