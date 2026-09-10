"""Authored conditional-generation fixtures, not mined text or a trained model."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
VERSION = "norishio-toy-1.0"
SPLITS = ("train", "validation", "test")
PAD, BOS, EOS, SEP, BYTE_OFFSET, VOCAB_SIZE = 0, 1, 2, 3, 4, 260
EVENT_SENSES = {"MEET": "meet_person", "STAY": "be_present", "SEPARATE": "separate_people"}
Row = dict[str, Any]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def seed_data() -> dict[str, Any]:
    value = json.loads((ROOT / "seed.json").read_text(encoding="utf-8"))
    if value.get("version") != VERSION:
        raise ValueError("unsupported seed version")
    if len(value["people"]) != 5 or len(value["times"]) != 5:
        raise ValueError("v1 uses a fixed 5 by 5 composition grid")
    if len(set(value["people"])) != 5 or len(set(value["times"])) != 5:
        raise ValueError("duplicate slot value")
    return value


def build(seed: dict[str, Any] | None = None) -> dict[str, list[Row]]:
    """All contrasts/paraphrases of one person/time situation stay in one split."""
    seed = seed_data() if seed is None else deepcopy(seed)
    result: dict[str, list[Row]] = {s: [] for s in (*SPLITS, "diagnostic")}
    for pi, person in enumerate(seed["people"]):
        for ti, time in enumerate(seed["times"]):
            split = {0: "test", 1: "validation"}.get((pi + ti) % 5, "train")
            group = f"situation-{pi}-{ti}"
            for condition in seed["conditions"]:
                event = condition["event"]
                for vi, template in enumerate(condition["source"]):
                    slots = {"person": person, "time": time}
                    result[split].append({
                        "id": f"{group}-{condition['id']}-{vi}",
                        "group_id": group,
                        "split": split,
                        "inputs": {"context": "", "text": template.format(**slots)},
                        "targets": {
                            "text": condition["target"].format(**slots),
                            "sense": EVENT_SENSES[event],
                            "sememes": [event, "HUMAN_INTERACTION"],
                            "concept": {
                                "event": event, "operators": list(condition["operators"]),
                                "agent": condition["agent"], "participant": person,
                                "time": time, "location": "HERE" if event == "STAY" else "UNSPECIFIED",
                                "repeat_marked": condition["repeat"],
                            },
                        },
                        "metadata": {"condition": condition["id"], "variant": vi,
                                     "provenance": deepcopy(seed["authorship"])},
                    })
    for i, example in enumerate(seed["diagnostics"]):
        result["diagnostic"].append({
            "id": f"diagnostic-{i:03}", "group_id": "diagnostic-" + example["group"],
            "split": "diagnostic",
            "inputs": {"context": example["context"], "text": example["text"]},
            "targets": {"text": example["target"], "sense": None, "sememes": None,
                        "concept": None, "interpretation": example["interpretation"]},
            "metadata": {"provenance": deepcopy(seed["authorship"]),
                         "perturbation": deepcopy(example.get("perturbation")),
                         "diagnostic_only": True},
        })
    return result


def model_inputs(row: Row) -> dict[str, str]:
    """An allowlist boundary; IDs, split, labels and provenance are not features."""
    inputs = row.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"context", "text"}:
        raise ValueError("inputs must contain only context and text")
    if not all(isinstance(v, str) for v in inputs.values()):
        raise ValueError("source context/text must be strings")
    return dict(inputs)


def source_ids(row: Row) -> list[int]:
    """Fixed UTF-8 byte vocabulary; no vocabulary fit on any evaluation split."""
    raw = canonical(model_inputs(row)).encode("utf-8")
    return [BOS, *(b + BYTE_OFFSET for b in raw), SEP]


def teacher_forcing(row: Row) -> dict[str, Any]:
    """Conditional causal decoder input. Only previous target bytes are visible.

    The multichannel encoder may read source_ids only, never this full input_ids.
    The source-end concept prediction sees the source prefix, not the answer.
    """
    source = source_ids(row)
    answer = row["targets"]["text"]
    if not isinstance(answer, str) or not answer:
        raise ValueError("a nonempty reference text is required")
    continuation = [*(b + BYTE_OFFSET for b in answer.encode("utf-8")), EOS]
    full = source + continuation
    return {"source_ids": source, "input_ids": full[:-1],
            "labels": [-100] * (len(source) - 1) + continuation,
            "concept_position": len(source) - 1}


def supervised_targets(row: Row, drop: Iterable[str] = ()) -> dict[str, Any]:
    """None means an absent annotation, never a negative/unknown semantic label."""
    if isinstance(drop, str):
        raise ValueError("drop must be an iterable of field names, not a string")
    dropped = set(drop)
    if dropped - {"sense", "sememes", "concept"}:
        raise ValueError("only auxiliary supervision can be dropped")
    result = deepcopy(row["targets"])
    for name in dropped:
        result[name] = None
    return result


def validate(bundle: dict[str, list[Row]]) -> dict[str, Any]:
    if set(bundle) != {*SPLITS, "diagnostic"}:
        raise ValueError("unexpected split set")
    ids: set[str] = set()
    groups: dict[str, str] = {}
    sources: dict[str, str] = {}
    references: dict[str, str] = {}
    labels: dict[str, set[str]] = {s: set() for s in SPLITS}
    counts = Counter()
    for split, rows in bundle.items():
        if not rows:
            raise ValueError("empty split: " + split)
        for row in rows:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate id or split mismatch")
            ids.add(row["id"])
            old = groups.setdefault(row["group_id"], split)
            if old != split:
                raise ValueError("sibling group crosses splits")
            inputs = model_inputs(row)
            if row["metadata"]["provenance"]["kind"] != "authored_demo":
                raise ValueError("authorship declaration lost")
            if not row["targets"]["text"]:
                raise ValueError("empty reference text")
            if split in SPLITS:
                key = canonical(inputs)
                if key in sources:
                    raise ValueError("duplicate core source")
                sources[key] = split
                ref = row["targets"]["text"]
                if references.setdefault(ref, split) != split:
                    raise ValueError("reference text crosses splits")
                concept = row["targets"]["concept"]
                event = concept["event"]
                if row["targets"]["sense"] != EVENT_SENSES[event]:
                    raise ValueError("sense/event mismatch")
                if row["targets"]["sememes"] != [event, "HUMAN_INTERACTION"]:
                    raise ValueError("sememe/event mismatch")
                if any(op not in {"NOT", "WANT", "PLAN", "POSSIBLE"} for op in concept["operators"]):
                    raise ValueError("unknown scope operator")
                if type(concept["repeat_marked"]) is not bool:
                    raise ValueError("repeat marker must be bool")
                labels[split].add(row["metadata"]["condition"])
            packed = teacher_forcing(row)
            if len(packed["input_ids"]) != len(packed["labels"]):
                raise ValueError("teacher forcing alignment error")
            counts[split] += 1
    if labels["train"] != labels["validation"] or labels["train"] != labels["test"]:
        raise ValueError("unbalanced semantic condition coverage")
    return {"version": VERSION, "counts": dict(counts),
            "groups": dict(Counter(groups.values())), "core_conditions": sorted(labels["train"]),
            "checks": {"unique_ids": True, "group_disjoint": True,
                       "unique_core_sources": True, "core_references_split_disjoint": True,
                       "teacher_forcing_alignment": True},
            "scope": "structural fixture validation only; no learned performance or independent linguistic validation"}


def write_bundle(out: Path) -> dict[str, Any]:
    """Refuse replacing an existing export. Artifacts have deterministic bytes."""
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    bundle = build()
    report = validate(bundle)
    out.mkdir(parents=True)
    hashes = {}
    for split, rows in bundle.items():
        payload = ("\n".join(canonical(row) for row in rows) + "\n").encode("utf-8")
        filename = split + ".jsonl"
        (out / filename).write_bytes(payload)
        hashes[filename] = hashlib.sha256(payload).hexdigest()
    report["sha256"] = hashes
    (out / "manifest.json").write_bytes((canonical(report) + "\n").encode("utf-8"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("codex/work_output/norishio-toy-v1"))
    parser.add_argument("--check", action="store_true", help="validate in memory without writing")
    args = parser.parse_args()
    try:
        report = validate(build()) if args.check else write_bundle(args.out)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"dataset error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
