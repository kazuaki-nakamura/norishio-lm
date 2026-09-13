"""Deterministic, CPU/offline factor corpus for the Issue #36 Phase 0 benchmark.

The fixture is deliberately authored as a small structural corpus.  It uses the
same four factor vocabularies and predicate family as benchmark-v2, but has its
own generator namespace, templates, fixed seeds, split rule, and final access
gate.  No model, network call, or categorical whole-surface feature is involved.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict


VERSION = "norishio-benchmark-v3.0"
SPLITS = ("train", "diagnostic-validation", "final-confirmation")
EVALUATION_SPLITS = ("diagnostic-validation", "final-confirmation")
FIELDS = ("participant", "time", "event", "operator")
INPUT_FIELDS = ("context", "text")
LEAKAGE_FORBIDDEN_FIELDS = ("targets", "gold", "id", "group_id", "template", "split", "full_surface_id")
BYTE_OFFSET = 4
BOS, EOS, SEP = 1, 2, 3
VOCAB_SIZE = 260

# These are intentionally distinct from benchmark-v2's 3401 seed.  They are
# part of the committed v3 schema and are never selected at runtime.
GENERATOR_SEED = 20260914
SPLIT_SEED = 90731

class Frame(TypedDict):
    participant: str
    time: str
    event: str
    operator: str


class Inputs(TypedDict):
    context: str
    text: str


class Targets(TypedDict):
    text: str
    frame: Frame


class Metadata(TypedDict):
    variant: int
    support_class: str
    generator_seed: int
    provenance: dict[str, Any]


class CorpusRow(TypedDict):
    id: str
    group_id: str
    split: str
    inputs: Inputs
    targets: Targets
    metadata: Metadata


JsonObject: TypeAlias = dict[str, Any]
Row: TypeAlias = CorpusRow


@dataclass(frozen=True)
class FactorValue:
    id: str
    surface: str


@dataclass(frozen=True)
class EventValue:
    id: str
    predicates: Mapping[str, str]


@dataclass(frozen=True)
class BenchmarkSpec:
    version: str
    generator_seed: int
    split_seed: int
    authorship: Mapping[str, Any]
    participants: tuple[FactorValue, ...]
    times: tuple[FactorValue, ...]
    events: tuple[EventValue, ...]
    operators: tuple[str, ...]
    source_templates: tuple[str, ...]
    target_templates: tuple[str, ...]


_SPEC_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark_v3_factor_path" / "spec.json"
_MANIFEST_PATH = _SPEC_PATH.with_name("expected-manifest.json")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_source_bytes(path: Path | None = None) -> bytes:
    source = Path(__file__) if path is None else Path(path)
    return source.read_bytes().replace(b"\r\n", b"\n")


def spec_data() -> dict[str, Any]:
    result = json.loads(_SPEC_PATH.read_text(encoding="utf-8"))
    validate_spec(result)
    return deepcopy(result)


def validate_spec(spec: Mapping[str, Any]) -> None:
    expected = {"version", "generator_seed", "split_seed", "authorship", "participants", "times",
                "events", "operators", "source_templates", "target_templates"}
    if set(spec) != expected or spec.get("version") != VERSION:
        raise ValueError("unsupported or incomplete benchmark-v3 specification")
    if type(spec["generator_seed"]) is not int or type(spec["split_seed"]) is not int:
        raise ValueError("generator and split seeds must be integers")
    if len(spec["participants"]) != 6 or len(spec["times"]) != 6 or len(spec["events"]) != 4 or len(spec["operators"]) != 4:
        raise ValueError("v3 freezes a 6x6x4x4 factor grid")
    if len(spec["source_templates"]) != 2 or len(spec["target_templates"]) != 2:
        raise ValueError("v3 freezes two aligned surface variants")
    for name in ("participants", "times"):
        entries = spec[name]
        if any(set(entry) != {"id", "surface"} for entry in entries):
            raise ValueError(f"invalid {name} entry")
        if len({entry["id"] for entry in entries}) != len(entries) or len({entry["surface"] for entry in entries}) != len(entries):
            raise ValueError(f"duplicate {name} value")
    operators = spec["operators"]
    if len(set(operators)) != 4 or not all(type(value) is str and value for value in operators):
        raise ValueError("operators must be four unique strings")
    for event in spec["events"]:
        if set(event) != {"id", "predicates"} or set(event["predicates"]) != set(operators):
            raise ValueError("every event requires one predicate per operator")
        if not isinstance(event["id"], str) or not all(isinstance(value, str) and value for value in event["predicates"].values()):
            raise ValueError("event values must be nonempty strings")
    if any("{participant}" not in template or "{time}" not in template or "{predicate}" not in template
           for template in (*spec["source_templates"], *spec["target_templates"])):
        raise ValueError("templates must expose participant, time, and predicate slots")


def split_plan(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    size, event_count = len(current["participants"]), len(current["events"])
    diagnostic_offset = current["split_seed"] % size
    final_offset = (diagnostic_offset + 3) % size
    return {
        "name": "v3-latin-pair-plus-per-pair-event-v1",
        "generator_seed": current["generator_seed"], "seed": current["split_seed"],
        "diagnostic_pair_offset": diagnostic_offset, "final_pair_offset": final_offset,
        "event_rule": "diagnostic=(participant+time+seed)%4; final=(diagnostic+1+(participant+time+seed)%3)%4",
        "event_count": event_count,
    }


def _split_for(pi: int, ti: int, ei: int, spec: Mapping[str, Any]) -> tuple[str, str]:
    plan = split_plan(spec)
    size, event_count = len(spec["participants"]), len(spec["events"])
    if ti == (pi + plan["diagnostic_pair_offset"]) % size:
        return "diagnostic-validation", "unseen_pair"
    if ti == (pi + plan["final_pair_offset"]) % size:
        return "final-confirmation", "unseen_pair"
    diagnostic_event = (pi + ti + plan["seed"]) % event_count
    final_event = (diagnostic_event + 1 + (pi + ti + plan["seed"]) % 3) % event_count
    if ei == diagnostic_event:
        return "diagnostic-validation", "unseen_triple"
    if ei == final_event:
        return "final-confirmation", "unseen_triple"
    return "train", "train"


def build(spec: Mapping[str, Any] | None = None) -> dict[str, list[Row]]:
    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    result: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for pi, participant in enumerate(current["participants"]):
        for ti, time in enumerate(current["times"]):
            for ei, event in enumerate(current["events"]):
                split, support = _split_for(pi, ti, ei, current)
                for oi, operator in enumerate(current["operators"]):
                    frame = {"participant": participant["id"], "time": time["id"],
                             "event": event["id"], "operator": operator}
                    slots = {"participant": participant["surface"], "time": time["surface"],
                             "predicate": event["predicates"][operator]}
                    group = f"v3-frame-{pi}-{ti}-{ei}-{oi}"
                    for variant, (source_template, target_template) in enumerate(zip(current["source_templates"], current["target_templates"], strict=True)):
                        result[split].append({
                            "id": f"{group}-v{variant}", "group_id": group, "split": split,
                            "inputs": {"context": "", "text": source_template.format(**slots)},
                            "targets": {"text": target_template.format(**slots), "frame": deepcopy(frame)},
                            "metadata": {"variant": variant, "support_class": support,
                                         "generator_seed": current["generator_seed"],
                                         "provenance": deepcopy(current["authorship"])},
                        })
    return result


def model_inputs(row: Mapping[str, Any]) -> dict[str, str]:
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_FIELDS):
        raise ValueError("inputs must contain only context and text")
    if not all(isinstance(inputs[field], str) for field in INPUT_FIELDS):
        raise ValueError("source context/text must be strings")
    return {field: inputs[field] for field in INPUT_FIELDS}


def source_ids(row: Mapping[str, Any]) -> list[int]:
    raw = canonical(model_inputs(row)).encode("utf-8")
    return [BOS, *(value + BYTE_OFFSET for value in raw), SEP]


def teacher_forcing(row: Mapping[str, Any]) -> dict[str, Any]:
    source = source_ids(row)
    target = row.get("targets", {}).get("text") if isinstance(row.get("targets"), Mapping) else None
    if not isinstance(target, str) or not target:
        raise ValueError("a nonempty reference text is required")
    continuation = [*(value + BYTE_OFFSET for value in target.encode("utf-8")), EOS]
    full = source + continuation
    return {"source_ids": source, "input_ids": full[:-1], "labels": [-100] * (len(source) - 1) + continuation,
            "frame_position": len(source) - 1}


def _target_lookup(spec: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for participant in spec["participants"]:
        for time in spec["times"]:
            for event in spec["events"]:
                for operator in spec["operators"]:
                    frame = {"participant": participant["id"], "time": time["id"], "event": event["id"], "operator": operator}
                    slots = {"participant": participant["surface"], "time": time["surface"], "predicate": event["predicates"][operator]}
                    for template in spec["target_templates"]:
                        rendered = template.format(**slots)
                        if rendered in lookup:
                            raise ValueError("frozen target grammar is ambiguous")
                        lookup[rendered] = frame
    return lookup


def parse_target_text(text: str, spec: Mapping[str, Any] | None = None) -> dict[str, str] | None:
    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    frame = _target_lookup(current).get(text)
    return deepcopy(frame) if frame is not None else None


def train_support(bundle: Mapping[str, Sequence[Row]]) -> dict[str, set[tuple[str, ...]]]:
    frames = [row["targets"]["frame"] for row in bundle["train"]]
    return {"participant_time": {(f["participant"], f["time"]) for f in frames},
            "participant_time_event": {(f["participant"], f["time"], f["event"]) for f in frames}}


def _payload(rows: Sequence[Row]) -> bytes:
    return ("\n".join(canonical(row) for row in rows) + "\n").encode("utf-8")


def _bundle_digest_source(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in ("version", "counts", "groups", "support_classes", "train_support", "split_plan", "leakage_allowlist", "spec_sha256", "generator_sha256", "sha256")}


def expected_manifest() -> dict[str, Any]:
    """Load the committed manifest that authenticates the v3 freeze."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("expected manifest must be a JSON object")
    return manifest


def check_expected(report: Mapping[str, Any]) -> None:
    """Require exact equality with the committed v3 manifest."""
    if canonical(dict(report)) != canonical(expected_manifest()):
        raise ValueError("generated benchmark does not match expected-manifest.json")


def validate(bundle: Mapping[str, Sequence[Row]], spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    if set(bundle) != set(SPLITS):
        raise ValueError("unexpected split set")
    expected = build(current)
    if canonical(bundle) != canonical(expected):
        raise ValueError("bundle differs from the frozen v3 generator result")
    ids: set[str] = set(); groups: dict[str, str] = {}; sources: set[str] = set(); target_texts: set[str] = set(); references: dict[str, str] = {}
    atoms = {field: set() for field in FIELDS}; counts = Counter(); support_classes = {split: Counter() for split in SPLITS}
    target_lookup = _target_lookup(current)
    for split in SPLITS:
        for row in bundle[split]:
            if row["id"] in ids or row["split"] != split: raise ValueError("duplicate id or split mismatch")
            ids.add(row["id"])
            if groups.setdefault(row["group_id"], split) != split: raise ValueError("surface variant group crosses splits")
            source = canonical(model_inputs(row))
            if source in sources: raise ValueError("duplicate source")
            sources.add(source)
            reference, frame = row["targets"]["text"], row["targets"]["frame"]
            if row["inputs"]["text"] == reference: raise ValueError("source text equals target text")
            target_texts.add(reference)
            if references.setdefault(reference, split) != split: raise ValueError("reference text crosses splits")
            if set(frame) != set(FIELDS) or not all(isinstance(frame[field], str) for field in FIELDS): raise ValueError("invalid frame")
            if target_lookup.get(reference) != frame: raise ValueError("reference does not match frozen target grammar")
            if split == "train":
                for field in FIELDS: atoms[field].add(frame[field])
            packed = teacher_forcing(row)
            if len(packed["input_ids"]) != len(packed["labels"]): raise ValueError("teacher forcing alignment error")
            counts[split] += 1; support_classes[split][row["metadata"]["support_class"]] += 1
    if sources & target_texts: raise ValueError("source and target text sets overlap")
    expected_atoms = {"participant": {x["id"] for x in current["participants"]}, "time": {x["id"] for x in current["times"]}, "event": {x["id"] for x in current["events"]}, "operator": set(current["operators"])}
    if atoms != expected_atoms: raise ValueError("not every atomic value is supported by train")
    support = train_support(bundle)
    for split in EVALUATION_SPLITS:
        for row in bundle[split]:
            frame = row["targets"]["frame"]; pair = (frame["participant"], frame["time"]); triple = (*pair, frame["event"])
            expected_class = "unseen_pair" if pair not in support["participant_time"] else "unseen_triple"
            if row["metadata"]["support_class"] != expected_class or triple in support["participant_time_event"]: raise ValueError("incorrect evaluation support class")
    hashes = {f"{split}.jsonl": sha256_bytes(_payload(bundle[split])) for split in SPLITS}
    report: dict[str, Any] = {
        "version": VERSION, "counts": dict(counts), "groups": dict(Counter(groups.values())),
        "support_classes": {split: dict(support_classes[split]) for split in SPLITS},
        "train_support": {name: len(values) for name, values in support.items()}, "split_plan": split_plan(current),
        "spec_sha256": sha256_bytes(canonical_bytes(current) + b"\n"), "generator_sha256": sha256_bytes(normalized_source_bytes()), "sha256": hashes,
        "leakage_allowlist": {"model_inputs": list(INPUT_FIELDS), "encoding": "canonical UTF-8 bytes with BOS/SEP; no whole-surface categorical id", "forbidden_row_fields": list(LEAKAGE_FORBIDDEN_FIELDS)},
        "checks": {"unique_ids": True, "group_disjoint": True, "unique_sources": True, "unique_targets": True, "source_target_disjoint": True, "references_split_disjoint": True, "train_atomic_coverage": True, "pair_and_triple_holdouts": True, "teacher_forcing_alignment": True, "source_allowlist": True, "target_grammar_unambiguous": True, "no_full_surface_categorical_id": True, "v2_rows_not_reused": True},
        "scope": "frozen authored structural benchmark; no trained-model or linguistic-quality claim",
    }
    report["content_digest_sha256"] = sha256_bytes(canonical_bytes(_bundle_digest_source(report)))
    return report


def evaluation_rows(bundle: Mapping[str, Sequence[Row]], split: str, *, evaluate_final: bool = False, manifest_digest: str | None = None) -> list[Row]:
    if split not in EVALUATION_SPLITS: raise ValueError("evaluation split must be diagnostic-validation or final-confirmation")
    report = validate(bundle)
    check_expected(report)
    if split == "final-confirmation":
        if evaluate_final is not True: raise PermissionError("final-confirmation requires explicit evaluate_final=True")
        if manifest_digest != report["content_digest_sha256"]: raise ValueError("final-confirmation manifest digest mismatch")
    return deepcopy(list(bundle[split]))


def write_bundle(out: str | Path) -> dict[str, Any]:
    target = Path(out)
    if target.exists(): raise FileExistsError(f"output already exists: {target}")
    bundle = build(); report = validate(bundle); check_expected(report); target.mkdir(parents=True)
    for split in SPLITS: (target / f"{split}.jsonl").write_bytes(_payload(bundle[split]))
    (target / "manifest.json").write_bytes((canonical(report) + "\n").encode("utf-8"))
    return report


__all__ = ["BenchmarkSpec", "CorpusRow", "EVALUATION_SPLITS", "EventValue", "FIELDS", "FactorValue", "Frame", "GENERATOR_SEED", "INPUT_FIELDS", "Inputs", "LEAKAGE_FORBIDDEN_FIELDS", "Metadata", "SPLITS", "SPLIT_SEED", "Targets", "build", "canonical", "canonical_bytes", "check_expected", "evaluation_rows", "expected_manifest", "model_inputs", "normalized_source_bytes", "parse_target_text", "sha256_bytes", "source_ids", "spec_data", "split_plan", "teacher_forcing", "train_support", "validate", "validate_spec", "write_bundle"]
