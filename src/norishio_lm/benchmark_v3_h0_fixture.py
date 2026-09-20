"""Authored future-only fixture for the H0 decoder-path experiment.

This module defines a small, deterministic structural corpus.  It contains no
training, inference, result, checkpoint, or network access path.  The fixture
is intentionally independent of the consumed benchmark-v2/v3 artifacts: its
surface vocabulary is an authored ``h0-`` namespace and its leakage audit is
performed from source specifications only.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict


VERSION = "norishio-benchmark-v3-h0-confirmation.v1"
SCHEMA = "norishio.benchmark-v3-h0-confirmation-fixture.v1"
SPLITS = ("train", "confirmation")
FIELDS = ("participant", "time", "event", "operator")
INPUT_FIELDS = ("context", "text")
SUPPORT_TRAIN = "train"
SUPPORT_UNSEEN_PAIR = "unseen_pair"
SUPPORT_SEEN_UNSEEN_TRIPLE = "seen_pair/unseen_triple"
CONFIRMATION_SUPPORTS = (SUPPORT_UNSEEN_PAIR, SUPPORT_SEEN_UNSEEN_TRIPLE)
BYTE_OFFSET = 4
BOS, EOS, SEP = 1, 2, 3
VOCAB_SIZE = 260
GENERATOR_SEED = 20260920
SPLIT_SEED = 44017

SPEC_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark_v3_h0_confirmation" / "spec.json"
MANIFEST_PATH = SPEC_PATH.with_name("expected-manifest.json")


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


Row: TypeAlias = CorpusRow
JsonObject: TypeAlias = dict[str, Any]


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
    selection: Mapping[str, Any]


def canonical(value: Any) -> str:
    """Serialize a JSON value with the fixture's canonical ordering."""

    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("value is not canonical JSON-compatible") from exc


def canonical_bytes(value: Any) -> bytes:
    return canonical(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_source_bytes(path: Path | None = None) -> bytes:
    source = Path(__file__) if path is None else Path(path)
    return source.read_bytes().replace(b"\r\n", b"\n")


def _authorship() -> dict[str, Any]:
    return {
        "kind": "authored_future_fixture",
        "source": "data/benchmark_v3_h0_confirmation/spec.json",
        "human_verified": False,
        "private_data_used": False,
        "derived_from": "new_h0_namespace_only; prior_v2_v3_artifacts_unopened",
    }


def spec_data() -> dict[str, Any]:
    """Load and validate the tracked authored specification."""

    try:
        value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H0 fixture specification is unreadable") from exc
    validate_spec(value)
    return deepcopy(value)


def validate_spec(spec: Mapping[str, Any]) -> None:
    expected = {
        "version", "generator_seed", "split_seed", "authorship", "participants",
        "times", "events", "operators", "source_templates", "target_templates",
        "selection",
    }
    if set(spec) != expected or spec.get("version") != VERSION:
        raise ValueError("unsupported or incomplete H0 fixture specification")
    if type(spec["generator_seed"]) is not int or type(spec["split_seed"]) is not int:
        raise ValueError("fixture seeds must be integers")
    authorship = spec["authorship"]
    if not isinstance(authorship, Mapping) or authorship != _authorship():
        raise ValueError("fixture authorship/provenance changed")
    if len(spec["participants"]) != 6 or len(spec["times"]) != 6:
        raise ValueError("fixture requires six participant and time values")
    if len(spec["events"]) != 4 or len(spec["operators"]) != 4:
        raise ValueError("fixture requires four event and operator values")
    for name in ("participants", "times"):
        values = spec[name]
        if any(set(item) != {"id", "surface"} for item in values):
            raise ValueError(f"invalid {name} entry")
        if len({item["id"] for item in values}) != len(values):
            raise ValueError(f"duplicate {name} id")
        if len({item["surface"] for item in values}) != len(values):
            raise ValueError(f"duplicate {name} surface")
        if not all(isinstance(item["surface"], str) and item["surface"] for item in values):
            raise ValueError(f"invalid {name} surface")
    operators = spec["operators"]
    if len(set(operators)) != 4 or not all(isinstance(item, str) and item for item in operators):
        raise ValueError("operators must be four unique strings")
    for event in spec["events"]:
        if set(event) != {"id", "predicates"}:
            raise ValueError("invalid event entry")
        if not isinstance(event["id"], str) or not event["id"]:
            raise ValueError("event ids must be nonempty strings")
        if set(event["predicates"]) != set(operators):
            raise ValueError("every event requires one predicate per operator")
        if not all(isinstance(value, str) and value for value in event["predicates"].values()):
            raise ValueError("event predicates must be nonempty strings")
    if len({event["id"] for event in spec["events"]}) != len(spec["events"]):
        raise ValueError("duplicate event id")
    for template in (*spec["source_templates"], *spec["target_templates"]):
        if not all(slot in template for slot in ("{participant}", "{time}", "{predicate}")):
            raise ValueError("templates must expose all factor surface slots")
    if len(spec["source_templates"]) != 2 or len(spec["target_templates"]) != 2:
        raise ValueError("fixture freezes exactly two source and target variants")
    selection = spec["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "unseen_pair_grid", "seen_pair_grid", "train_pair_exclusion",
        "train_event_rule", "confirmation_event_rule", "confirmation_operator_rule",
    }:
        raise ValueError("fixture selection plan changed")


def _spec_lookup(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    return (
        list(spec["participants"]), list(spec["times"]), list(spec["events"]),
        list(spec["operators"]),
    )


def _grid_pairs(spec: Mapping[str, Any]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return train, unseen-pair, and intentionally unused pair coordinates."""

    unseen = {(pi, ti) for pi in (0, 1) for ti in (0, 1, 2)}
    seen_confirmation = {(pi, ti) for pi in (2, 3) for ti in (3, 4, 5)}
    unused = {(pi, ti) for pi in (4, 5) for ti in (3, 4, 5)}
    train = {
        (pi, ti) for pi in range(6) for ti in range(6)
        if (pi, ti) not in unseen and (pi, ti) not in unused
    }
    if len(train) != 24 or len(unseen) != 6 or len(seen_confirmation) != 6:
        raise ValueError("fixture pair plan does not have the frozen shape")
    if not seen_confirmation <= train:
        raise ValueError("seen confirmation pairs must be train-supported")
    return train, unseen, unused


def split_plan(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    train, unseen, unused = _grid_pairs(current)
    return {
        "name": "h0-2x3-support-grid-v1",
        "generator_seed": current["generator_seed"],
        "seed": current["split_seed"],
        "train_pair_count": len(train),
        "unseen_pair_coordinates": [[p, t] for p, t in sorted(unseen)],
        "seen_confirmation_coordinates": [[p, t] for p, t in sorted({(p, t) for p, t in train if p in (2, 3) and t in (3, 4, 5)})],
        "unused_pair_coordinates": [[p, t] for p, t in sorted(unused)],
        "train_event_rule": "non_seen_pair=(participant+time)%4 and next modulo 4; seen_confirmation pairs use 0 and 1",
        "confirmation_rule": "unseen pair grid uses events 0,1 and operators 0,1; seen pair grid uses events 2,3 and operators 2,3",
    }


def _frame_rows(spec: Mapping[str, Any]) -> list[tuple[str, str, Frame, str]]:
    participants, times, events, operators = _spec_lookup(spec)
    train_pairs, unseen_pairs, _ = _grid_pairs(spec)
    seen_pairs = {(pi, ti) for pi in (2, 3) for ti in (3, 4, 5)}
    rows: list[tuple[str, str, Frame, str]] = []
    for pi in range(6):
        for ti in range(6):
            pair = (pi, ti)
            if pair in unseen_pairs:
                configurations = [((0, 1), (0, 1), SUPPORT_UNSEEN_PAIR, "confirmation")]
            elif pair in seen_pairs:
                configurations = [
                    ((0, 1), (0, 1, 2, 3), SUPPORT_TRAIN, "train"),
                    ((2, 3), (2, 3), SUPPORT_SEEN_UNSEEN_TRIPLE, "confirmation"),
                ]
            elif pair in train_pairs:
                base = (pi + ti) % 4
                configurations = [
                    ((base, (base + 1) % 4), (0, 1, 2, 3), SUPPORT_TRAIN, "train")
                ]
            else:
                continue
            for event_indices, operator_indices, support, split in configurations:
                for ei in event_indices:
                    for oi in operator_indices:
                        event = events[ei]
                        frame: Frame = {
                            "participant": participants[pi]["id"],
                            "time": times[ti]["id"],
                            "event": event["id"],
                            "operator": operators[oi],
                        }
                        group = f"h0-frame-p{pi}-t{ti}-e{ei}-o{oi}"
                        rows.append((group, split, frame, support))
    # There are 24*2 train frames and 12*2 confirmation frames.
    if sum(split == "train" for _, split, _, _ in rows) != 192:
        raise ValueError("fixture train frame count changed")
    if sum(split == "confirmation" for _, split, _, _ in rows) != 48:
        raise ValueError("fixture confirmation frame count changed")
    return rows


def build(spec: Mapping[str, Any] | None = None) -> dict[str, list[Row]]:
    """Build the complete authored fixture in canonical row order."""

    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    participants, times, events, operators = _spec_lookup(current)
    result: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for group, split, frame, support in _frame_rows(current):
        pi = next(i for i, item in enumerate(participants) if item["id"] == frame["participant"])
        ti = next(i for i, item in enumerate(times) if item["id"] == frame["time"])
        event = next(item for item in events if item["id"] == frame["event"])
        oi = operators.index(frame["operator"])
        slots = {
            "participant": participants[pi]["surface"],
            "time": times[ti]["surface"],
            "predicate": event["predicates"][frame["operator"]],
        }
        for variant, (source_template, target_template) in enumerate(
            zip(current["source_templates"], current["target_templates"], strict=True)
        ):
            row: Row = {
                "id": f"{group}-v{variant}",
                "group_id": group,
                "split": split,
                "inputs": {"context": "", "text": source_template.format(**slots)},
                "targets": {"text": target_template.format(**slots), "frame": deepcopy(frame)},
                "metadata": {
                    "variant": variant,
                    "support_class": support,
                    "generator_seed": current["generator_seed"],
                    "provenance": deepcopy(current["authorship"]),
                },
            }
            result[split].append(row)
    return result


def model_inputs(row: Mapping[str, Any]) -> Inputs:
    inputs = row.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_FIELDS):
        raise ValueError("inputs must contain only context and text")
    if not all(isinstance(inputs[field], str) for field in INPUT_FIELDS):
        raise ValueError("source context/text must be strings")
    return {"context": inputs["context"], "text": inputs["text"]}


def source_ids(row: Mapping[str, Any]) -> list[int]:
    raw = canonical(model_inputs(row)).encode("utf-8")
    return [BOS, *(value + BYTE_OFFSET for value in raw), SEP]


def teacher_forcing(row: Mapping[str, Any]) -> dict[str, Any]:
    source = source_ids(row)
    targets = row.get("targets")
    target = targets.get("text") if isinstance(targets, Mapping) else None
    if not isinstance(target, str) or not target:
        raise ValueError("a nonempty reference text is required")
    continuation = [*(value + BYTE_OFFSET for value in target.encode("utf-8")), EOS]
    full = source + continuation
    return {
        "source_ids": source,
        "input_ids": full[:-1],
        "labels": [-100] * (len(source) - 1) + continuation,
        "frame_position": len(source) - 1,
    }


def _target_lookup(spec: Mapping[str, Any]) -> dict[str, Frame]:
    participants, times, events, operators = _spec_lookup(spec)
    lookup: dict[str, Frame] = {}
    for participant in participants:
        for time in times:
            for event in events:
                for operator in operators:
                    frame: Frame = {
                        "participant": participant["id"], "time": time["id"],
                        "event": event["id"], "operator": operator,
                    }
                    slots = {
                        "participant": participant["surface"], "time": time["surface"],
                        "predicate": event["predicates"][operator],
                    }
                    for template in spec["target_templates"]:
                        rendered = template.format(**slots)
                        if rendered in lookup:
                            raise ValueError("fixture target grammar is ambiguous")
                        lookup[rendered] = deepcopy(frame)
    return lookup


def parse_target_text(text: str, spec: Mapping[str, Any] | None = None) -> Frame | None:
    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    frame = _target_lookup(current).get(text)
    return deepcopy(frame) if frame is not None else None


def train_support(bundle: Mapping[str, Sequence[Row]]) -> dict[str, set[tuple[str, ...]]]:
    frames = [row["targets"]["frame"] for row in bundle["train"]]
    return {
        "participant_time": {(f["participant"], f["time"]) for f in frames},
        "participant_time_event": {(f["participant"], f["time"], f["event"]) for f in frames},
    }


def _payload(rows: Sequence[Row]) -> bytes:
    return ("\n".join(canonical(row) for row in rows) + "\n").encode("utf-8")


def _frame_id(row: Row) -> str:
    return row["group_id"] + "-v0"


def _confirmation_frames(bundle: Mapping[str, Sequence[Row]], support: str) -> list[Row]:
    return [row for row in bundle["confirmation"] if row["metadata"]["support_class"] == support and row["metadata"]["variant"] == 0]


def _frame_only(row: Row) -> tuple[str, str, str, str]:
    frame = row["targets"]["frame"]
    return tuple(frame[field] for field in FIELDS)


def source_swap_pairs(bundle: Mapping[str, Sequence[Row]] | None = None) -> dict[str, dict[str, list[str]]]:
    """Return the lexicographically first deterministic one-factor swaps."""

    current = build() if bundle is None else bundle
    result: dict[str, dict[str, tuple[str, str]]] = {}
    for support in CONFIRMATION_SUPPORTS:
        rows = sorted(_confirmation_frames(current, support), key=lambda row: row["id"])
        by_factor: dict[str, list[str]] = {}
        for factor_index, factor in enumerate(FIELDS):
            candidates: list[tuple[str, str]] = []
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1:]:
                    left_frame, right_frame = _frame_only(left), _frame_only(right)
                    differences = [index for index, (a, b) in enumerate(zip(left_frame, right_frame)) if a != b]
                    if differences == [factor_index]:
                        candidates.append((_frame_id(left), _frame_id(right)))
            if not candidates:
                raise ValueError(f"no deterministic source swap for {support}/{factor}")
            by_factor[factor] = list(min(candidates))
        result[support] = by_factor
    return result


def probe_identities(bundle: Mapping[str, Sequence[Row]] | None = None) -> dict[str, dict[str, str]]:
    """Return eight fixed probe frame IDs, four factors in each support group."""

    current = build() if bundle is None else bundle
    result: dict[str, dict[str, str]] = {}
    for support in CONFIRMATION_SUPPORTS:
        rows = sorted(_confirmation_frames(current, support), key=lambda row: row["id"])
        if len(rows) < len(FIELDS):
            raise ValueError("confirmation support group cannot provide four probes")
        result[support] = {
            factor: _frame_id(rows[index]) for index, factor in enumerate(FIELDS)
        }
    return result


def _prior_surface_namespace_ok(text: str) -> bool:
    return text.startswith("h0-source-") or text.startswith("h0-target-")


def _prior_generated_surfaces() -> set[str]:
    """Render prior source specifications only, never consumed artifacts."""

    root = Path(__file__).resolve().parents[2]
    surfaces: set[str] = set()
    for relative in ("data/benchmark_v2/spec.json", "data/benchmark_v3_factor_path/spec.json"):
        try:
            prior = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("prior benchmark source specification is unreadable") from exc
        for participant in prior["participants"]:
            for time in prior["times"]:
                for event in prior["events"]:
                    for operator in prior["operators"]:
                        slots = {
                            "participant": participant["surface"],
                            "time": time["surface"],
                            "predicate": event["predicates"][operator],
                        }
                        for template in (*prior["source_templates"], *prior["target_templates"]):
                            surfaces.add(template.format(**slots))
    return surfaces


def _bundle_digest_source(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in (
        "schema", "version", "counts", "groups", "support_classes", "train_support",
        "split_plan", "selection", "leakage_contract", "spec_sha256", "generator_sha256",
        "sha256", "checks",
    )}


def validate(bundle: Mapping[str, Sequence[Row]], spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate row identity, support semantics, leakage, and frozen selections."""

    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    if set(bundle) != set(SPLITS):
        raise ValueError("unexpected H0 fixture split set")
    expected = build(current)
    if canonical(bundle) != canonical(expected):
        raise ValueError("bundle differs from the authored H0 generator result")
    ids: set[str] = set()
    groups: dict[str, str] = {}
    sources: set[str] = set()
    targets: set[str] = set()
    references: dict[str, str] = {}
    atoms = {field: set() for field in FIELDS}
    counts: Counter[str] = Counter()
    support_classes: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_lookup = _target_lookup(current)
    for split in SPLITS:
        for row in bundle[split]:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate row ID or split mismatch")
            ids.add(row["id"])
            if groups.setdefault(row["group_id"], split) != split:
                raise ValueError("surface variant group crosses splits")
            if row["metadata"]["variant"] not in (0, 1):
                raise ValueError("variant must be zero or one")
            source = canonical(model_inputs(row))
            if source in sources:
                raise ValueError("duplicate source surface")
            sources.add(source)
            source_text = row["inputs"]["text"]
            if not _prior_surface_namespace_ok(source_text):
                raise ValueError("source is outside the authored H0 namespace")
            reference = row["targets"]["text"]
            if reference in targets:
                raise ValueError("duplicate target surface")
            targets.add(reference)
            if source_text == reference:
                raise ValueError("source and target surface overlap")
            if references.setdefault(reference, split) != split:
                raise ValueError("target surface crosses splits")
            frame = row["targets"]["frame"]
            if set(frame) != set(FIELDS) or not all(isinstance(frame[field], str) for field in FIELDS):
                raise ValueError("invalid target frame")
            if target_lookup.get(reference) != frame:
                raise ValueError("target does not match authored grammar")
            if split == "train":
                for field in FIELDS:
                    atoms[field].add(frame[field])
            packed = teacher_forcing(row)
            if len(packed["input_ids"]) != len(packed["labels"]):
                raise ValueError("teacher forcing alignment error")
            counts[split] += 1
            support_classes[split][row["metadata"]["support_class"]] += 1
    if sources & targets:
        raise ValueError("source and target full surface sets overlap")
    prior_surfaces = _prior_generated_surfaces()
    if sources & prior_surfaces or targets & prior_surfaces:
        raise ValueError("H0 surfaces overlap generated prior v2/v3 surfaces")
    if len(bundle["train"]) != 384 or len(bundle["confirmation"]) != 96:
        raise ValueError("frozen H0 row counts changed")
    expected_atoms = {
        "participant": {item["id"] for item in current["participants"]},
        "time": {item["id"] for item in current["times"]},
        "event": {item["id"] for item in current["events"]},
        "operator": set(current["operators"]),
    }
    if atoms != expected_atoms:
        raise ValueError("not every factor atom is covered in train")
    support = train_support(bundle)
    for row in bundle["confirmation"]:
        frame = row["targets"]["frame"]
        pair = (frame["participant"], frame["time"])
        triple = (*pair, frame["event"])
        declared = row["metadata"]["support_class"]
        if declared == SUPPORT_UNSEEN_PAIR:
            if pair in support["participant_time"] or triple in support["participant_time_event"]:
                raise ValueError("unseen-pair confirmation is train-supported")
        elif declared == SUPPORT_SEEN_UNSEEN_TRIPLE:
            if pair not in support["participant_time"] or triple in support["participant_time_event"]:
                raise ValueError("seen-pair/unseen-triple confirmation has wrong support")
        else:
            raise ValueError("unknown confirmation support class")
    frame_groups = {row["group_id"] for split in SPLITS for row in bundle[split]}
    if len(frame_groups) != 240 or any(sum(row["group_id"] == group for split in SPLITS for row in bundle[split]) != 2 for group in frame_groups):
        raise ValueError("each authored frame must have exactly two variants")
    swaps = source_swap_pairs(bundle)
    probes = probe_identities(bundle)
    expected_manifest = globals().get("_EXPECTED_MANIFEST_CACHE")
    # The manifest is checked by check_expected; selection fields are still
    # validated here so a caller cannot silently reseat probes or swap pairs.
    if expected_manifest:
        if canonical(swaps) != canonical(expected_manifest["selection"]["source_swap_pairs"]):
            raise ValueError("source-swap selection was reseated")
        if canonical(probes) != canonical(expected_manifest["selection"]["probe_identities"]):
            raise ValueError("probe selection was reseated")
    hashes = {f"{split}.jsonl": sha256_bytes(_payload(bundle[split])) for split in SPLITS}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "counts": dict(counts),
        "groups": {split: sum(1 for group in groups.values() if group == split) for split in SPLITS},
        "support_classes": {split: dict(support_classes[split]) for split in SPLITS},
        "train_support": {name: len(values) for name, values in support.items()},
        "split_plan": split_plan(current),
        "selection": {"source_swap_pairs": swaps, "probe_identities": probes},
        "leakage_contract": {
            "model_inputs": list(INPUT_FIELDS),
            "encoding": "canonical UTF-8 bytes with BOS/SEP; no whole-surface categorical id",
            "forbidden_row_fields": ["id", "group_id", "split", "template", "support_class", "gold", "targets"],
            "prior_surface_source": ["data/benchmark_v2/spec.json", "data/benchmark_v3_factor_path/spec.json"],
            "consumed_artifacts_opened": False,
        },
        "spec_sha256": sha256_bytes(canonical_bytes(current) + b"\n"),
        "generator_sha256": sha256_bytes(normalized_source_bytes()),
        "sha256": hashes,
        "checks": {
            "unique_ids": True, "unique_frame_groups": True, "unique_sources": True,
            "unique_targets": True, "source_target_disjoint": True,
            "prior_v2_v3_evaluation_surfaces_disjoint": True,
            "train_atomic_coverage": True, "confirmation_support_semantics": True,
            "source_swap_pairs_complete": True, "probe_identities_complete": True,
            "selection_not_reseated": True, "two_variants_per_frame": True,
            "teacher_forcing_alignment": True, "source_allowlist": True,
            "target_grammar_unambiguous": True, "no_full_surface_categorical_id": True,
        },
        "scope": "future-only authored H0 confirmation fixture; no trained-model or linguistic-quality claim",
    }
    report["content_digest_sha256"] = sha256_bytes(canonical_bytes(_bundle_digest_source(report)))
    return report


def expected_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("H0 fixture expected manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError("expected manifest must be a JSON object")
    return value


def check_expected(report: Mapping[str, Any]) -> None:
    expected = expected_manifest()
    if canonical(dict(report)) != canonical(expected):
        raise ValueError("generated H0 fixture does not match expected-manifest.json")


def confirmation_rows(bundle: Mapping[str, Sequence[Row]] | None = None, *, manifest_digest: str | None = None) -> list[Row]:
    """Expose confirmation rows only when the canonical manifest is supplied."""

    current = build() if bundle is None else bundle
    report = validate(current)
    check_expected(report)
    if manifest_digest != report["content_digest_sha256"]:
        raise ValueError("confirmation manifest digest mismatch")
    return deepcopy(list(current["confirmation"]))


# Populated only after the tracked manifest is read.  It lets validate reject
# a row bundle whose deterministic selections changed while remaining safe for
# temporary test bundles before a manifest exists.
try:
    _EXPECTED_MANIFEST_CACHE: dict[str, Any] = expected_manifest()
except ValueError:
    _EXPECTED_MANIFEST_CACHE = {}


__all__ = [
    "BenchmarkSpec", "CorpusRow", "CONFIRMATION_SUPPORTS", "FIELDS", "Frame",
    "GENERATOR_SEED", "INPUT_FIELDS", "Inputs", "MANIFEST_PATH", "Metadata",
    "Row", "SCHEMA", "SPLITS", "SPEC_PATH", "SPLIT_SEED", "SUPPORT_SEEN_UNSEEN_TRIPLE",
    "SUPPORT_TRAIN", "SUPPORT_UNSEEN_PAIR", "Targets", "build", "canonical",
    "canonical_bytes", "check_expected", "confirmation_rows", "expected_manifest",
    "model_inputs", "normalized_source_bytes", "parse_target_text", "probe_identities",
    "sha256_bytes", "source_ids", "source_swap_pairs", "spec_data", "split_plan",
    "teacher_forcing", "train_support", "validate", "validate_spec",
]
