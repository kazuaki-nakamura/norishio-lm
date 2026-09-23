"""Deterministic, authored fixture for the v3 head-learning comparison.

The fixture is deliberately a structural corpus.  It supplies a balanced
training graph for a JOINT versus FACTOR_ONLY comparison and two confirmation
strata; it does not train a model, run inference, or make a linguistic claim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias, TypedDict


VERSION = "norishio-benchmark-v3-head-learning.v1"
SCHEMA = "norishio.benchmark-v3-head-learning-fixture.v1"
SPLITS = ("train", "confirmation")
FIELDS = ("participant", "time", "event", "operator")
INPUT_FIELDS = ("context", "text")
SUPPORT_TRAIN = "train"
SUPPORT_UNSEEN_PAIR = "unseen_pair"
SUPPORT_SEEN_UNSEEN_TRIPLE = "seen_pair/unseen_triple"
CONFIRMATION_SUPPORTS = (SUPPORT_UNSEEN_PAIR, SUPPORT_SEEN_UNSEEN_TRIPLE)
VARIANTS = (0, 1)
BYTE_OFFSET = 4
BOS, EOS, SEP = 1, 2, 3
GENERATOR_SEED = 20260923
SPLIT_SEED = 46017

SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "benchmark_v3_head_learning"
    / "spec.json"
)
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


def canonical(value: Any) -> str:
    """Return the canonical JSON representation used by all digests."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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
        "source": "data/benchmark_v3_head_learning/spec.json",
        "human_verified": False,
        "private_data_used": False,
        "derived_from": "new_head_learning_namespace_only; prior_v2_v3_artifacts_unopened",
    }


def spec_data() -> dict[str, Any]:
    """Load and validate the tracked authored specification."""

    try:
        value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("head-learning fixture specification is unreadable") from exc
    validate_spec(value)
    return deepcopy(value)


def validate_spec(spec: Mapping[str, Any]) -> None:
    required = {
        "version", "generator_seed", "split_seed", "authorship", "participants",
        "times", "events", "operators", "source_templates", "target_templates",
        "selection",
    }
    if set(spec) != required or spec.get("version") != VERSION:
        raise ValueError("unsupported or incomplete head-learning specification")
    if spec.get("generator_seed") != GENERATOR_SEED or not isinstance(spec.get("split_seed"), int):
        raise ValueError("fixture seeds changed")
    if spec["authorship"] != _authorship():
        raise ValueError("fixture authorship/provenance changed")
    for name, count in (("participants", 6), ("times", 6)):
        values = spec[name]
        if len(values) != count:
            raise ValueError(f"fixture requires {count} {name}")
        if any(set(item) != {"id", "surface"} for item in values):
            raise ValueError(f"invalid {name} entry")
        if len({item["id"] for item in values}) != count or len({item["surface"] for item in values}) != count:
            raise ValueError(f"duplicate {name} value")
        if not all(isinstance(item["id"], str) and isinstance(item["surface"], str) and item["surface"] for item in values):
            raise ValueError(f"invalid {name} value")
    operators = spec["operators"]
    if len(operators) != 4 or len(set(operators)) != 4 or not all(isinstance(value, str) and value for value in operators):
        raise ValueError("operators must be four unique strings")
    events = spec["events"]
    if len(events) != 4 or len({event["id"] for event in events}) != 4:
        raise ValueError("events must be four unique values")
    for event in events:
        if set(event) != {"id", "predicates"} or set(event["predicates"]) != set(operators):
            raise ValueError("each event requires one predicate per operator")
        if not all(isinstance(value, str) and value for value in event["predicates"].values()):
            raise ValueError("event predicates must be nonempty strings")
    for template in (*spec["source_templates"], *spec["target_templates"]):
        if not all(slot in template for slot in ("{participant}", "{time}", "{predicate}")):
            raise ValueError("templates must expose all factor slots")
    if len(spec["source_templates"]) != 2 or len(spec["target_templates"]) != 2:
        raise ValueError("fixture freezes exactly two variants")
    selection = spec["selection"]
    expected_selection = {
        "unseen_pair_offsets", "unused_pair_offsets", "train_pair_offsets",
        "train_events_by_offset", "unseen_pair_events", "unseen_pair_operators",
        "seen_pair_offset", "seen_pair_events", "seen_pair_operators", "variant_count",
    }
    if set(selection) != expected_selection or selection["variant_count"] != 2:
        raise ValueError("fixture selection plan changed")
    offsets = {
        "unseen_pair_offsets": selection["unseen_pair_offsets"],
        "unused_pair_offsets": selection["unused_pair_offsets"],
        "train_pair_offsets": selection["train_pair_offsets"],
    }
    if offsets != {"unseen_pair_offsets": [0], "unused_pair_offsets": [1], "train_pair_offsets": [2, 3, 4, 5]}:
        raise ValueError("fixture pair graph changed")
    if selection["seen_pair_offset"] != 2:
        raise ValueError("fixture seen-pair stratum changed")
    expected_events = {"2": ["E0", "E1"], "3": ["E1", "E2"], "4": ["E2", "E3"], "5": ["E3", "E0"]}
    if selection["train_events_by_offset"] != expected_events:
        raise ValueError("fixture event counterbalance changed")
    if selection["unseen_pair_events"] != ["E0", "E1"] or selection["seen_pair_events"] != ["E2", "E3"]:
        raise ValueError("fixture confirmation events changed")
    if selection["unseen_pair_operators"] != ["O0", "O1"] or selection["seen_pair_operators"] != ["O2", "O3"]:
        raise ValueError("fixture confirmation operators changed")


def _lookups(spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    return list(spec["participants"]), list(spec["times"]), list(spec["events"]), list(spec["operators"])


def split_plan(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    selection = current["selection"]
    return {
        "name": "head-learning-regular-bipartite-graph-v1",
        "generator_seed": current["generator_seed"],
        "seed": current["split_seed"],
        "unseen_pair_offsets": list(selection["unseen_pair_offsets"]),
        "unused_pair_offsets": list(selection["unused_pair_offsets"]),
        "train_pair_offsets": list(selection["train_pair_offsets"]),
        "train_pair_count": 24,
        "confirmation_pair_count_by_support": {SUPPORT_UNSEEN_PAIR: 6, SUPPORT_SEEN_UNSEEN_TRIPLE: 6},
        "graph_definition": "offset=(time_index-participant_index) mod 6; offsets 0 and 1 are held out, offsets 2..5 train",
    }


def _pair_rows(spec: Mapping[str, Any]) -> list[tuple[int, int, tuple[str, ...], tuple[str, ...], str, str]]:
    selection = spec["selection"]
    rows: list[tuple[int, int, tuple[str, ...], tuple[str, ...], str, str]] = []
    for pi in range(6):
        for ti in range(6):
            offset = (ti - pi) % 6
            if offset in selection["train_pair_offsets"]:
                events = tuple(selection["train_events_by_offset"][str(offset)])
                rows.append((pi, ti, events, tuple(spec["operators"]), SUPPORT_TRAIN, "train"))
                if offset == selection["seen_pair_offset"]:
                    rows.append((pi, ti, tuple(selection["seen_pair_events"]), tuple(selection["seen_pair_operators"]), SUPPORT_SEEN_UNSEEN_TRIPLE, "confirmation"))
            elif offset in selection["unseen_pair_offsets"]:
                rows.append((pi, ti, tuple(selection["unseen_pair_events"]), tuple(selection["unseen_pair_operators"]), SUPPORT_UNSEEN_PAIR, "confirmation"))
    return rows


def _target_lookup(spec: Mapping[str, Any]) -> dict[str, Frame]:
    participants, times, events, operators = _lookups(spec)
    result: dict[str, Frame] = {}
    for participant in participants:
        for time in times:
            for event in events:
                for operator in operators:
                    slots = {"participant": participant["surface"], "time": time["surface"], "predicate": event["predicates"][operator]}
                    frame: Frame = {"participant": participant["id"], "time": time["id"], "event": event["id"], "operator": operator}
                    for template in spec["target_templates"]:
                        text = template.format(**slots)
                        if text in result:
                            raise ValueError("fixture target grammar is ambiguous")
                        result[text] = deepcopy(frame)
    return result


def build(spec: Mapping[str, Any] | None = None) -> dict[str, list[Row]]:
    """Build the complete authored fixture in deterministic row order."""

    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    participants, times, events, operators = _lookups(current)
    p_by_id = {item["id"]: item for item in participants}
    t_by_id = {item["id"]: item for item in times}
    e_by_id = {item["id"]: item for item in events}
    result: dict[str, list[Row]] = {split: [] for split in SPLITS}
    for pi, ti, event_ids, operator_ids, support, split in _pair_rows(current):
        for event_id in event_ids:
            for operator_id in operator_ids:
                frame: Frame = {"participant": participants[pi]["id"], "time": times[ti]["id"], "event": event_id, "operator": operator_id}
                slots = {"participant": participants[pi]["surface"], "time": times[ti]["surface"], "predicate": e_by_id[event_id]["predicates"][operator_id]}
                for variant, (source_template, target_template) in enumerate(zip(current["source_templates"], current["target_templates"], strict=True)):
                    group_id = f"head-{support.replace('/', '-')}-p{pi}-t{ti}-e{event_id[1:]}-o{operator_id[1:]}"
                    result[split].append({
                        "id": f"{group_id}-v{variant}",
                        "group_id": group_id,
                        "split": split,
                        "inputs": {"context": "", "text": source_template.format(**slots)},
                        "targets": {"text": target_template.format(**slots), "frame": deepcopy(frame)},
                        "metadata": {"variant": variant, "support_class": support, "generator_seed": current["generator_seed"], "provenance": deepcopy(current["authorship"])},
                    })
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


def parse_target_text(text: str, spec: Mapping[str, Any] | None = None) -> Frame | None:
    if not isinstance(text, str):
        raise TypeError("generated text must be a string")
    current = spec_data() if spec is None else dict(spec)
    validate_spec(current)
    return deepcopy(_target_lookup(current).get(text))


def train_support(bundle: Mapping[str, Sequence[Row]]) -> dict[str, set[tuple[str, ...]]]:
    frames = [row["targets"]["frame"] for row in bundle["train"] if row["metadata"]["variant"] == 0]
    return {
        "participant_time": {(f["participant"], f["time"]) for f in frames},
        "participant_time_event": {(f["participant"], f["time"], f["event"]) for f in frames},
        "participant_time_predicate": {(f["participant"], f["time"], f["event"], f["operator"]) for f in frames},
    }


def _prior_surfaces() -> set[str]:
    root = Path(__file__).resolve().parents[2]
    surfaces: set[str] = set()
    for relative in ("data/benchmark_v2/spec.json", "data/benchmark_v3_factor_path/spec.json"):
        try:
            prior = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("prior benchmark specification is unreadable") from exc
        for participant in prior["participants"]:
            for time in prior["times"]:
                for event in prior["events"]:
                    for operator in prior["operators"]:
                        predicate = event["predicates"][operator]
                        slots = {"participant": participant["surface"], "time": time["surface"], "predicate": predicate}
                        for template in (*prior["source_templates"], *prior["target_templates"]):
                            surfaces.add(template.format(**slots))
    return surfaces


def _all_surfaces(spec: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    current = build(spec)
    sources = {row["inputs"]["text"] for rows in current.values() for row in rows}
    targets = {row["targets"]["text"] for rows in current.values() for row in rows}
    return sources, targets


def _payload(rows: Sequence[Row]) -> bytes:
    return ("\n".join(canonical(row) for row in rows) + "\n").encode("utf-8")


def validate(bundle: Mapping[str, Sequence[Row]], spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = spec_data() if spec is None else deepcopy(dict(spec))
    validate_spec(current)
    if set(bundle) != set(SPLITS) or canonical(bundle) != canonical(build(current)):
        raise ValueError("bundle differs from the frozen head-learning generator result")
    expected_atoms = {"participant": {x["id"] for x in current["participants"]}, "time": {x["id"] for x in current["times"]}, "event": {x["id"] for x in current["events"]}, "operator": set(current["operators"])}
    ids: set[str] = set(); groups: dict[str, list[Row]] = {}; sources: set[str] = set(); targets: set[str] = set()
    support_classes = {split: Counter() for split in SPLITS}; atoms = {field: set() for field in FIELDS}
    train_frames: list[Frame] = []
    target_lookup = _target_lookup(current)
    for split in SPLITS:
        for row in bundle[split]:
            if row["id"] in ids or row["split"] != split:
                raise ValueError("duplicate id or split mismatch")
            ids.add(row["id"]); groups.setdefault(row["group_id"], []).append(row)
            if row["metadata"]["variant"] not in VARIANTS or row["metadata"]["support_class"] not in (SUPPORT_TRAIN, *CONFIRMATION_SUPPORTS):
                raise ValueError("invalid row metadata")
            source = canonical(model_inputs(row));
            if source in sources: raise ValueError("duplicate source")
            sources.add(source)
            target = row["targets"]["text"]; frame = row["targets"]["frame"]
            if target in targets or row["inputs"]["text"] == target or target_lookup.get(target) != frame:
                raise ValueError("duplicate or invalid target")
            targets.add(target); support_classes[split][row["metadata"]["support_class"]] += 1
            if split == "train":
                for field in FIELDS: atoms[field].add(frame[field])
                if row["metadata"]["variant"] == 0: train_frames.append(frame)
            source_ids(row)
    if any(len(rows) != 2 or len({row["split"] for row in rows}) != 1 for rows in groups.values()):
        raise ValueError("variant groups must contain exactly two rows in one split")
    if not sources.isdisjoint(targets): raise ValueError("source and target surface namespaces overlap")
    if atoms != expected_atoms: raise ValueError("not every atomic value is supported by train")
    support = train_support(bundle)
    for row in bundle["confirmation"]:
        if row["metadata"]["variant"] != 0: continue
        frame = row["targets"]["frame"]; pair = (frame["participant"], frame["time"]); triple = (*pair, frame["event"])
        expected_support = SUPPORT_UNSEEN_PAIR if pair not in support["participant_time"] else SUPPORT_SEEN_UNSEEN_TRIPLE
        if row["metadata"]["support_class"] != expected_support or (expected_support == SUPPORT_SEEN_UNSEEN_TRIPLE and triple in support["participant_time_event"]):
            raise ValueError("confirmation support class is incorrect")
    counts = {split: len(bundle[split]) for split in SPLITS}
    groups_count = {split: sum(1 for rows in groups.values() if rows[0]["split"] == split) for split in SPLITS}
    train_pair_offsets = Counter((int(frame["time"][1:]) - int(frame["participant"][1:])) % 6 for frame in train_frames)
    participant_event = Counter((frame["participant"], frame["event"]) for frame in train_frames)
    time_event = Counter((frame["time"], frame["event"]) for frame in train_frames)
    predicate_counts = Counter((frame["participant"], frame["event"], frame["operator"]) for frame in train_frames)
    time_predicate_counts = Counter((frame["time"], frame["event"], frame["operator"]) for frame in train_frames)
    balanced = (
        all(set(counter.values()) == {8} for counter in (participant_event, time_event))
        and all(set(counter.values()) == {2} for counter in (predicate_counts, time_predicate_counts))
    )
    if not balanced: raise ValueError("train event/operator exposure is not counterbalanced")
    prior = _prior_surfaces()
    if (sources | targets) & prior: raise ValueError("head-learning surfaces reuse a prior namespace")
    report: dict[str, Any] = {
        "version": VERSION,
        "schema": SCHEMA,
        "counts": counts,
        "groups": groups_count,
        "support_classes": {split: dict(support_classes[split]) for split in SPLITS},
        "train_support": {name: len(values) for name, values in support.items()},
        "split_plan": split_plan(current),
        "train_pair_offset_counts": {str(key): value for key, value in sorted(train_pair_offsets.items())},
        "balance": {"participant_event_count": 8, "time_event_count": 8, "participant_predicate_count": 2, "time_predicate_count": 2, "all_equal": True},
        "spec_sha256": sha256_bytes(canonical_bytes(current) + b"\n"),
        "generator_sha256": sha256_bytes(normalized_source_bytes()),
        "sha256": {f"{split}.jsonl": sha256_bytes(_payload(bundle[split])) for split in SPLITS},
        "checks": {"unique_ids": True, "variant_groups": True, "unique_sources": True, "unique_targets": True, "source_target_disjoint": True, "train_atomic_coverage": True, "explicit_train_graph": True, "pair_and_triple_holdouts": True, "counterbalanced_predicates": True, "prior_surfaces_disjoint": True, "no_model_execution": True},
        "scope": "future-only authored structural fixture; no trained-model or linguistic-quality claim",
    }
    digest_payload = dict(report)
    report["content_digest_sha256"] = sha256_bytes(canonical_bytes(digest_payload))
    return report


def expected_manifest() -> dict[str, Any]:
    try:
        value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("expected head-learning manifest is unreadable") from exc
    if not isinstance(value, dict): raise ValueError("expected manifest must be an object")
    return value


def load(path: str | Path | None = None) -> dict[str, list[Row]]:
    """Load an exported fixture, or build the canonical fixture when omitted."""

    if path is None:
        return build()
    root = Path(path)
    bundle: dict[str, list[Row]] = {}
    try:
        for split in SPLITS:
            rows = []
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                if line:
                    rows.append(json.loads(line))
            bundle[split] = rows
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("exported head-learning fixture is unreadable") from exc
    validate(bundle)
    return bundle


def manifest(bundle: Mapping[str, Sequence[Row]] | None = None) -> dict[str, Any]:
    """Return the frozen manifest, optionally after validating a bundle."""

    if bundle is None:
        return expected_manifest()
    report = validate(bundle)
    check_expected(report)
    return report


def check_expected(report: Mapping[str, Any]) -> None:
    if canonical(dict(report)) != canonical(expected_manifest()):
        raise ValueError("generated head-learning fixture does not match expected-manifest.json")


def manifest_digest(bundle: Mapping[str, Sequence[Row]] | None = None) -> str:
    report = validate(build() if bundle is None else bundle)
    check_expected(report)
    return report["content_digest_sha256"]


def write_bundle(out: str | Path) -> dict[str, Any]:
    target = Path(out)
    if target.exists(): raise FileExistsError(f"output already exists: {target}")
    bundle = build(); report = validate(bundle); check_expected(report); target.mkdir(parents=True)
    for split in SPLITS: (target / f"{split}.jsonl").write_bytes(_payload(bundle[split]))
    (target / "manifest.json").write_bytes((canonical(report) + "\n").encode("utf-8"))
    return report


__all__ = [
    "CONFIRMATION_SUPPORTS", "FIELDS", "Frame", "GENERATOR_SEED", "INPUT_FIELDS",
    "MANIFEST_PATH", "Row", "SCHEMA", "SPLITS", "SPEC_PATH", "SPLIT_SEED",
    "SUPPORT_SEEN_UNSEEN_TRIPLE", "SUPPORT_TRAIN", "SUPPORT_UNSEEN_PAIR", "VERSION",
    "build", "canonical", "canonical_bytes", "check_expected", "expected_manifest",
    "load", "manifest", "manifest_digest", "model_inputs", "normalized_source_bytes", "parse_target_text",
    "sha256_bytes", "source_ids", "spec_data", "split_plan", "train_support", "validate",
    "validate_spec", "write_bundle",
]
