"""Evaluation-only source swaps and probability interventions for Issue #44.

This module deliberately owns no training or checkpoint loading.  A caller
supplies an already constructed :class:`FutureH0PathModel` and a descriptor
which has been validated against the authored H0 fixture.  The evaluator
retains one causal BOS baseline per probe, then applies the three frozen
controls without reseating donors or recomputing a baseline after an
intervention.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import torch
from torch import Tensor

from . import benchmark_v3_h0_fixture as fixture
from .benchmark_v3_fsm import ByteTag, FrozenLocalPrefixFSM
from .benchmark_v3_h0_metrics import score_control_record
from .benchmark_v3_h0_model import FutureH0PathModel
from .benchmark_v3_model import BOS_ID, BYTE_OFFSET, EOS_ID, FACTOR_ORDER, FACTOR_SIZES, PAD_ID, VOCAB_SIZE
from .benchmark_v3_runner import MAX_NEW_TOKENS
from .benchmark_v3_h0_runner import load_h0_confirmation_rows, load_h0_train_rows


SCHEMA = "norishio.issue44.h0-intervention-descriptor.v1"
RESULT_SCHEMA = "norishio.issue44.h0-intervention-result.v1"
CANONICAL_PROTOCOL_SCHEMA = "norishio.issue44.h0-confirmation-protocol.v1"
PRIMARY_PREFIX_STRATUM = "bos_start_primary"
CONTROL_TYPES = (
    "same_class_soft_shape",
    "alternate_class_one_hot",
    "alternate_class_donor_soft",
)

ProbabilityMap: TypeAlias = dict[str, list[float]]
Frame: TypeAlias = dict[str, str]
JsonObject: TypeAlias = dict[str, Any]


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _manifest() -> dict[str, Any]:
    bundle = fixture.build()
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    return report


def _vocabulary() -> dict[str, tuple[str, ...]]:
    spec = fixture.spec_data()
    result = {
        "participant": tuple(item["id"] for item in spec["participants"]),
        "time": tuple(item["id"] for item in spec["times"]),
        "event": tuple(item["id"] for item in spec["events"]),
        "operator": tuple(spec["operators"]),
    }
    if {field: len(result[field]) for field in FACTOR_ORDER} != FACTOR_SIZES:
        raise ValueError("H0 factor vocabulary changed")
    return result


def _frame(row: Mapping[str, Any]) -> Frame:
    targets = row.get("targets")
    if not isinstance(targets, Mapping) or not isinstance(targets.get("frame"), Mapping):
        raise ValueError("fixture row is missing a target frame")
    raw = targets["frame"]
    if set(raw) != set(FACTOR_ORDER):
        raise ValueError("fixture target frame fields changed")
    return {field: str(raw[field]) for field in FACTOR_ORDER}


def _row_text(row: Mapping[str, Any]) -> str:
    targets = row.get("targets")
    text = targets.get("text") if isinstance(targets, Mapping) else None
    if not isinstance(text, str) or not text:
        raise ValueError("fixture row is missing target text")
    return text


def _factor_class(frame: Mapping[str, str], factor: str, vocabulary: Mapping[str, Sequence[str]]) -> int:
    try:
        return list(vocabulary[factor]).index(frame[factor])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown {factor} value in fixture frame") from exc


def _candidate_for_text(machine: FrozenLocalPrefixFSM, text: str):
    candidates = tuple(candidate for candidate in machine.candidates if candidate.text == text)
    if len(candidates) != 1:
        raise ValueError("authored target text must resolve to exactly one grammar candidate")
    return candidates[0]


def _slot_bounds(candidate: Any, factor: str) -> tuple[int, int]:
    if factor in ("event", "operator"):
        wanted = ByteTag.PREDICATE
    elif factor == "participant":
        wanted = ByteTag.PARTICIPANT
    elif factor == "time":
        wanted = ByteTag.TIME
    else:
        raise ValueError(f"unknown factor {factor!r}")
    positions = [index for index, tag in enumerate(candidate.tags) if tag == wanted]
    if not positions:
        raise ValueError(f"target candidate has no {factor} byte slot")
    return min(positions), max(positions) + 1


def _expected_trace(machine: FrozenLocalPrefixFSM, text: str, factor: str) -> tuple[list[list[bool]], list[bool], bool, tuple[int, int]]:
    candidate = _candidate_for_text(machine, text)
    schedule = [
        list(machine.gates([BOS_ID, *[BYTE_OFFSET + value for value in candidate.byte_values[:position]]]))
        for position in range(len(candidate.byte_values))
    ]
    factor_index = FACTOR_ORDER.index(factor)
    mask = [step[factor_index] for step in schedule]
    slot_start, slot_end = _slot_bounds(candidate, factor)
    reachable = any(mask[position] and position < slot_end for position in range(len(mask)))
    if not reachable:
        raise ValueError("primary probe target gate is unreachable after BOS")
    return schedule, mask, reachable, (slot_start, slot_end)


def _source_ids(row: Mapping[str, Any]) -> list[int]:
    return list(fixture.source_ids(row))


def _source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    source_ids = _source_ids(row)
    return {
        "row_id": str(row["id"]),
        "source_ids": source_ids,
        "source_sha256": _digest(source_ids),
    }


def _payload_without_digest(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in descriptor.items() if key != "descriptor_sha256"}


def _tracked_canonical_descriptor() -> tuple[dict[str, Any], str]:
    """Read the published descriptor without importing its protocol module."""

    path = Path(__file__).resolve().parents[2] / "data" / "benchmark_v3_h0_confirmation" / "experiment-descriptor-v1.json"
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("canonical Issue #44 descriptor is unreadable") from exc
    if not isinstance(wrapper, Mapping) or set(wrapper) != {"schema", "descriptor_sha256", "descriptor"}:
        raise ValueError("canonical Issue #44 descriptor wrapper fields changed")
    descriptor = wrapper["descriptor"]
    digest = wrapper["descriptor_sha256"]
    if not isinstance(descriptor, Mapping) or not isinstance(digest, str) or _digest(descriptor) != digest:
        raise ValueError("canonical Issue #44 descriptor digest mismatch")
    return deepcopy(dict(descriptor)), digest


def _adapt_canonical_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the published protocol shape to the evaluator's compact shape."""

    canonical, digest = _tracked_canonical_descriptor()
    if dict(descriptor) != canonical:
        raise ValueError("canonical Issue #44 descriptor differs from the tracked freeze")
    compact = build_h0_intervention_descriptor()
    probes = canonical["probes"]["internal_probe_identities"]
    compact["source_swap_pairs"] = deepcopy(canonical["probes"]["source_swap_pairs"])
    compact["probe_identities"] = deepcopy(probes)
    compact["expected_probes"] = {support: {} for support in fixture.CONFIRMATION_SUPPORTS}
    for item in canonical["fsm"]["primary_probe_records"]:
        support = item["support_group"]
        factor = item["target_factor"]
        compact["expected_probes"][support][factor] = {
            "row_id": item["row_id"],
            "support_class": support,
            "target_frame": deepcopy(item["target_frame"]),
            "target_text": item["text"],
            "expected_target_gate_schedule": deepcopy(item["expected_target_gate_schedule"]),
            "expected_target_gate_mask": deepcopy(item["expected_target_gate_mask"]),
            "target_gate_reachable_after_prefix": bool(item["target_gate_reachable_after_prefix"]),
            "target_slot_bounds": [int(item["slot_start"]), int(item["slot_end"])],
        }
    vocabulary = _vocabulary()
    compact["probe_donor_tables"] = {}
    for support in fixture.CONFIRMATION_SUPPORTS:
        compact["probe_donor_tables"][support] = {}
        for probe_id in probes[support].values():
            compact["probe_donor_tables"][support][probe_id] = {}
            source = canonical["donors"]["table"][support][probe_id]
            for factor in FACTOR_ORDER:
                table: dict[str, str] = {}
                for authored_class, donor in source[factor].items():
                    class_index = vocabulary[factor].index(authored_class)
                    table[str(class_index)] = str(donor["row_id"])
                compact["probe_donor_tables"][support][probe_id][factor] = table
    # The flat table remains useful in records; per-probe tables drive donor
    # selection because the canonical freeze binds donors to each probe.
    first_support = fixture.CONFIRMATION_SUPPORTS[0]
    first_probe = next(iter(probes[first_support].values()))
    compact["donor_table"] = deepcopy(compact["probe_donor_tables"][first_support][first_probe])
    compact["descriptor_sha256"] = digest
    compact["canonical_descriptor_sha256"] = digest
    compact["canonical_descriptor"] = True
    return compact


def _donor_table(train_rows: Sequence[Mapping[str, Any]], vocabulary: Mapping[str, Sequence[str]]) -> dict[str, dict[str, str]]:
    rows = sorted(
        (row for row in train_rows if row.get("metadata", {}).get("variant") == 0),
        key=lambda row: str(row.get("id", "")),
    )
    table: dict[str, dict[str, str]] = {}
    for factor in FACTOR_ORDER:
        table[factor] = {}
        for class_index, value in enumerate(vocabulary[factor]):
            matches = [row for row in rows if _frame(row)[factor] == value]
            if not matches:
                raise ValueError(f"no train donor for {factor} class {class_index}")
            table[factor][str(class_index)] = str(matches[0]["id"])
    return table


def build_h0_intervention_descriptor() -> dict[str, Any]:
    """Build the deterministic, future-only descriptor used by the engine."""

    bundle = fixture.build()
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    machine = FrozenLocalPrefixFSM(fixture.spec_data())
    vocabulary = _vocabulary()
    confirmation = {str(row["id"]): row for row in bundle["confirmation"] if row.get("metadata", {}).get("variant") == 0}
    probes = fixture.probe_identities(bundle)
    expected: dict[str, dict[str, Any]] = {}
    for support, factor_rows in probes.items():
        expected[support] = {}
        for factor, row_id in factor_rows.items():
            row = confirmation[row_id]
            schedule, mask, reachable, bounds = _expected_trace(machine, _row_text(row), factor)
            expected[support][factor] = {
                "row_id": row_id,
                "support_class": support,
                "target_frame": _frame(row),
                "target_text": _row_text(row),
                "expected_target_gate_schedule": schedule,
                "expected_target_gate_mask": mask,
                "target_gate_reachable_after_prefix": reachable,
                "target_slot_bounds": list(bounds),
            }
    donor = _donor_table(bundle["train"], vocabulary)
    # Keep a per-probe binding as well as the compact table.  The duplicated
    # binding makes the evaluator's no-reseating contract inspectable per row.
    donor_bindings = {
        support: {factor: dict(donor[factor]) for factor in FACTOR_ORDER}
        for support in fixture.CONFIRMATION_SUPPORTS
    }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "fixture_manifest_sha256": report["content_digest_sha256"],
        "prefix_stratum": PRIMARY_PREFIX_STRATUM,
        "prefix_token_ids": [BOS_ID],
        "source_swap_pairs": fixture.source_swap_pairs(bundle),
        "probe_identities": probes,
        "expected_probes": expected,
        "donor_table": donor,
        "probe_donor_bindings": donor_bindings,
    }
    payload["descriptor_sha256"] = _digest(payload)
    return payload


def validate_h0_intervention_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Validate descriptor identity, digest, probes, donors, and FSM traces."""

    if not isinstance(descriptor, Mapping):
        raise TypeError("intervention descriptor must be a mapping")
    if descriptor.get("schema") == CANONICAL_PROTOCOL_SCHEMA:
        return _adapt_canonical_descriptor(descriptor)
    value = deepcopy(dict(descriptor))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported H0 intervention descriptor schema")
    digest = value.get("descriptor_sha256")
    if not isinstance(digest, str) or digest != _digest(_payload_without_digest(value)):
        raise ValueError("intervention descriptor digest mismatch")
    expected = build_h0_intervention_descriptor()
    if value != expected:
        raise ValueError("intervention descriptor differs from the frozen authored selection")
    if value["fixture_manifest_sha256"] != _manifest()["content_digest_sha256"]:
        raise ValueError("fixture manifest digest mismatch")
    if value["prefix_stratum"] != PRIMARY_PREFIX_STRATUM or value["prefix_token_ids"] != [BOS_ID]:
        raise ValueError("primary intervention prefix is not the frozen BOS prefix")
    machine = FrozenLocalPrefixFSM(fixture.spec_data())
    for support, factors in value["expected_probes"].items():
        for factor, item in factors.items():
            schedule, mask, reachable, bounds = _expected_trace(machine, item["target_text"], factor)
            if item["expected_target_gate_schedule"] != schedule or item["expected_target_gate_mask"] != mask:
                raise ValueError("descriptor expected gate trace differs from the authored target")
            if item["target_gate_reachable_after_prefix"] is not reachable or item["target_slot_bounds"] != list(bounds):
                raise ValueError("descriptor target gate reachability differs from the authored target")
    return value


def _source_batch(source_rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    if not source_rows:
        raise ValueError("source rows cannot be empty")
    width = max(len(row) for row in source_rows)
    source = torch.full((len(source_rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(source_rows), width), dtype=torch.bool)
    for index, values in enumerate(source_rows):
        if not values or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in values):
            raise ValueError("source row contains an invalid token")
        source[index, :len(values)] = torch.tensor(list(values), dtype=torch.long)
        mask[index, :len(values)] = True
    return source, mask


def _probability_map(probabilities: Mapping[str, Tensor], index: int = 0) -> ProbabilityMap:
    if set(probabilities) != set(FACTOR_ORDER):
        raise ValueError("factor map must contain exactly four factors")
    return {field: [float(x) for x in probabilities[field][index].detach().cpu().tolist()] for field in FACTOR_ORDER}


def _argmax_map(probabilities: Mapping[str, Sequence[float]]) -> dict[str, int]:
    return {field: max(range(len(probabilities[field])), key=lambda index: probabilities[field][index]) for field in FACTOR_ORDER}


def _one_hot(width: int, index: int) -> list[float]:
    if not 0 <= index < width:
        raise ValueError("one-hot class is out of range")
    return [1.0 if position == index else 0.0 for position in range(width)]


def _replacement_map(baseline: ProbabilityMap, factor: str, replacement: Sequence[float]) -> ProbabilityMap:
    result = {field: list(values) for field, values in baseline.items()}
    result[factor] = [float(value) for value in replacement]
    return result


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _decode(tokens: Sequence[int]) -> dict[str, Any]:
    values = list(tokens)
    ended_eos = bool(values and values[-1] == EOS_ID)
    content = values[:-1] if ended_eos else values
    invalid = [token for token in content if token < BYTE_OFFSET or token >= VOCAB_SIZE]
    raw = bytes(token - BYTE_OFFSET for token in content if BYTE_OFFSET <= token < VOCAB_SIZE)
    try:
        text = raw.decode("utf-8") if not invalid else None
    except UnicodeDecodeError:
        text = None
    return {
        "text": text,
        "tokens": values,
        "ended_eos": ended_eos,
        "valid_utf8": text is not None,
        "invalid_special_tokens": invalid,
        "unique_output": True,
    }


@torch.no_grad()
def _generate_with_map(
    model: FutureH0PathModel,
    source_ids: Sequence[int],
    probability_map: ProbabilityMap,
    machine: FrozenLocalPrefixFSM,
) -> dict[str, Any]:
    """Generate from BOS while holding one complete probability map fixed."""

    source, source_mask = _source_batch([source_ids])
    decoder = torch.tensor([[BOS_ID]], dtype=torch.long)
    tokens: list[int] = []
    model.eval()
    tensors = {field: torch.tensor([probability_map[field]], dtype=torch.float32) for field in FACTOR_ORDER}
    for _ in range(MAX_NEW_TOKENS):
        gates = machine.batch_gates(decoder, as_tensor=True)
        output = model(source, decoder, source_mask=source_mask, local_gates=gates)
        condition = model._decoder_condition(tensors, decoder, gates)  # type: ignore[attr-defined]
        logits = model.decoder(decoder, h0=model._decoder_h0(output["latent"]), conditioning=condition)  # type: ignore[attr-defined]
        token = int(logits[:, -1].argmax(dim=-1).item())
        tokens.append(token)
        if token == EOS_ID:
            break
        decoder = torch.tensor([[BOS_ID, *tokens]], dtype=torch.long)
    return _decode(tokens)


def _observed_trace(tokens: Sequence[int], machine: FrozenLocalPrefixFSM, factor: str) -> tuple[list[list[bool]], list[bool], bool, str | None]:
    schedule = [list(machine.gates([BOS_ID, *tokens[:position]])) for position in range(len(tokens))]
    mask = [step[FACTOR_ORDER.index(factor)] for step in schedule]
    reached = any(mask)
    return schedule, mask, not reached, None if reached else "target_gate_not_reached_in_generated_history"


def _generated_digest(generated: Mapping[str, Any]) -> str:
    return _digest({key: generated[key] for key in ("tokens", "text", "ended_eos", "valid_utf8", "invalid_special_tokens", "unique_output")})


def _map_digest(probability_map: ProbabilityMap) -> str:
    return _digest(probability_map)


def _compute_maps(model: FutureH0PathModel, rows: Sequence[Mapping[str, Any]]) -> dict[str, ProbabilityMap]:
    source_rows = [_source_ids(row) for row in rows]
    source, mask = _source_batch(source_rows)
    model.eval()
    output = model(source, source_mask=mask)
    return {str(row["id"]): _probability_map(output.factor_probs, index) for index, row in enumerate(rows)}


def _row_lookup() -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    train = {str(row["id"]): row for row in load_h0_train_rows() if row.get("metadata", {}).get("variant") == 0}
    confirmation = {str(row["id"]): row for row in load_h0_confirmation_rows() if row.get("metadata", {}).get("variant") == 0}
    return train, confirmation


def _source_swap_record(
    factor: str,
    support: str,
    baseline_row: Mapping[str, Any],
    changed_row: Mapping[str, Any],
    baseline_generated: Mapping[str, Any],
    changed_generated: Mapping[str, Any],
) -> JsonObject:
    baseline_text = baseline_generated.get("text")
    changed_text = changed_generated.get("text")
    baseline_parsed = fixture.parse_target_text(baseline_text) if isinstance(baseline_text, str) else None
    changed_parsed = fixture.parse_target_text(changed_text) if isinstance(changed_text, str) else None
    expected_base, expected_changed = _frame(baseline_row), _frame(changed_row)
    both = baseline_parsed is not None and changed_parsed is not None
    target_changed = bool(both and baseline_parsed[factor] != changed_parsed[factor])
    non_target_preserved = bool(both and all(baseline_parsed[field] == changed_parsed[field] for field in FACTOR_ORDER if field != factor))
    return {
        "record_type": "source_swap",
        "source_swap_pair_id": f"{support}/{factor}",
        "row_id": str(baseline_row["id"]),
        "support_class": support,
        "support_group": support,
        "factor": factor,
        "target_factor": factor,
        "baseline_source_row_id": str(baseline_row["id"]),
        "changed_source_row_id": str(changed_row["id"]),
        "baseline_source_identity": _source_identity(baseline_row),
        "changed_source_identity": _source_identity(changed_row),
        "left_source_identity": _source_identity(baseline_row),
        "right_source_identity": _source_identity(changed_row),
        "baseline_prefix_token_ids": [BOS_ID],
        "changed_prefix_token_ids": [BOS_ID],
        "baseline_expected_frame": expected_base,
        "changed_expected_frame": expected_changed,
        "expected_left_frame": expected_base,
        "expected_right_frame": expected_changed,
        "baseline_expected_text": _row_text(baseline_row),
        "changed_expected_text": _row_text(changed_row),
        "baseline_generated": deepcopy(dict(baseline_generated)),
        "changed_generated": deepcopy(dict(changed_generated)),
        "baseline_generated_digest": _generated_digest(baseline_generated),
        "changed_generated_digest": _generated_digest(changed_generated),
        "baseline_parsed_frame": baseline_parsed,
        "changed_parsed_frame": changed_parsed,
        "baseline_generated_token_ids": list(baseline_generated["tokens"]),
        "changed_generated_token_ids": list(changed_generated["tokens"]),
        "baseline_generated_text": baseline_text,
        "changed_generated_text": changed_text,
        "baseline_digest": _generated_digest(baseline_generated),
        "baseline_parse_failed": baseline_parsed is None,
        "changed_parse_failed": changed_parsed is None,
        "target_changed": target_changed,
        "non_target_preserved": non_target_preserved,
        "target_change_correct": bool(changed_parsed is not None and changed_parsed[factor] == expected_changed[factor]),
        "exact_target_text": bool(changed_text == _row_text(changed_row)),
        "parse_failure_reason": "baseline_or_changed_generation_unparseable" if baseline_parsed is None or changed_parsed is None else None,
        "scheduled_count": 1,
        "comparable_count": int(both),
        "target_change_correct_denominator": int(changed_parsed is not None),
    }


def _control_record(
    *,
    model: FutureH0PathModel,
    machine: FrozenLocalPrefixFSM,
    descriptor_item: Mapping[str, Any],
    support: str,
    factor: str,
    row: Mapping[str, Any],
    baseline_map: ProbabilityMap,
    baseline_generated: Mapping[str, Any],
    donor_table: Mapping[str, Mapping[str, str]],
    donor_maps: Mapping[str, ProbabilityMap],
    donor_map_digests: Mapping[str, str],
) -> JsonObject:
    width = FACTOR_SIZES[factor]
    baseline_argmax = _argmax_map(baseline_map)[factor]
    requested_class = (baseline_argmax + 1) % width if factor else baseline_argmax
    donor_row_id: str | None = None
    donor_map: ProbabilityMap | None = None
    donor_argmax: int | None = None
    donor_missing_reason: str | None = None
    records: list[JsonObject] = []
    controls = (("same_class_soft_shape", baseline_argmax), ("alternate_class_one_hot", requested_class), ("alternate_class_donor_soft", requested_class))
    for control_type, selected_class in controls:
        replacement: list[float]
        unchanged_shape = False
        missing_donor = False
        if control_type == "same_class_soft_shape":
            replacement = [0.25 / width] * width
            replacement[baseline_argmax] += 0.75
        elif control_type == "alternate_class_one_hot":
            replacement = _one_hot(width, selected_class)
        else:
            donor_row_id = str(donor_table[factor].get(str(selected_class), "")) or None
            donor_map = donor_maps.get(donor_row_id) if donor_row_id is not None else None
            donor_argmax = _argmax_map(donor_map)[factor] if donor_map is not None else None
            if donor_map is None:
                missing_donor = True
                donor_missing_reason = "prebound_donor_row_or_map_missing"
                replacement = list(baseline_map[factor])
            elif donor_argmax != selected_class:
                missing_donor = True
                donor_missing_reason = "prebound_donor_argmax_does_not_match_requested_class"
                replacement = list(baseline_map[factor])
            else:
                replacement = list(donor_map[factor])
        distance = _l1(baseline_map[factor], replacement)
        if control_type == "same_class_soft_shape" and distance <= 1e-6:
            unchanged_shape = True
        replacement_map = _replacement_map(baseline_map, factor, replacement)
        expected_schedule = deepcopy(descriptor_item["expected_target_gate_schedule"])
        expected_mask = deepcopy(descriptor_item["expected_target_gate_mask"])
        base_schedule, base_mask, base_not_reached, base_reason = _observed_trace(baseline_generated["tokens"], machine, factor)
        intervention_generated: dict[str, Any] | None = None
        intervention_parsed: Frame | None = None
        if not missing_donor and not unchanged_shape:
            intervention_generated = _generate_with_map(model, _source_ids(row), replacement_map, machine)
            text = intervention_generated.get("text")
            intervention_parsed = fixture.parse_target_text(text) if isinstance(text, str) else None
            intervention_schedule, intervention_mask, intervention_not_reached, intervention_reason = _observed_trace(intervention_generated["tokens"], machine, factor)
        else:
            intervention_schedule, intervention_mask = deepcopy(base_schedule), deepcopy(base_mask)
            intervention_not_reached = base_not_reached
            intervention_reason = "intervention_not_generated_structural_unavailability"
        raw: JsonObject = {
            "record_type": "internal_control",
            "control_type": control_type,
            "prefix_stratum": PRIMARY_PREFIX_STRATUM,
            "prefix_token_ids": [BOS_ID],
            "prefix_text": "",
            "baseline_prefix_token_ids": [BOS_ID],
            "intervention_prefix_token_ids": [BOS_ID],
            "support_class": support,
            "support_group": support,
            "probe_row_id": str(row["id"]),
            "source_row_id": str(row["id"]),
            "source_identity": _source_identity(row),
            "target_factor": factor,
            "target_frame": deepcopy(descriptor_item["target_frame"]),
            "target_text": descriptor_item["target_text"],
            "expected_target_gate_schedule": expected_schedule,
            "expected_target_gate_mask": expected_mask,
            "target_gate_reachable_after_prefix": bool(descriptor_item["target_gate_reachable_after_prefix"]),
            "baseline_observed_target_gate_schedule": base_schedule,
            "baseline_target_gate_mask": base_mask,
            "baseline_observed_gate_not_reached": base_not_reached,
            "baseline_observed_gate_not_reached_reason": base_reason,
            "intervention_observed_target_gate_schedule": intervention_schedule,
            "intervention_target_gate_mask": intervention_mask,
            "intervention_observed_gate_not_reached": intervention_not_reached,
            "intervention_observed_gate_not_reached_reason": intervention_reason,
            "target_slot_already_emitted_before_intervention": False,
            "partial_target_slot": False,
            "intervention_unavailable": False,
            "intervention_unavailable_reason": None,
            "structural_unavailable_reason": "missing_donor" if missing_donor else ("unchanged_shape" if unchanged_shape else None),
            "baseline_probability_map": deepcopy(baseline_map),
            "baseline_probability_map_digest": _map_digest(baseline_map),
            "replacement_probability_map": deepcopy(replacement_map),
            "replacement_probability_map_digest": _map_digest(replacement_map),
            "target_replacement_map": list(replacement),
            "target_replacement_probability": list(replacement),
            "unchanged_non_target_probability_maps": {field: list(baseline_map[field]) for field in FACTOR_ORDER if field != factor},
            "non_target_probability_maps": {field: list(baseline_map[field]) for field in FACTOR_ORDER if field != factor},
            "baseline_argmax": baseline_argmax,
            "baseline_factor_argmax": _argmax_map(baseline_map),
            "declared_intervention_class": selected_class,
            "actual_intervention_class": _argmax_map(replacement_map)[factor],
            "requested_class": selected_class,
            "requested_value": _vocabulary()[factor][selected_class],
            "same_class_l1_distance": distance,
            "l1_distance": distance,
            "donor_table": deepcopy(dict(donor_table)),
            "frozen_donor_table": deepcopy(dict(donor_table)),
            "donor_row_id": donor_row_id,
            "selected_donor_row_id": donor_row_id,
            "donor_factor": factor if donor_row_id is not None else None,
            "selected_donor_factor": factor if donor_row_id is not None else None,
            "donor_probability_map": deepcopy(donor_map) if donor_map is not None else None,
            "selected_donor_probability_map": deepcopy(donor_map) if donor_map is not None else None,
            "donor_probability_map_digest": donor_map_digests.get(donor_row_id) if donor_row_id is not None else None,
            "selected_donor_probability_map_digest": donor_map_digests.get(donor_row_id) if donor_row_id is not None else None,
            "donor_argmax_class": donor_argmax,
            "donor_available": not missing_donor,
            "missing_donor": missing_donor,
            "missing_donor_reason": donor_missing_reason,
            "missing_reason": donor_missing_reason,
            "unchanged_shape": unchanged_shape,
            "actual_one_hot": list(replacement) if control_type == "alternate_class_one_hot" else None,
            "availability": not (missing_donor or unchanged_shape),
            "available": not (missing_donor or unchanged_shape),
            "baseline_generated": deepcopy(dict(baseline_generated)),
            "baseline_generated_digest": _generated_digest(baseline_generated),
            "baseline_output": deepcopy(dict(baseline_generated)),
            "baseline_output_digest": _generated_digest(baseline_generated),
            "intervention_generated": deepcopy(intervention_generated) if intervention_generated is not None else None,
            "intervention_generated_digest": _generated_digest(intervention_generated) if intervention_generated is not None else None,
            "baseline_parsed_frame": fixture.parse_target_text(baseline_generated["text"]) if isinstance(baseline_generated.get("text"), str) else None,
            "intervention_parsed_frame": intervention_parsed,
        }
        vocabulary = _vocabulary()
        raw["factor_probability_vectors"] = deepcopy(baseline_map)
        raw["factor_argmax"] = _argmax_map(baseline_map)
        raw["intermediate_frame"] = {field: vocabulary[field][_argmax_map(baseline_map)[field]] for field in FACTOR_ORDER}
        raw["generated_token_ids"] = list(baseline_generated["tokens"])
        raw["generated_text"] = baseline_generated["text"]
        raw["eos"] = bool(baseline_generated["ended_eos"])
        raw["utf8_valid"] = bool(baseline_generated["valid_utf8"])
        raw["unique"] = bool(baseline_generated["unique_output"])
        raw["parsed_frame"] = raw["baseline_parsed_frame"]
        raw["parse_failure_reason"] = None if raw["parsed_frame"] is not None else "generated_text_did_not_match_fixture_grammar"
        raw["teacher_forced_byte_counts"] = None
        raw["intervention_available"] = raw["available"]
        scored = score_control_record(raw)
        scored.update({
            # One arm/seed has eight probes per control type.  The
            # three-seed scheduled denominator is a later aggregate and is
            # kept under an explicit name to avoid conflating the two.
            "scheduled_count": 8,
            "available_count": int(scored["available"]),
            "structural_unavailable_count": int(not scored["available"]),
            "intervention_unavailable_count": 0,
            "missing_count": int(missing_donor),
            "unchanged_shape_count": int(unchanged_shape),
            "scored_count": int(scored["scored"]),
            "nontrivial_eligible_count": int(scored["nontrivial_eligible"]),
            "scheduled_denominator": 8,
            "scheduled_count_three_seed": 24,
            "available_denominator": int(scored["available"]),
            "scored_denominator": int(scored["scored"]),
            "nontrivial_eligible_denominator": int(scored["nontrivial_eligible"]),
            "baseline_already_requested_denominator": int(scored["available"]),
            "baseline_parse_failed_denominator": int(scored["available"]),
            "intervention_parse_failed_denominator": int(scored["available"]),
        })
        scored["record_sha256"] = _digest(scored)
        records.append(scored)
    return {"records": records}


@torch.no_grad()
def evaluate_h0_interventions(model: FutureH0PathModel, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all eight swaps and twenty-four primary control records."""

    if not isinstance(model, FutureH0PathModel):
        raise TypeError("evaluation requires a FutureH0PathModel")
    validated = validate_h0_intervention_descriptor(descriptor)
    train_rows, confirmation_rows = _row_lookup()
    machine = FrozenLocalPrefixFSM(fixture.spec_data())
    if validated.get("probe_donor_tables"):
        donor_ids = sorted({
            row_id
            for support_tables in validated["probe_donor_tables"].values()
            for probe_table in support_tables.values()
            for factors in probe_table.values()
            for row_id in factors.values()
        })
    else:
        donor_ids = sorted({row_id for factors in validated["donor_table"].values() for row_id in factors.values()})
    donor_rows = [train_rows[row_id] for row_id in donor_ids]
    probe_ids = [validated["probe_identities"][support][factor] for support in fixture.CONFIRMATION_SUPPORTS for factor in FACTOR_ORDER]
    probe_rows = [confirmation_rows[row_id] for row_id in probe_ids]
    maps = _compute_maps(model, [*probe_rows, *donor_rows])
    probe_maps = {row_id: maps[row_id] for row_id in probe_ids}
    donor_maps = {row_id: maps[row_id] for row_id in donor_ids}
    donor_map_digests = {row_id: _map_digest(value) for row_id, value in donor_maps.items()}
    baseline_outputs = {
        row_id: _generate_with_map(model, _source_ids(confirmation_rows[row_id]), probe_maps[row_id], machine)
        for row_id in probe_ids
    }
    controls: list[JsonObject] = []
    for support in fixture.CONFIRMATION_SUPPORTS:
        for factor in FACTOR_ORDER:
            row_id = validated["probe_identities"][support][factor]
            result = _control_record(
                model=model, machine=machine, descriptor_item=validated["expected_probes"][support][factor],
                support=support, factor=factor, row=confirmation_rows[row_id],
                baseline_map=probe_maps[row_id], baseline_generated=baseline_outputs[row_id],
                donor_table=validated.get("probe_donor_tables", {}).get(support, {}).get(row_id, validated["donor_table"]),
                donor_maps=donor_maps, donor_map_digests=donor_map_digests,
            )
            for record in result["records"]:
                record["arm"] = model.arm_id
                record["seed"] = int(model.seed)
                record["row_id"] = row_id
                record["record_sha256"] = _digest(record)
                controls.append(record)
    swaps: list[JsonObject] = []
    for support in fixture.CONFIRMATION_SUPPORTS:
        for factor in FACTOR_ORDER:
            left_id, right_id = validated["source_swap_pairs"][support][factor]
            left = confirmation_rows[left_id]
            right = confirmation_rows[right_id]
            # Generate and retain the left baseline before generating the changed source.
            left_map = _compute_maps(model, [left])[left_id]
            left_generated = _generate_with_map(model, _source_ids(left), left_map, machine)
            right_map = _compute_maps(model, [right])[right_id]
            right_generated = _generate_with_map(model, _source_ids(right), right_map, machine)
            record = _source_swap_record(factor, support, left, right, left_generated, right_generated)
            record.update({
                "arm": model.arm_id,
                "seed": int(model.seed),
                "baseline_probability_map": left_map,
                "changed_probability_map": right_map,
                "baseline_probability_map_digest": _map_digest(left_map),
                "changed_probability_map_digest": _map_digest(right_map),
                "baseline_observed_target_gate_schedule": _observed_trace(left_generated["tokens"], machine, factor)[0],
                "changed_observed_target_gate_schedule": _observed_trace(right_generated["tokens"], machine, factor)[0],
            })
            record["record_sha256"] = _digest(record)
            swaps.append(record)
    return {
        "schema": RESULT_SCHEMA,
        "descriptor_sha256": validated["descriptor_sha256"],
        "arm": model.arm_id,
        "seed": int(model.seed),
        "source_swaps": swaps,
        "source_swap_records": swaps,
        "controls": controls,
        "control_records": controls,
        "probe_probability_maps": deepcopy(probe_maps),
        "probe_probability_map_digests": {row_id: _map_digest(value) for row_id, value in probe_maps.items()},
        "donor_probability_maps": deepcopy(donor_maps),
        "donor_probability_map_digests": donor_map_digests,
        "counts": {"source_swaps": len(swaps), "controls": len(controls)},
    }


run_h0_intervention_engine = evaluate_h0_interventions
evaluate_source_swaps_and_controls = evaluate_h0_interventions
evaluate_h0_confirmation_interventions = evaluate_h0_interventions
build_intervention_descriptor = build_h0_intervention_descriptor
validate_intervention_descriptor = validate_h0_intervention_descriptor


__all__ = [
    "CONTROL_TYPES", "PRIMARY_PREFIX_STRATUM", "RESULT_SCHEMA", "SCHEMA",
    "build_h0_intervention_descriptor", "build_intervention_descriptor",
    "evaluate_h0_interventions", "run_h0_intervention_engine",
    "evaluate_source_swaps_and_controls", "evaluate_h0_confirmation_interventions",
    "validate_h0_intervention_descriptor", "validate_intervention_descriptor",
]
