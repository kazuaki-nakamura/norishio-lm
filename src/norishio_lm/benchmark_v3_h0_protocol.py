"""Frozen pre-training protocol for the Issue #44 H0 confirmation experiment.

The descriptor in ``data/benchmark_v3_h0_confirmation`` is generated from the
authored fixture, the inherited future protocol, the H0 model, and the causal
FSM.  It is then published with a literal SHA-256.  Loading or binding a run
always compares the tracked descriptor with those canonical inputs, so a
changed fixture, probe, donor, gate schedule, budget, or initialization stops
before training or scoring.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import benchmark_v3_h0_fixture as fixture
from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from .benchmark_v3_fsm import BYTE_OFFSET, BOS, FACTOR_ORDER, ByteTag, FrozenLocalPrefixFSM, GrammarCandidate, compile_target_grammar
from .benchmark_v3_h0_model import H0_ARM_IDS, PARAMETER_COUNT, build_future_h0_model
from .benchmark_v3_future_protocol import FUTURE_PROTOCOL_SHA256
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_runner import ADAM_BETAS, ADAM_EPS, BATCH_SIZE, GRADIENT_CLIP, LEARNING_RATE, TRAIN_STEPS


SCHEMA = "norishio.issue44.h0-confirmation-protocol.v1"
FREEZE_SCHEMA = "norishio.issue44.h0-confirmation-freeze.v1"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "data" / "benchmark_v3_h0_confirmation" / "experiment-descriptor-v1.json"
FIXTURE_CONTENT_SHA256 = "296f32138f9ee448ca8fe975200047f53a6679343af35cbe7423d1e3435e1945"
INHERITED_FUTURE_PROTOCOL_SHA256 = "bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c"
SEEDS = (7, 17, 29)
ARMS = ("H1L1", "H1L1_ANCHOR")
PRIMARY_PREFIX = (BOS,)
MAX_ATTEMPTS = 6
MAX_UPDATES = 3600
MAX_WALL_SECONDS = 7200


def canonical_payload(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("value is not canonical JSON-compatible") from exc


def protocol_digest(descriptor: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(descriptor)).hexdigest()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _fixture_bundle() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bundle = fixture.build()
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    if report.get("content_digest_sha256") != FIXTURE_CONTENT_SHA256:
        raise ValueError("H0 fixture content digest drift")
    return bundle, report


def _vocabulary(spec: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    values = {
        "participant": tuple(item["id"] for item in spec["participants"]),
        "time": tuple(item["id"] for item in spec["times"]),
        "event": tuple(item["id"] for item in spec["events"]),
        "operator": tuple(spec["operators"]),
    }
    if tuple(values) != FACTOR_ORDER or any(len(values[f]) not in (4, 6) for f in FACTOR_ORDER):
        raise ValueError("H0 factor vocabulary drift")
    return values


def _row_by_id(bundle: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Mapping[str, Any]]:
    rows = [row for split in ("train", "confirmation") for row in bundle[split]]
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or row_id in result:
            raise ValueError("duplicate or invalid fixture row ID")
        result[row_id] = row
    return result


def _candidate_record(candidate: GrammarCandidate, target_factor: str, fsm: FrozenLocalPrefixFSM) -> dict[str, Any]:
    tag = ByteTag.PREDICATE if target_factor in ("event", "operator") else ByteTag(target_factor)
    positions = [i for i, value in enumerate(candidate.tags) if value == tag]
    if not positions:
        raise ValueError("target factor has no candidate byte interval")
    start, end = min(positions), max(positions) + 1
    schedule: list[list[bool]] = []
    mask: list[bool] = []
    histories: list[list[int]] = []
    for consumed in range(len(candidate.byte_values)):
        history = [BOS, *[BYTE_OFFSET + value for value in candidate.byte_values[:consumed]]]
        gates = list(fsm.gates(history))
        histories.append(history)
        schedule.append(gates)
        mask.append(bool(gates[FACTOR_ORDER.index(target_factor)]))
    reachable = any(mask[pos] and pos >= 0 and pos < end for pos in range(len(mask)))
    return {
        "text": candidate.text,
        "byte_values": list(candidate.byte_values),
        "tags": [value.value for value in candidate.tags],
        "slot_start": start,
        "slot_end": end,
        "history_prefixes": histories,
        "expected_target_gate_schedule": schedule,
        "expected_target_gate_mask": mask,
        "target_gate_reachable_after_prefix": reachable,
    }


def _probe_records(bundle: Mapping[str, Sequence[Mapping[str, Any]]], probes: Mapping[str, Mapping[str, str]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _row_by_id(bundle)
    candidates = compile_target_grammar(spec)
    by_text: dict[str, GrammarCandidate] = {}
    for candidate in candidates:
        if candidate.text in by_text:
            raise ValueError("duplicate GrammarCandidate")
        by_text[candidate.text] = candidate
    fsm = FrozenLocalPrefixFSM(spec)
    result: list[dict[str, Any]] = []
    for support in fixture.CONFIRMATION_SUPPORTS:
        for factor in FACTOR_ORDER:
            row_id = probes[support][factor]
            row = rows.get(row_id)
            if row is None:
                raise ValueError("probe row is missing")
            text = row["targets"]["text"]
            candidate = by_text.get(text)
            if candidate is None:
                raise ValueError("probe target does not resolve to one GrammarCandidate")
            record = _candidate_record(candidate, factor, fsm)
            record.update({
                "support_group": support,
                "target_factor": factor,
                "row_id": row_id,
                "target_frame": deepcopy(row["targets"]["frame"]),
                "prefix_stratum": "bos_start_primary",
                "prefix_token_ids": list(PRIMARY_PREFIX),
                "target_slot_already_emitted_before_intervention": False,
                "partial_target_slot": False,
                "intervention_unavailable": False,
            })
            if not record["target_gate_reachable_after_prefix"]:
                raise ValueError("target gate is unreachable after the primary prefix")
            result.append(record)
    if len(result) != 8 or len({record["row_id"] for record in result}) != 8:
        raise ValueError("duplicate probe identity")
    return result


def _donor_table(bundle: Mapping[str, Sequence[Mapping[str, Any]]], spec: Mapping[str, Any], probes: Mapping[str, Mapping[str, str]]) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    values = _vocabulary(spec)
    confirmation_probe_ids = {row_id for support in probes.values() for row_id in support.values()}
    train_rows = [row for row in bundle["train"] if row["id"] not in confirmation_probe_ids]
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for support in fixture.CONFIRMATION_SUPPORTS:
        result[support] = {}
        for probe_id in probes[support].values():
            result[support][probe_id] = {}
            for factor in FACTOR_ORDER:
                result[support][probe_id][factor] = {}
                for requested_class in values[factor]:
                    eligible = sorted((row for row in train_rows if row["targets"]["frame"][factor] == requested_class), key=lambda row: row["id"])
                    if not eligible:
                        raise ValueError(f"missing donor for {support}/{probe_id}/{factor}/{requested_class}")
                    donor = eligible[0]
                    result[support][probe_id][factor][requested_class] = {
                        "row_id": donor["id"],
                        "factor": factor,
                        "authored_class": requested_class,
                        "source_identity": deepcopy(donor["inputs"]),
                    }
    return result


def _initial_state_digests() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        ordinary = build_future_h0_model("H1L1", seed=seed)
        anchor = build_future_h0_model("H1L1_ANCHOR", seed=seed)
        left, right = ordinary.state_dict(), anchor.state_dict()
        if tuple(left) != tuple(right) or any(not __import__("torch").equal(left[name], right[name]) for name in left):
            raise ValueError("paired initial state_dict tensors differ")
        if trainable_parameter_count(ordinary) != PARAMETER_COUNT or trainable_parameter_count(anchor) != PARAMETER_COUNT:
            raise ValueError("H0 trainable parameter count drift")
        digest = state_sha256(left)
        if digest != state_sha256(right):
            raise ValueError("paired initial state digest differs")
        result[str(seed)] = {"H1L1": digest, "H1L1_ANCHOR": digest, "bitwise_equal": True, "trainable_parameters": PARAMETER_COUNT}
    return result


RAW_REQUIRED_FIELDS = [
    "arm", "seed", "row_id", "support_group", "source_identity", "target_frame", "target_text",
    "factor_probability_vectors", "factor_argmax", "intermediate_frame", "generated_token_ids", "generated_text",
    "eos", "utf8_valid", "unique", "parsed_frame", "parse_failure_reason", "teacher_forced_byte_counts",
]
RAW_CONTROL_FIELDS = [
    "control_type", "prefix_stratum", "prefix_token_ids", "prefix_text", "target_factor",
    "expected_target_gate_schedule", "expected_target_gate_mask", "target_gate_reachable_after_prefix",
    "baseline_observed_target_gate_schedule", "intervention_observed_target_gate_schedule",
    "baseline_target_gate_mask", "intervention_target_gate_mask", "baseline_observed_gate_not_reached",
    "intervention_observed_gate_not_reached", "target_slot_already_emitted_before_intervention", "partial_target_slot",
    "intervention_unavailable", "structural_unavailable_reason", "baseline_prefix_token_ids", "intervention_prefix_token_ids",
    "baseline_output", "baseline_probability_map", "baseline_output_digest", "target_replacement_map", "non_target_probability_maps",
    "same_class_l1_distance", "baseline_argmax", "declared_intervention_class", "actual_intervention_class", "requested_class",
    "actual_one_hot", "donor_table", "selected_donor_row_id", "selected_donor_factor", "selected_donor_probability_map",
    "available", "missing_reason", "baseline_parse_failed", "intervention_parse_failed", "baseline_already_requested",
    "target_changed", "requested_value_success", "nontrivial_requested_success", "non_target_preserved", "nontrivial_joint_success",
    "scheduled_count", "available_count", "structural_unavailable_count", "intervention_unavailable_count", "missing_count",
    "unchanged_shape_count", "scored_count", "nontrivial_eligible_count",
]


def _canonical_descriptor() -> dict[str, Any]:
    bundle, report = _fixture_bundle()
    spec = fixture.spec_data()
    probes = fixture.probe_identities(bundle)
    swaps = fixture.source_swap_pairs(bundle)
    probe_records = _probe_records(bundle, probes, spec)
    donors = _donor_table(bundle, spec, probes)
    schedules = {str(seed): schedule_sha256(batch_schedule(seed=seed, train_rows=384, steps=600, batch_size=16)) for seed in SEEDS}
    return {
        "schema": SCHEMA,
        "scope": {"applies_to": "future_benchmark_v3_h0_confirmation_only", "historical_artifacts": "excluded_immutable", "final_split": None, "no_final_split": True},
        "inherited_future_protocol": {"schema": "norishio.issue39.future-protocol-freeze.v1", "descriptor_sha256": INHERITED_FUTURE_PROTOCOL_SHA256, "future_protocol_sha256": INHERITED_FUTURE_PROTOCOL_SHA256},
        "fixture": {"schema": fixture.SCHEMA, "content_digest_sha256": report["content_digest_sha256"], "counts": report["counts"], "source": "data/benchmark_v3_h0_confirmation/expected-manifest.json"},
        "arms": [{"arm": arm, "internal_arm": "H1L1", "h0_mode": "source_latent" if arm == "H1L1" else "anchor_bos_sep"} for arm in ARMS],
        "seeds": list(SEEDS),
        "model": {"trainable_parameter_count": PARAMETER_COUNT, "initialization_family": "torch.manual_seed(seed); build_future_h0_model", "initial_state_by_seed": _initial_state_digests(), "paired_state_rule": "bitwise_equal_state_dict_and_equal_sha256_for_each_seed"},
        "training": {"optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPS, "weight_decay": 0.0}, "schedule": {"train_rows": 384, "updates_per_run": TRAIN_STEPS, "batch_size": BATCH_SIZE, "replacement": True, "seeded_cpu_torch_schedule_sha256": schedules}, "gradient_clip": GRADIENT_CLIP, "torch_num_threads": 1, "deterministic_algorithms": True, "run_order": ["H1L1 seed7", "H1L1_ANCHOR seed7", "H1L1 seed17", "H1L1_ANCHOR seed17", "H1L1 seed29", "H1L1_ANCHOR seed29"], "evaluation_order": ["ordinary_confirmation", "baseline_probability_maps_and_outputs", "donor_probability_maps", "internal_controls_in_fixture_support_factor_control_order", "source_swaps_in_fixture_support_factor_order"], "budget": {"max_attempts": MAX_ATTEMPTS, "max_optimizer_updates": MAX_UPDATES, "max_wall_seconds": MAX_WALL_SECONDS, "gpu": False, "network": False, "paid_compute": False}},
        "selection": {"primary": "all.generation_frame_exact.accuracy", "aggregation": "unweighted_three_seed_mean", "direction": "maximize", "no_final_split": True, "support_denominators": {"all": 96, "each_support_group": 48}, "proposed_thresholds": {"preserved_max_drop": 0.05, "improved_min_gain": 0.10, "strong_head_min": 0.80, "weak_scheduled_joint_max": 0.05, "positive_scheduled_joint_min": 0.25, "same_class_material_effect_min": 0.10, "constant_source_close_max_gap": 0.05, "unseen_pair_near_zero_max": 0.02, "seen_vs_unseen_material_gap_min": 0.10}},
        "probes": {"source_swap_pairs": swaps, "internal_probe_identities": probes, "counts": {"source_swap_pairs": 8, "internal_probes": 8, "pairs_per_seed": 8}, "primary_prefix": {"prefix_stratum": "bos_start_primary", "prefix_token_ids": list(PRIMARY_PREFIX), "byte_offset": BYTE_OFFSET}},
        "fsm": {"candidate_count": len(compile_target_grammar(spec)), "candidate_uniqueness": "exact target text maps to one GrammarCandidate", "byte_offset": BYTE_OFFSET, "primary_probe_records": probe_records, "schedule_rule": "history_p=[BOS_ID]+[BYTE_OFFSET+b for b in candidate.byte_values[:p]], p=0..len(byte_values)-1", "reachability_rule": "any target mask true at or after consumed prefix and before target slot end"},
        "donors": {"selection_rule": "lexicographically first eligible train row ID for every authored class; exclude all confirmation probe IDs; freeze before training", "table": donors},
        "controls": {"types": ["same_class_soft_shape", "alternate_class_one_hot", "alternate_class_donor_soft"], "same_class_soft_shape": "0.75*one_hot(baseline_argmax)+0.25*uniform(width)", "alternate_class_one_hot": "one_hot((baseline_argmax+1)%width)", "alternate_class_donor_soft": "frozen donor row for q=(baseline_argmax+1)%width, only if donor recorded soft argmax=q", "structural_reason_priority": ["missing_donor", "unchanged_shape", "invalid_or_ambiguous_common_prefix", "target_slot_already_emitted_before_intervention", "target_gate_unreachable_after_prefix"], "scheduled_denominator": 24, "scheduled_by_factor": 6, "scheduled_by_support_group": 12, "availability_formula": "available_count=scheduled_count-structural_unavailable_count", "structural_formula": "structural_unavailable_count=missing_count+unchanged_shape_count+intervention_unavailable_count"},
        "raw_artifact_schema": {"common_required_fields": RAW_REQUIRED_FIELDS, "internal_control_required_fields": RAW_CONTROL_FIELDS, "source_swap_required_fields": ["source_swap_pair_id", "left_source_identity", "right_source_identity", "expected_left_frame", "expected_right_frame", "baseline_generated_token_ids", "baseline_generated_text", "baseline_parsed_frame", "baseline_digest", "changed_generated_token_ids", "changed_generated_text", "changed_parsed_frame", "target_changed", "non_target_preserved", "exact_target_text", "parse_failure_reason"], "aggregate_required_fields": ["scheduled_count", "available_count", "structural_unavailable_count", "intervention_unavailable_count", "missing_count", "unchanged_shape_count", "scored_count", "baseline_already_requested_count", "baseline_parse_failed_count", "intervention_parse_failed_count", "nontrivial_eligible_count", "target_changed_count", "requested_value_success_count", "nontrivial_requested_success_count", "non_target_preserved_count", "nontrivial_joint_success_count", "denominator"]},
        "decision_branches": [{"id": "h_bypass_supported", "condition": "H1L1_ANCHOR preserves head and ordinary generation frame accuracy within 0.05 and improves alternate-one-hot scheduled-24 nontrivial_joint_success by at least 0.10"}, {"id": "h_bypass_weakened", "condition": "both arms have strong heads and positive scheduled joint follow-through and ordinary frame means differ by less than 0.05"}, {"id": "decoder_follow_through_or_continuous_limit", "condition": "both arms have strong heads but weak scheduled joint follow-through"}, {"id": "decoder_path_inconclusive", "condition": "a head-frame rate is below 0.80"}, {"id": "same_class_shape_sensitivity", "condition": "same-class soft-shape changes parsed target values or lowers non-target preservation by at least 0.10; withhold canonical-class interpretation"}, {"id": "constant_source_close", "condition": "H1L1 constant-source ablation is within 0.05 absolute of the H1L1 three-seed ordinary frame mean; withhold source-dependent interpretation and do not add a trained arm"}, {"id": "fixture_specific_compositional_failure", "condition": "unseen_pair frame exact mean is at most 0.02 and seen_pair/unseen_triple exceeds it by at least 0.10"}, {"id": "frame_recovery_without_surface_recovery", "condition": "frame exact improves without exact target text recovery; report frame recovery without surface-variant reconstruction claim"}, {"id": "one_hot_only_interpretation", "condition": "alternate one-hot has nontrivial joint success while donor-soft does not; continuous soft-distribution following remains unconfirmed"}, {"id": "donor_soft_only_interpretation", "condition": "donor-soft has nontrivial joint success while alternate one-hot does not; shape sensitivity is shown without canonical-class control evidence"}, {"id": "withhold", "condition": "any preflight drift, budget breach, primary gate unreachability, raw-field omission, or protocol mismatch"}],
    }


def build_descriptor() -> dict[str, Any]:
    return deepcopy(_canonical_descriptor())


# This literal is deliberately updated only alongside the tracked JSON freeze.
PROTOCOL_SHA256 = "19c11e5fa7b5ae90dd2bc3b6ddca20366e6d2b9798b98e3ccf1c7c8bcdf10533"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Issue #44 protocol descriptor is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "descriptor_sha256", "descriptor"}:
        raise ValueError("Issue #44 protocol wrapper fields changed")
    if payload.get("schema") != FREEZE_SCHEMA or payload.get("descriptor_sha256") != PROTOCOL_SHA256:
        raise ValueError("Issue #44 protocol literal digest changed")
    descriptor = payload.get("descriptor")
    if not isinstance(descriptor, Mapping) or protocol_digest(descriptor) != PROTOCOL_SHA256:
        raise ValueError("Issue #44 protocol descriptor digest mismatch")
    return deepcopy(dict(descriptor))


def validate_protocol(descriptor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = load_protocol() if descriptor is None else descriptor
    if not isinstance(value, Mapping):
        raise ValueError("Issue #44 descriptor must be a mapping")
    expected = build_descriptor()
    if dict(value) != expected:
        raise ValueError("Issue #44 descriptor differs from canonical deterministic inputs")
    if protocol_digest(value) != _sha(PROTOCOL_SHA256, "Issue #44 protocol digest"):
        raise ValueError("Issue #44 descriptor digest mismatch")
    if value.get("inherited_future_protocol", {}).get("descriptor_sha256") != INHERITED_FUTURE_PROTOCOL_SHA256:
        raise ValueError("inherited future protocol digest mismatch")
    return deepcopy(dict(value))


def bind_run_metadata(run_id: str, descriptor: Mapping[str, Any] | None = None) -> dict[str, str]:
    validate_protocol(descriptor)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a nonempty string")
    return {"schema": "norishio.issue44.h0-run-metadata.v1", "run_id": run_id, "experiment_descriptor_sha256": PROTOCOL_SHA256, "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256}


def validate_run_metadata(metadata: Mapping[str, Any], descriptor: Mapping[str, Any] | None = None) -> dict[str, str]:
    validate_protocol(descriptor)
    if not isinstance(metadata, Mapping) or set(metadata) != {"schema", "run_id", "experiment_descriptor_sha256", "fixture_content_digest_sha256"}:
        raise ValueError("run metadata fields changed")
    if metadata.get("schema") != "norishio.issue44.h0-run-metadata.v1" or metadata.get("experiment_descriptor_sha256") != PROTOCOL_SHA256 or metadata.get("fixture_content_digest_sha256") != FIXTURE_CONTENT_SHA256:
        raise ValueError("run metadata protocol binding mismatch")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"].strip():
        raise ValueError("run metadata requires nonempty run_id")
    return dict(metadata)


def _validate_raw_fields(raw_artifact: Mapping[str, Any]) -> None:
    required = set(RAW_REQUIRED_FIELDS)
    if not required <= set(raw_artifact):
        missing = sorted(required - set(raw_artifact))
        raise ValueError(f"missing raw fields: {missing}")
    if "control_type" in raw_artifact:
        missing = sorted(set(RAW_CONTROL_FIELDS) - set(raw_artifact))
        if missing:
            raise ValueError(f"missing raw control fields: {missing}")
    if "source_swap_pair_id" in raw_artifact:
        # The source-swap field list is descriptor-authenticated and kept
        # separate from the per-row control fields.
        source_fields = set(build_descriptor()["raw_artifact_schema"]["source_swap_required_fields"])
        missing = sorted(source_fields - set(raw_artifact))
        if missing:
            raise ValueError(f"missing raw source-swap fields: {missing}")


def preflight_protocol(descriptor: Mapping[str, Any] | None = None, *, fixture_bundle: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, raw_artifact: Mapping[str, Any] | None = None, arm: str | None = None, seed: int | None = None) -> dict[str, Any]:
    value = validate_protocol(descriptor)
    bundle = fixture.build() if fixture_bundle is None else fixture_bundle
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    if report.get("content_digest_sha256") != FIXTURE_CONTENT_SHA256:
        raise ValueError("fixture content digest drift")
    if arm is not None and arm not in ARMS:
        raise ValueError("unknown H0 arm")
    if seed is not None and seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    seeds = SEEDS if seed is None else (seed,)
    for current_seed in seeds:
        ordinary = build_future_h0_model("H1L1", seed=current_seed)
        anchor = build_future_h0_model("H1L1_ANCHOR", seed=current_seed)
        if tuple(ordinary.state_dict()) != tuple(anchor.state_dict()) or any(not __import__("torch").equal(ordinary.state_dict()[name], anchor.state_dict()[name]) for name in ordinary.state_dict()):
            raise ValueError("initial state mismatch")
        digest = state_sha256(ordinary.state_dict())
        if digest != value["model"]["initial_state_by_seed"][str(current_seed)]["H1L1"]:
            raise ValueError("initial state digest mismatch")
    if raw_artifact is not None:
        _validate_raw_fields(raw_artifact)
    return {"descriptor_sha256": PROTOCOL_SHA256, "fixture_content_digest_sha256": FIXTURE_CONTENT_SHA256, "preflight": True, "arms": list(ARMS if arm is None else (arm,)), "seeds": list(seeds)}


# Explicit aliases make the future-only API discoverable beside Issue #39.
load_h0_protocol = load_protocol
validate_h0_protocol = validate_protocol
bind_h0_run_metadata = bind_run_metadata
validate_h0_run_metadata = validate_run_metadata
preflight_h0_protocol = preflight_protocol
load_h0_experiment_descriptor = load_protocol
validate_h0_experiment_descriptor = validate_protocol
bind_h0_run = bind_run_metadata
preflight_h0_experiment = preflight_protocol
EXPERIMENT_DESCRIPTOR_PATH = PROTOCOL_PATH
EXPERIMENT_DESCRIPTOR_SHA256 = PROTOCOL_SHA256
DESCRIPTOR_SHA256 = PROTOCOL_SHA256
H0_PROTOCOL_SHA256 = PROTOCOL_SHA256


__all__ = [
    "ARMS", "DESCRIPTOR_SHA256", "EXPERIMENT_DESCRIPTOR_PATH", "EXPERIMENT_DESCRIPTOR_SHA256", "FIXTURE_CONTENT_SHA256", "H0_PROTOCOL_SHA256", "INHERITED_FUTURE_PROTOCOL_SHA256", "MAX_ATTEMPTS", "MAX_UPDATES", "MAX_WALL_SECONDS", "PRIMARY_PREFIX", "PROTOCOL_PATH", "PROTOCOL_SHA256", "SCHEMA", "SEEDS", "bind_h0_run", "bind_h0_run_metadata", "bind_run_metadata", "build_descriptor", "canonical_payload", "load_h0_experiment_descriptor", "load_h0_protocol", "load_protocol", "preflight_h0_experiment", "preflight_h0_protocol", "preflight_protocol", "protocol_digest", "validate_h0_experiment_descriptor", "validate_h0_protocol", "validate_h0_run_metadata", "validate_protocol", "validate_run_metadata",
]
