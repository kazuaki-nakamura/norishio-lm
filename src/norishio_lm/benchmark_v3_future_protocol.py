"""Future-only benchmark-v3 protocol binding.

Issue #39 found a selection-path mismatch after the Issue #36 artifacts were
already consumed. This module defines a separately versioned future protocol;
it does not import or mutate the historical runner, checkpoint, terminal,
marker, result, or attestation paths.

Every protected callback receives authenticated run metadata. Validation
happens before training, evaluator invocation, final marker publication, and
final-row access. Future artifacts returned by the callbacks must carry the
same published protocol digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .benchmark_v3_contract import INTERVENTION_RULE, SELECTION_CONTRACT


T = TypeVar("T")
SCHEMA = "norishio.issue39.future-benchmark-v3-protocol.v1"
RUN_METADATA_SCHEMA = "norishio.issue39.future-benchmark-v3-run-metadata.v1"
DIGEST_FIELD = "future_benchmark_v3_protocol_sha256"
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "data" / "benchmark_v3_factor_path" / "future-protocol-v1.json"
USAGE_PHASES = (
    "future_training",
    "future_evaluator",
    "future_final_marker_publication",
    "future_final_row_access",
)

_EXPECTED_DESCRIPTOR: dict[str, Any] = {
    "schema": SCHEMA,
    "scope": {
        "applies_to": "future_benchmark_v3_runs_only",
        "historical_issue36_artifacts": "excluded_immutable",
    },
    "selection": copy.deepcopy(SELECTION_CONTRACT),
    "intervention": {
        "rule": INTERVENTION_RULE,
        "class_selection": "(baseline_argmax + 1) % width",
        "requires_different_class": True,
    },
    "execution_binding": {
        "required_before": list(USAGE_PHASES),
        "run_metadata_schema": RUN_METADATA_SCHEMA,
        "digest_field": DIGEST_FIELD,
        "required_future_artifacts": [
            "trainer_result",
            "development_result",
            "checkpoint",
            "terminal_record",
            "final_marker",
            "final_result",
            "final_attestation",
        ],
    },
}

# Public inspection copy. Validation always compares against the private
# descriptor and the literal published digest below.
FUTURE_PROTOCOL_DESCRIPTOR: dict[str, Any] = copy.deepcopy(_EXPECTED_DESCRIPTOR)


def canonical_payload(value: Any) -> bytes:
    """Serialize strict JSON deterministically as UTF-8 bytes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON-compatible") from exc


def protocol_digest(descriptor: Mapping[str, Any] | None = None) -> str:
    """Return the canonical descriptor SHA-256."""

    value = FUTURE_PROTOCOL_DESCRIPTOR if descriptor is None else descriptor
    return hashlib.sha256(canonical_payload(value)).hexdigest()


# Published freeze value. Do not derive it at import time: changing the
# descriptor without deliberately revising this value must fail validation.
FUTURE_PROTOCOL_SHA256 = "bacf9d952ca159048740a06a941f9bdf9e3b6d531b719da666f034aa8a2bae6c"


def _lower_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def load_future_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Load the tracked descriptor wrapper without invoking any run path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("future protocol file is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema", "descriptor_sha256", "descriptor",
    }:
        raise ValueError("future protocol file fields changed")
    if payload.get("schema") != "norishio.issue39.future-protocol-freeze.v1":
        raise ValueError("future protocol file schema changed")
    if payload.get("descriptor_sha256") != FUTURE_PROTOCOL_SHA256:
        raise ValueError("future protocol file digest changed")
    descriptor = payload.get("descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("future protocol file descriptor is missing")
    if protocol_digest(descriptor) != FUTURE_PROTOCOL_SHA256:
        raise ValueError("future protocol file descriptor digest mismatch")
    return copy.deepcopy(dict(descriptor))


def validate_future_protocol(
    descriptor: Mapping[str, Any] | None = None,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    """Validate the exact descriptor and its published digest."""

    value = load_future_protocol() if descriptor is None else descriptor
    if not isinstance(value, Mapping):
        raise ValueError("future protocol descriptor must be a mapping")
    if dict(value) != _EXPECTED_DESCRIPTOR:
        raise ValueError("future protocol descriptor differs from the frozen payload")
    if protocol_digest(value) != _lower_sha256(
        FUTURE_PROTOCOL_SHA256, "future protocol digest"
    ):
        raise ValueError("future protocol descriptor digest mismatch")
    if phase is not None and phase not in USAGE_PHASES:
        raise ValueError(f"unsupported future protocol usage phase: {phase!r}")
    return copy.deepcopy(dict(value))


def bind_future_run_metadata(
    run_id: str,
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Create run metadata only after validating the frozen descriptor."""

    validate_future_protocol(descriptor)
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("future run_id must be a nonempty string")
    return {
        "schema": RUN_METADATA_SCHEMA,
        "run_id": run_id,
        DIGEST_FIELD: FUTURE_PROTOCOL_SHA256,
    }


def validate_future_run_metadata(
    metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
    *,
    phase: str | None = None,
) -> dict[str, str]:
    """Validate a future run binding before a protected phase."""

    validate_future_protocol(descriptor, phase=phase)
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "schema", "run_id", DIGEST_FIELD,
    }:
        raise ValueError("future run metadata fields changed")
    if metadata.get("schema") != RUN_METADATA_SCHEMA:
        raise ValueError("future run metadata schema changed")
    if not isinstance(metadata.get("run_id"), str) or not metadata["run_id"].strip():
        raise ValueError("future run metadata requires a nonempty run_id")
    if metadata.get(DIGEST_FIELD) != FUTURE_PROTOCOL_SHA256:
        raise ValueError("future run metadata protocol digest mismatch")
    return dict(metadata)


def validate_future_artifact_binding(
    artifact: Mapping[str, Any], *, label: str,
) -> None:
    """Require a future artifact to retain the pre-run protocol digest."""

    if not isinstance(artifact, Mapping):
        raise ValueError(f"{label} must be a mapping with protocol binding")
    if artifact.get(DIGEST_FIELD) != FUTURE_PROTOCOL_SHA256:
        raise ValueError(f"{label} protocol digest mismatch")


def run_validated_future_phase(
    phase: str,
    callback: Callable[[Mapping[str, str]], T],
    *,
    run_metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
) -> T:
    """Validate descriptor and run binding, then call one phase callback."""

    metadata = validate_future_run_metadata(
        run_metadata, descriptor, phase=phase,
    )
    if not callable(callback):
        raise TypeError("future phase callback must be callable")
    return callback(metadata)


def run_future_development(
    trainer: Callable[[Mapping[str, str]], Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]],
    *,
    run_metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Future development order: validate, train, revalidate, evaluate."""

    training = run_validated_future_phase(
        "future_training", trainer,
        run_metadata=run_metadata, descriptor=descriptor,
    )
    validate_future_artifact_binding(training, label="trainer result")
    metadata = validate_future_run_metadata(
        run_metadata, descriptor, phase="future_evaluator",
    )
    if not callable(evaluator):
        raise TypeError("future evaluator must be callable")
    evaluation = evaluator(training, metadata)
    validate_future_artifact_binding(evaluation, label="development result")
    return {"run_metadata": metadata, "training": training, "evaluation": evaluation}


def run_future_final(
    marker_publisher: Callable[[Mapping[str, str]], Mapping[str, Any]],
    final_rows_accessor: Callable[[Mapping[str, str]], Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any], Mapping[str, str]], Mapping[str, Any]],
    *,
    run_metadata: Mapping[str, Any],
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Future final order with validation before marker and row access."""

    marker = run_validated_future_phase(
        "future_final_marker_publication", marker_publisher,
        run_metadata=run_metadata, descriptor=descriptor,
    )
    validate_future_artifact_binding(marker, label="final marker")
    rows = run_validated_future_phase(
        "future_final_row_access", final_rows_accessor,
        run_metadata=run_metadata, descriptor=descriptor,
    )
    validate_future_artifact_binding(rows, label="final row bundle")
    metadata = validate_future_run_metadata(
        run_metadata, descriptor, phase="future_evaluator",
    )
    if not callable(evaluator):
        raise TypeError("future evaluator must be callable")
    result = evaluator(rows, metadata)
    validate_future_artifact_binding(result, label="final result")
    return {"run_metadata": metadata, "marker": marker, "rows": rows, "result": result}


__all__ = [
    "DIGEST_FIELD",
    "FUTURE_PROTOCOL_DESCRIPTOR",
    "FUTURE_PROTOCOL_SHA256",
    "RUN_METADATA_SCHEMA",
    "PROTOCOL_PATH",
    "SCHEMA",
    "USAGE_PHASES",
    "bind_future_run_metadata",
    "canonical_payload",
    "load_future_protocol",
    "protocol_digest",
    "run_future_development",
    "run_future_final",
    "run_validated_future_phase",
    "validate_future_artifact_binding",
    "validate_future_protocol",
    "validate_future_run_metadata",
]
