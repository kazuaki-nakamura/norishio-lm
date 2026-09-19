"""Validated, exclusive checkpoints for the frozen benchmark-v3 tournament."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .benchmark_v3_model import BenchmarkV3Model, build_model
from .benchmark_v3_tournament import (
    ARM_COUNTS,
    BENCHMARK_DIGEST,
    EXPECTED_CONFIG_SHA256,
    SEEDS,
)


CHECKPOINT_VERSION = "norishio.issue36.checkpoint.v1"
_PAYLOAD_KEYS = {"format", "metadata", "model_metadata", "state_schema_sha256", "state_dict"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _checked_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a nonempty mapping")
    checked: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map string names to tensors")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
        checked[name] = value
    return checked


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, exact tensor metadata, and bytes in stable key order."""
    checked = _checked_state(state)
    digest = hashlib.sha256()
    for name in sorted(checked):
        value = checked[name]
        header = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        encoded = _canonical(header)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        # Keep hashing independent of NumPy; the issue-3 CPU environment is
        # intentionally Torch-only.
        digest.update(bytes(value.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_schedule_sha256(seed: int) -> str:
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    schedule = torch.randint(384, (600, 16), generator=generator).tolist()
    return hashlib.sha256(_canonical(schedule)).hexdigest()


def _state_schema_sha256(state: Mapping[str, torch.Tensor]) -> str:
    checked = _checked_state(state)
    schema = [
        {"name": name, "dtype": str(checked[name].dtype), "shape": list(checked[name].shape)}
        for name in sorted(checked)
    ]
    return hashlib.sha256(_canonical(schema)).hexdigest()


def _fresh_model(arm: str, seed: int) -> BenchmarkV3Model:
    return build_model(arm, seed=seed)


def _architecture_metadata(model: BenchmarkV3Model) -> dict[str, Any]:
    """Return the complete deterministic architecture/initialization binding."""
    return {
        "schema": "norishio.issue36.model.v1",
        "arm": model.arm,
        "seed": model.seed,
        "source_seed": model.source_seed,
        "decoder_seed": model.decoder_seed,
        "h0_seed": model.h0_seed,
        "factor_seed": model.factor_seed,
        "projection_seed": model.projection_seed,
        "arm_metadata": dict(model.arm_metadata),
        "trainable_parameters": trainable_parameter_count(model),
    }


def _metadata(*, arm: str, seed: int, schedule_sha256: str,
              initial_state_sha256: str, final_state_sha256: str) -> dict[str, Any]:
    if arm not in ARM_COUNTS or seed not in SEEDS:
        raise ValueError("unknown tournament arm or seed")
    for name, value in {
        "schedule_sha256": schedule_sha256,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
    }.items():
        _sha256(value, name)
    return {
        "arm": arm,
        "seed": seed,
        "trainable_parameters": ARM_COUNTS[arm],
        "tournament_config_sha256": EXPECTED_CONFIG_SHA256,
        "benchmark_content_digest_sha256": BENCHMARK_DIGEST,
        "schedule_sha256": schedule_sha256,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
    }


def _publish_exclusive(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
        # Rename is atomic and fails when the destination already exists on
        # Windows.  POSIX uses a hard link for the same exclusive guarantee.
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
            os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_tournament_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    *,
    arm: str,
    seed: int,
    schedule_sha256: str,
    initial_state_sha256: str,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish one CPU state checkpoint without replacing a prior file."""
    if not isinstance(model, BenchmarkV3Model) or model.arm != arm or model.seed != seed:
        raise TypeError("model must be the matching frozen BenchmarkV3Model arm and seed")
    if trainable_parameter_count(model) != ARM_COUNTS.get(arm):
        raise ValueError("model trainable parameter count does not match frozen arm")
    model_meta = _architecture_metadata(model)
    try:
        _canonical(model_meta)
    except (TypeError, ValueError) as exc:
        raise ValueError("model metadata must be JSON-safe") from exc
    if model_metadata is not None:
        try:
            _canonical(model_metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("model metadata must be JSON-safe") from exc
        if dict(model_metadata) != model_meta:
            raise ValueError("supplied model metadata differs from frozen architecture")
    fresh = _fresh_model(arm, seed)
    expected_initial = state_sha256(fresh.state_dict())
    if initial_state_sha256 != expected_initial:
        raise ValueError("initial state hash differs from deterministic frozen initialization")
    if schedule_sha256 != expected_schedule_sha256(seed):
        raise ValueError("schedule hash differs from the frozen run-seed schedule")
    state = _checked_state({name: value.detach().cpu().clone() for name, value in model.state_dict().items()})
    final_digest = state_sha256(state)
    metadata = _metadata(
        arm=arm, seed=seed, schedule_sha256=schedule_sha256,
        initial_state_sha256=initial_state_sha256, final_state_sha256=final_digest,
    )
    payload = {
        "format": CHECKPOINT_VERSION,
        "metadata": metadata,
        "model_metadata": model_meta,
        "state_schema_sha256": _state_schema_sha256(state),
        "state_dict": state,
    }
    target = Path(path)
    buffer = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
    try:
        torch.save(payload, buffer)
        buffer.seek(0)
        serialized = buffer.read()
    finally:
        buffer.close()
    _publish_exclusive(target, serialized)
    return {**metadata, "checkpoint_file_sha256": checkpoint_file_sha256(target)}


def load_tournament_checkpoint(
    path: str | os.PathLike[str], *, arm: str, seed: int
) -> dict[str, Any]:
    """Load and authenticate a tensor-only CPU checkpoint payload."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid tournament checkpoint payload") from exc
    if not isinstance(raw, dict) or set(raw) != _PAYLOAD_KEYS:
        raise ValueError("invalid tournament checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported tournament checkpoint format")
    metadata = raw["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("invalid tournament checkpoint metadata")
    expected = _metadata(
        arm=arm, seed=seed,
        schedule_sha256=metadata.get("schedule_sha256"),
        initial_state_sha256=metadata.get("initial_state_sha256"),
        final_state_sha256=metadata.get("final_state_sha256"),
    )
    if metadata != expected:
        raise ValueError("checkpoint metadata differs from frozen bindings")
    fresh = _fresh_model(arm, seed)
    expected_model_meta = _architecture_metadata(fresh)
    if raw["model_metadata"] != expected_model_meta:
        raise ValueError("checkpoint model metadata differs from frozen architecture")
    if raw["model_metadata"].get("trainable_parameters") != ARM_COUNTS[arm]:
        raise ValueError("checkpoint model parameter count mismatch")
    fresh_state = fresh.state_dict()
    if metadata["initial_state_sha256"] != state_sha256(fresh_state):
        raise ValueError("checkpoint initial state digest mismatch")
    if metadata["schedule_sha256"] != expected_schedule_sha256(seed):
        raise ValueError("checkpoint schedule digest mismatch")
    state = raw["state_dict"]
    if not isinstance(state, Mapping) or set(state) != set(fresh_state):
        raise ValueError("checkpoint state keys differ from frozen architecture")
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint state must contain CPU tensors")
        if tensor.device.type != "cpu":
            raise ValueError("checkpoint state tensors must be on CPU")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
    if raw["state_schema_sha256"] != _state_schema_sha256(fresh_state):
        raise ValueError("checkpoint frozen state schema mismatch")
    if raw["state_schema_sha256"] != _state_schema_sha256(state):
        raise ValueError("checkpoint state schema digest mismatch")
    if state_sha256(state) != metadata["final_state_sha256"]:
        raise ValueError("checkpoint state digest mismatch")
    try:
        fresh.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state differs from frozen architecture") from exc
    fresh.eval()
    raw["model"] = fresh
    return raw


__all__ = [
    "CHECKPOINT_VERSION", "checkpoint_file_sha256", "expected_schedule_sha256",
    "load_tournament_checkpoint", "save_tournament_checkpoint", "state_sha256",
    "trainable_parameter_count",
]
