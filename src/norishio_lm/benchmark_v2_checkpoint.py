"""Validated, exclusive checkpoints for the frozen benchmark-v2 tournament."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .benchmark_v2_tournament import (
    ARM_COUNTS,
    BENCHMARK_DIGEST,
    EXPECTED_CONFIG_SHA256,
    SEEDS,
)
from .benchmark_v2_model import BenchmarkV2Model, FROZEN_DERANGEMENTS, build_model


CHECKPOINT_VERSION = "norishio.issue34.checkpoint.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, exact tensor metadata, and bytes in a stable key order."""
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a nonempty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map string names to tensors")
        value = tensor.detach().cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
        header = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        digest.update(len(_canonical(header)).to_bytes(8, "big"))
        digest.update(_canonical(header))
        # The model extra intentionally requires torch but not NumPy.
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
    schema = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("invalid model state schema")
        schema.append({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
    return hashlib.sha256(_canonical(schema)).hexdigest()


def _fresh_model(arm: str, seed: int) -> BenchmarkV2Model:
    derangements = FROZEN_DERANGEMENTS if arm == "E" else None
    return build_model(arm, seed=seed, derangement_mappings=derangements)


def _metadata(*, arm: str, seed: int, schedule_sha256: str,
              initial_state_sha256: str, final_state_sha256: str) -> dict[str, Any]:
    if arm not in ARM_COUNTS or seed not in SEEDS:
        raise ValueError("unknown tournament arm or seed")
    for name, value in {
        "schedule_sha256": schedule_sha256,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
    }.items():
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256")
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
    if not isinstance(model, BenchmarkV2Model) or model.arm != arm or model.seed != seed:
        raise TypeError("model must be the matching frozen BenchmarkV2Model arm and seed")
    if trainable_parameter_count(model) != ARM_COUNTS.get(arm):
        raise ValueError("model trainable parameter count does not match frozen arm")
    model_meta = model.checkpoint_metadata()
    if model_metadata is not None and dict(model_metadata) != model_meta:
        raise ValueError("supplied model metadata differs from frozen architecture")
    fresh = _fresh_model(arm, seed)
    expected_initial = state_sha256(fresh.state_dict())
    if initial_state_sha256 != expected_initial:
        raise ValueError("initial state hash differs from deterministic frozen initialization")
    if schedule_sha256 != expected_schedule_sha256(seed):
        raise ValueError("schedule hash differs from the frozen run-seed schedule")
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    final_digest = state_sha256(state)
    metadata = _metadata(
        arm=arm,
        seed=seed,
        schedule_sha256=schedule_sha256,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_digest,
    )
    payload = {
        "format": CHECKPOINT_VERSION,
        "metadata": metadata,
        "model_metadata": model_meta,
        "state_schema_sha256": _state_schema_sha256(state),
        "state_dict": state,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
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
    return {**metadata, "checkpoint_file_sha256": checkpoint_file_sha256(target)}


def load_tournament_checkpoint(
    path: str | os.PathLike[str], *, arm: str, seed: int
) -> dict[str, Any]:
    """Load a tensor-only payload and validate every frozen binding and state digest."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != {
        "format", "metadata", "model_metadata", "state_schema_sha256", "state_dict"
    }:
        raise ValueError("invalid tournament checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported tournament checkpoint format")
    metadata = raw["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("invalid tournament checkpoint metadata")
    expected = _metadata(
        arm=arm,
        seed=seed,
        schedule_sha256=metadata.get("schedule_sha256"),
        initial_state_sha256=metadata.get("initial_state_sha256"),
        final_state_sha256=metadata.get("final_state_sha256"),
    )
    if metadata != expected:
        raise ValueError("checkpoint metadata differs from frozen bindings")
    fresh = _fresh_model(arm, seed)
    if raw["model_metadata"] != fresh.checkpoint_metadata():
        raise ValueError("checkpoint model metadata differs from frozen architecture")
    if metadata["initial_state_sha256"] != state_sha256(fresh.state_dict()):
        raise ValueError("checkpoint initial state digest mismatch")
    if metadata["schedule_sha256"] != expected_schedule_sha256(seed):
        raise ValueError("checkpoint schedule digest mismatch")
    if raw["state_schema_sha256"] != _state_schema_sha256(fresh.state_dict()):
        raise ValueError("checkpoint frozen state schema mismatch")
    if raw["state_schema_sha256"] != _state_schema_sha256(raw["state_dict"]):
        raise ValueError("checkpoint state schema digest mismatch")
    if state_sha256(raw["state_dict"]) != metadata["final_state_sha256"]:
        raise ValueError("checkpoint state digest mismatch")
    try:
        fresh.load_state_dict(raw["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state differs from frozen architecture") from exc
    fresh.eval()
    raw["model"] = fresh
    return raw


__all__ = [
    "CHECKPOINT_VERSION",
    "checkpoint_file_sha256",
    "expected_schedule_sha256",
    "load_tournament_checkpoint",
    "save_tournament_checkpoint",
    "state_sha256",
    "trainable_parameter_count",
]
