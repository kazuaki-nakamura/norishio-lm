"""A small, inspectable CPU-only plastic memory adapter.

This module is deliberately independent of the language-model backbone.  It
stores an associative key/value matrix and exposes the state boundary clearly:
``detach_writes=True`` makes each write a detached observation, while
``False`` keeps the state connected through time for gradient experiments.
The memory is an experimental adapter, not evidence that a model has learned
semantic memory or that consolidation improves language generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import torch
from torch import nn


def _is_exact_int(value: object) -> bool:
    return type(value) is int


@dataclass(frozen=True, slots=True)
class PlasticAdapterConfig:
    """Validated dimensions and update/read controls for :class:`PlasticAdapter`."""

    key_dim: int
    value_dim: int
    detach_writes: bool = True
    learning_rate: float = 1.0
    read_scale: float = 1.0

    def __post_init__(self) -> None:
        if not _is_exact_int(self.key_dim) or self.key_dim <= 0:
            raise ValueError("key_dim must be a positive int")
        if not _is_exact_int(self.value_dim) or self.value_dim <= 0:
            raise ValueError("value_dim must be a positive int")
        if type(self.detach_writes) is not bool:
            raise TypeError("detach_writes must be a bool")
        for name, value in (("learning_rate", self.learning_rate), ("read_scale", self.read_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be greater than zero")
        if float(self.read_scale) == 0.0:
            raise ValueError("read_scale must be nonzero")

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible representation."""
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tensor_bytes(value: torch.Tensor) -> bytes:
    """Get tensor bytes without depending on NumPy."""
    contiguous = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return bytes(contiguous.view(torch.uint8).reshape(-1).tolist())


def _validate_float32_cpu_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if not bool(torch.isfinite(value.detach()).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _config_from_value(value: Any) -> PlasticAdapterConfig:
    if isinstance(value, PlasticAdapterConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("snapshot config must be a mapping")
    try:
        return PlasticAdapterConfig(
            key_dim=value["key_dim"],
            value_dim=value["value_dim"],
            detach_writes=value["detach_writes"],
            learning_rate=value["learning_rate"],
            read_scale=value["read_scale"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid snapshot config") from exc


class PlasticAdapter(nn.Module):
    """A CPU float32 key/value memory with an explicit state-gradient boundary."""

    def __init__(self, config: PlasticAdapterConfig, *, base_fingerprint: str = "unbound") -> None:
        super().__init__()
        if not isinstance(config, PlasticAdapterConfig):
            raise TypeError("config must be a PlasticAdapterConfig")
        if type(base_fingerprint) is not str or not base_fingerprint:
            raise ValueError("base_fingerprint must be a non-empty string")
        self.config = config
        self.base_fingerprint = base_fingerprint
        self.key_projection = nn.Linear(config.key_dim, config.key_dim, bias=False, device="cpu", dtype=torch.float32)
        self.value_projection = nn.Linear(config.value_dim, config.value_dim, bias=False, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.key_projection.weight.copy_(torch.eye(config.key_dim, dtype=torch.float32))
            self.value_projection.weight.copy_(torch.eye(config.value_dim, dtype=torch.float32))
        self.register_buffer("_memory", torch.zeros(config.key_dim, config.value_dim, dtype=torch.float32))
        self._read_count = 0
        self._write_count = 0

    @property
    def read_count(self) -> int:
        return self._read_count

    @property
    def write_count(self) -> int:
        return self._write_count

    def counts(self) -> dict[str, int]:
        """Return operation counts without exposing mutable internal state."""
        return {"reads": self._read_count, "writes": self._write_count}

    def restore_counts(self, *, reads: int, writes: int) -> None:
        """Restore authenticated audit counters from a checkpoint."""
        if not _is_exact_int(reads) or reads < 0 or not _is_exact_int(writes) or writes < 0:
            raise ValueError("reads and writes must be nonnegative integers")
        self._read_count = reads
        self._write_count = writes

    @property
    def memory(self) -> torch.Tensor:
        """Return a detached copy of the current memory matrix."""
        return self._memory.detach().clone()

    def _validate_parameters(self) -> None:
        for name, parameter, shape in (
            ("key_projection.weight", self.key_projection.weight, (self.config.key_dim, self.config.key_dim)),
            ("value_projection.weight", self.value_projection.weight, (self.config.value_dim, self.config.value_dim)),
        ):
            _validate_float32_cpu_tensor(parameter, name)
            if tuple(parameter.shape) != shape:
                raise ValueError(f"{name} has invalid shape {tuple(parameter.shape)}")
        _validate_float32_cpu_tensor(self._memory, "memory")
        if tuple(self._memory.shape) != (self.config.key_dim, self.config.value_dim):
            raise ValueError("memory has invalid shape")

    def _validate_features(self, value: Any, name: str, width: int) -> torch.Tensor:
        value = _validate_float32_cpu_tensor(value, name)
        if value.ndim not in (1, 2) or value.shape[-1] != width:
            raise ValueError(f"{name} must have shape ({width},) or (batch, {width})")
        return value

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Read values for one query or a batch of queries."""
        self._validate_parameters()
        query = self._validate_features(query, "query", self.config.key_dim)
        projected = query @ self.key_projection.weight.t()
        if not bool(torch.isfinite(projected.detach()).all()):
            raise ValueError("projected query must contain only finite values")
        result = (projected @ self._memory) * float(self.config.read_scale)
        if not bool(torch.isfinite(result.detach()).all()):
            raise ValueError("read result must contain only finite values")
        self._read_count += int(query.shape[0]) if query.ndim == 2 else 1
        return result

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one key/value pair or matching batches into the memory.

        With detached writes, neither inputs nor projection parameters receive
        gradients through the state update.  Through-time writes retain the
        update graph, so a later read can train both inputs and projections.
        """
        self._validate_parameters()
        key = self._validate_features(key, "key", self.config.key_dim)
        value = self._validate_features(value, "value", self.config.value_dim)
        if key.ndim != value.ndim or key.shape[:-1] != value.shape[:-1]:
            raise ValueError("key and value must have matching batch shapes")
        projected_key = key @ self.key_projection.weight.t()
        projected_value = value @ self.value_projection.weight.t()
        if not bool(torch.isfinite(projected_key.detach()).all()) or not bool(torch.isfinite(projected_value.detach()).all()):
            raise ValueError("projected write must contain only finite values")
        keys = projected_key.reshape(-1, self.config.key_dim)
        values = projected_value.reshape(-1, self.config.value_dim)
        next_memory = self._memory
        for row_key, row_value in zip(keys.unbind(0), values.unbind(0)):
            if self.config.detach_writes:
                row_key = row_key.detach()
                row_value = row_value.detach()
                next_memory = next_memory.detach()
            next_memory = next_memory + float(self.config.learning_rate) * torch.outer(row_key, row_value)
            if not bool(torch.isfinite(next_memory.detach()).all()):
                raise ValueError("memory update must contain only finite values")
        self._memory = next_memory
        self._write_count += int(keys.shape[0])

    def reset(self) -> None:
        """Clear memory and operation counts, also severing any old graph."""
        self._memory = torch.zeros_like(self._memory, device="cpu", dtype=torch.float32)
        self._read_count = 0
        self._write_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Return a detached CPU snapshot suitable for persistence/consolidation."""
        self._validate_parameters()
        return {
            "config": self.config.as_dict(),
            "base_fingerprint": self.base_fingerprint,
            "state": self._memory.detach().clone(),
            "key_projection": self.key_projection.weight.detach().clone(),
            "value_projection": self.value_projection.weight.detach().clone(),
            "counts": self.counts(),
        }

    state_snapshot = snapshot

    def digest(self) -> str:
        """Return a stable integrity digest for config, base, state, and counts."""
        snap = self.snapshot()
        metadata = {
            "config": snap["config"],
            "base_fingerprint": snap["base_fingerprint"],
            "counts": snap["counts"],
            "shape": list(snap["state"].shape),
            "dtype": str(snap["state"].dtype),
        }
        digest = hashlib.sha256()
        digest.update(_canonical_json(metadata))
        for name in ("state", "key_projection", "value_projection"):
            digest.update(name.encode("ascii"))
            digest.update(_tensor_bytes(snap[name]))
        return digest.hexdigest()

    state_digest = digest

    def load_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a compatible snapshot after validating all boundaries."""
        validated = _validate_snapshot(snapshot)
        if validated["config"] != self.config:
            raise ValueError("snapshot config mismatch")
        if validated["base_fingerprint"] != self.base_fingerprint:
            raise ValueError("snapshot base fingerprint mismatch")
        self._memory = validated["state"].detach().clone()
        with torch.no_grad():
            self.key_projection.weight.copy_(validated["key_projection"])
            self.value_projection.weight.copy_(validated["value_projection"])
        self._read_count = validated["counts"]["reads"]
        self._write_count = validated["counts"]["writes"]

    restore = load_snapshot

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "PlasticAdapter":
        validated = _validate_snapshot(snapshot)
        adapter = cls(validated["config"], base_fingerprint=validated["base_fingerprint"])
        adapter.load_snapshot(snapshot)
        return adapter

    @staticmethod
    def consolidate_snapshots(snapshots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        return consolidate_snapshots(snapshots)


def _validate_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a mapping")
    try:
        config = _config_from_value(snapshot["config"])
        base_fingerprint = snapshot["base_fingerprint"]
        state = _validate_float32_cpu_tensor(snapshot["state"], "snapshot state")
        key_projection = _validate_float32_cpu_tensor(snapshot["key_projection"], "snapshot key projection")
        value_projection = _validate_float32_cpu_tensor(snapshot["value_projection"], "snapshot value projection")
        counts = snapshot["counts"]
    except KeyError as exc:
        raise ValueError(f"snapshot missing {exc.args[0]}") from exc
    if type(base_fingerprint) is not str or not base_fingerprint:
        raise ValueError("snapshot base_fingerprint must be a non-empty string")
    if tuple(state.shape) != (config.key_dim, config.value_dim):
        raise ValueError("snapshot state has invalid shape")
    if tuple(key_projection.shape) != (config.key_dim, config.key_dim):
        raise ValueError("snapshot key projection has invalid shape")
    if tuple(value_projection.shape) != (config.value_dim, config.value_dim):
        raise ValueError("snapshot value projection has invalid shape")
    if not isinstance(counts, Mapping) or set(counts) != {"reads", "writes"}:
        raise ValueError("snapshot counts must contain reads and writes")
    checked_counts: dict[str, int] = {}
    for name in ("reads", "writes"):
        count = counts[name]
        if not _is_exact_int(count) or count < 0:
            raise ValueError(f"snapshot count {name} must be a nonnegative int")
        checked_counts[name] = count
    return {
        "config": config,
        "base_fingerprint": base_fingerprint,
        "state": state.detach().clone(),
        "key_projection": key_projection.detach().clone(),
        "value_projection": value_projection.detach().clone(),
        "counts": checked_counts,
    }


def _snapshot_sort_key(snapshot: Mapping[str, Any]) -> str:
    state = snapshot["state"]
    metadata = {
        "config": snapshot["config"].as_dict(),
        "base_fingerprint": snapshot["base_fingerprint"],
        "counts": snapshot["counts"],
        "shape": list(state.shape),
    }
    digest = hashlib.sha256(_canonical_json(metadata))
    for name in ("state", "key_projection", "value_projection"):
        digest.update(name.encode("ascii"))
        digest.update(_tensor_bytes(snapshot[name]))
    return digest.hexdigest()


def consolidate_snapshots(snapshots: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average at least two compatible snapshots deterministically.

    Consolidation is an arithmetic state average over independent snapshots;
    counts are summed for auditability.  Inputs are sorted by canonical digest
    before summation, making the result independent of input ordering.
    """
    if isinstance(snapshots, (str, bytes)):
        raise TypeError("snapshots must be an iterable of mappings")
    raw = list(snapshots)
    if len(raw) < 2:
        raise ValueError("consolidation requires at least two snapshots")
    validated = [_validate_snapshot(item) for item in raw]
    first = validated[0]
    for item in validated[1:]:
        if item["config"] != first["config"]:
            raise ValueError("snapshot config mismatch")
        if item["base_fingerprint"] != first["base_fingerprint"]:
            raise ValueError("snapshot base fingerprint mismatch")
    ordered = sorted(validated, key=_snapshot_sort_key)
    stacked = torch.stack([item["state"] for item in ordered], dim=0)
    state = stacked.mean(dim=0)
    key_projection = torch.stack([item["key_projection"] for item in ordered], dim=0).mean(dim=0)
    value_projection = torch.stack([item["value_projection"] for item in ordered], dim=0).mean(dim=0)
    if not all(bool(torch.isfinite(value).all()) for value in (state, key_projection, value_projection)):
        raise ValueError("consolidated state must contain only finite values")
    return {
        "config": first["config"].as_dict(),
        "base_fingerprint": first["base_fingerprint"],
        "state": state.to(device="cpu", dtype=torch.float32),
        "key_projection": key_projection.to(device="cpu", dtype=torch.float32),
        "value_projection": value_projection.to(device="cpu", dtype=torch.float32),
        "counts": {
            "reads": sum(item["counts"]["reads"] for item in validated),
            "writes": sum(item["counts"]["writes"] for item in validated),
        },
        "consolidated_from": len(validated),
    }


__all__ = ["PlasticAdapter", "PlasticAdapterConfig", "consolidate_snapshots"]
