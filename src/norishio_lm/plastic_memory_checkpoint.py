"""Safe, exclusive checkpoints for plastic-memory adapters.

Only a JSON-shaped record and CPU ``float32`` tensors are serialized.  The
adapter object itself is never put in the checkpoint, which keeps loading
safe with ``weights_only=True`` and makes the format usable by a later
``PlasticAdapter`` implementation as well as a plain ``torch.nn.Module``.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch


CHECKPOINT_VERSION = "norishio.plastic-memory.checkpoint.v1"
_PAYLOAD_KEYS = {
    "format",
    "adapter_id",
    "config",
    "base_fingerprint",
    "source_episode_ids",
    "created_at",
    "update_count",
    "tags",
    "importance",
    "operation_counts",
    "metadata",
    "state_schema_sha256",
    "state_content_sha256",
    "metadata_sha256",
    "state_dict",
}


def _canonical(value: Any) -> bytes:
    """Encode a strictly JSON-safe value deterministically."""
    def validate(item: Any) -> None:
        if item is None or type(item) is bool or type(item) is str or type(item) is int:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("checkpoint metadata must be JSON-safe")
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("checkpoint metadata must be JSON-safe")
            for key, child in item.items():
                validate(key)
                validate(child)
            return
        raise ValueError("checkpoint metadata must be JSON-safe")

    try:
        validate(value)
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON-safe") from exc
    return encoded.encode("utf-8")


def _json_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    # json.dumps catches non-finite floats and unsupported values.  Keep the
    # original mapping so callers get the same primitive metadata back.
    _canonical(dict(value))
    return dict(value)


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _episode_ids(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("source_episode_ids must be a sequence of strings")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("source_episode_ids must contain non-empty strings")
    return result


def _tags(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("tags must be a sequence of strings")
    result = list(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError("tags must contain non-empty strings")
    return result


def _importance(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("importance must be a finite number or None")
    if not math.isfinite(float(value)):
        raise ValueError("importance must be a finite number or None")
    return value


def _operation_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != {"reads", "writes"}:
        raise ValueError("operation_counts must contain reads and writes")
    result: dict[str, int] = {}
    for name in ("reads", "writes"):
        count = value[name]
        if type(count) is not int or count < 0:
            raise ValueError(f"operation_counts {name} must be a non-negative integer")
        result[name] = count
    return result


def _config_from_adapter(adapter: Any) -> dict[str, Any]:
    config = getattr(adapter, "config", None)
    if config is None:
        raise ValueError("config must be supplied when adapter has no config")
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    elif hasattr(config, "to_dict") and callable(config.to_dict):
        config = config.to_dict()
    elif hasattr(config, "as_dict") and callable(config.as_dict):
        config = config.as_dict()
    return _json_object(config, name="config")


def _resolve_config(adapter: Any, config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return _config_from_adapter(adapter)
    return _json_object(config, name="config")


def _resolve_metadata(adapter: Any, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is not None:
        return _json_object(metadata, name="metadata")
    candidate = getattr(adapter, "metadata", {})
    if candidate is None:
        candidate = {}
    return _json_object(candidate, name="metadata")


def _checked_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state_dict must be a non-empty mapping")
    checked: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("state_dict must map string names to tensors")
        if tensor.dtype != torch.float32:
            raise ValueError(f"checkpoint tensor {name!r} must have dtype float32")
        value = tensor.detach().to(device="cpu").contiguous().clone()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite checkpoint tensor: {name}")
        checked[name] = value
    return checked


def state_schema_sha256(state: Mapping[str, torch.Tensor]) -> str:
    checked = _checked_state(state)
    schema = [
        {"name": name, "dtype": str(checked[name].dtype), "shape": list(checked[name].shape)}
        for name in sorted(checked)
    ]
    return hashlib.sha256(_canonical(schema)).hexdigest()


def state_content_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and bytes without requiring NumPy."""
    checked = _checked_state(state)
    digest = hashlib.sha256()
    for name in sorted(checked):
        value = checked[name]
        header = _canonical({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(bytes(value.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def _metadata_digest(payload: Mapping[str, Any]) -> str:
    fields = {
        key: payload[key]
        for key in (
            "format", "adapter_id", "config", "base_fingerprint",
            "source_episode_ids", "created_at", "update_count", "tags",
            "importance", "operation_counts", "metadata", "state_schema_sha256", "state_content_sha256",
        )
    }
    return hashlib.sha256(_canonical(fields)).hexdigest()


def _publish_exclusive(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # os.rename is exclusive on Windows.  On POSIX, link() is the
        # non-replacing publication primitive; both close the existence race.
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


def save_plastic_memory_checkpoint(
    path: str | os.PathLike[str],
    adapter: torch.nn.Module,
    *,
    created_at: str,
    base_fingerprint: str | None = None,
    adapter_id: str | None = None,
    source_episode_ids: Sequence[str] = (),
    update_count: int = 0,
    tags: Sequence[str] = (),
    importance: int | float | None = None,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one exclusive, tensor-only adapter checkpoint."""
    if not isinstance(adapter, torch.nn.Module):
        raise TypeError("adapter must be a torch.nn.Module")
    resolved_config = _resolve_config(adapter, config)
    if base_fingerprint is None:
        base_fingerprint = getattr(adapter, "base_fingerprint", None)
    if adapter_id is None:
        adapter_id = getattr(adapter, "adapter_id", None)
    base_fingerprint = _nonempty_string(base_fingerprint, name="base_fingerprint")
    adapter_id = _nonempty_string(adapter_id, name="adapter_id")
    created_at = _nonempty_string(created_at, name="created_at")
    source_episode_ids = _episode_ids(source_episode_ids)
    tags = _tags(tags)
    if type(update_count) is not int or update_count < 0:
        raise ValueError("update_count must be a non-negative integer")
    importance = _importance(importance)
    resolved_metadata = _resolve_metadata(adapter, metadata)
    counts_method = getattr(adapter, "counts", None)
    operation_counts = _operation_counts(counts_method()) if callable(counts_method) else {"reads": 0, "writes": update_count}
    if callable(counts_method) and operation_counts["writes"] != update_count:
        raise ValueError("update_count must match adapter write count")
    state = _checked_state(adapter.state_dict())
    payload: dict[str, Any] = {
        "format": CHECKPOINT_VERSION,
        "adapter_id": adapter_id,
        "config": resolved_config,
        "base_fingerprint": base_fingerprint,
        "source_episode_ids": source_episode_ids,
        "created_at": created_at,
        "update_count": update_count,
        "tags": tags,
        "importance": importance,
        "operation_counts": operation_counts,
        "metadata": resolved_metadata,
        "state_schema_sha256": state_schema_sha256(state),
        "state_content_sha256": state_content_sha256(state),
        "state_dict": state,
    }
    payload["metadata_sha256"] = _metadata_digest(payload)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _publish_exclusive(Path(path), buffer.getvalue())
    return dict(payload)


def _expect_equal(actual: Any, expected: Any, *, name: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"checkpoint {name} mismatch")


def load_plastic_memory_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: torch.nn.Module | None = None,
    expected_base_fingerprint: str | None = None,
    expected_adapter_id: str | None = None,
    expected_source_episode_ids: Sequence[str] | None = None,
    expected_created_at: str | None = None,
    expected_update_count: int | None = None,
    expected_tags: Sequence[str] | None = None,
    expected_importance: int | float | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    expected_dimension: int | None = None,
) -> dict[str, Any]:
    """Load and authenticate a CPU-only adapter checkpoint."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid plastic-memory checkpoint payload") from exc
    if not isinstance(raw, dict) or set(raw) != _PAYLOAD_KEYS:
        raise ValueError("invalid plastic-memory checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported plastic-memory checkpoint format")
    adapter_id = _nonempty_string(raw["adapter_id"], name="adapter_id")
    base_fingerprint = _nonempty_string(raw["base_fingerprint"], name="base_fingerprint")
    created_at = _nonempty_string(raw["created_at"], name="created_at")
    config = _json_object(raw["config"], name="config")
    metadata = _json_object(raw["metadata"], name="metadata")
    source_episode_ids = _episode_ids(raw["source_episode_ids"])
    tags = _tags(raw["tags"])
    if type(raw["update_count"]) is not int or raw["update_count"] < 0:
        raise ValueError("invalid checkpoint update_count")
    importance = _importance(raw["importance"])
    operation_counts = _operation_counts(raw["operation_counts"])
    if operation_counts["writes"] != raw["update_count"]:
        raise ValueError("checkpoint update_count differs from operation counts")
    if _metadata_digest(raw) != raw["metadata_sha256"]:
        raise ValueError("checkpoint metadata integrity digest mismatch")
    state = raw["state_dict"]
    if isinstance(state, Mapping) and any(
        isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu"
        for tensor in state.values()
    ):
        raise ValueError("checkpoint state tensors must be on CPU")
    checked = _checked_state(state)
    if raw["state_schema_sha256"] != state_schema_sha256(checked):
        raise ValueError("checkpoint state schema digest mismatch")
    if raw["state_content_sha256"] != state_content_sha256(checked):
        raise ValueError("checkpoint state content digest mismatch")
    _expect_equal(base_fingerprint, expected_base_fingerprint, name="base fingerprint")
    _expect_equal(adapter_id, expected_adapter_id, name="adapter id")
    _expect_equal(source_episode_ids, None if expected_source_episode_ids is None else _episode_ids(expected_source_episode_ids), name="source episode IDs")
    _expect_equal(created_at, expected_created_at, name="created_at")
    _expect_equal(raw["update_count"], expected_update_count, name="update_count")
    _expect_equal(tags, None if expected_tags is None else _tags(expected_tags), name="tags")
    if expected_importance is not None:
        _expect_equal(importance, _importance(expected_importance), name="importance")
    if expected_config is not None and config != _json_object(expected_config, name="expected_config"):
        raise ValueError("checkpoint config mismatch")
    if expected_metadata is not None and metadata != _json_object(expected_metadata, name="expected_metadata"):
        raise ValueError("checkpoint metadata mismatch")
    if expected_dimension is not None:
        if type(expected_dimension) is not int or expected_dimension < 1:
            raise ValueError("expected_dimension must be a positive integer")
        dimension_keys = {"dimension", "dim", "embedding_dim", "adapter_dim", "hidden_dim"}
        found = [config[key] for key in dimension_keys if key in config]
        if not found or any(type(value) is not int or value != expected_dimension for value in found):
            raise ValueError("checkpoint dimension mismatch")
    if adapter is not None:
        if not isinstance(adapter, torch.nn.Module):
            raise TypeError("adapter must be a torch.nn.Module")
        # A generic module may have no config attribute; its explicit config
        # was already authenticated above.  PlasticAdapter instances expose
        # config and are bound to it here.
        if getattr(adapter, "config", None) is not None:
            adapter_config = _resolve_config(adapter, None)
            if adapter_config != config:
                raise ValueError("checkpoint config differs from adapter config")
        adapter_base = getattr(adapter, "base_fingerprint", None)
        adapter_name = getattr(adapter, "adapter_id", None)
        if adapter_base is not None and adapter_base != base_fingerprint:
            raise ValueError("checkpoint base fingerprint differs from adapter")
        if adapter_name is not None and adapter_name != adapter_id:
            raise ValueError("checkpoint adapter id differs from adapter")
        expected_state = _checked_state(adapter.state_dict())
        if state_schema_sha256(expected_state) != raw["state_schema_sha256"]:
            raise ValueError("checkpoint state schema differs from adapter")
        try:
            adapter.load_state_dict(checked, strict=True)
        except RuntimeError as exc:
            raise ValueError("checkpoint state differs from adapter dimensions") from exc
        restore_counts = getattr(adapter, "restore_counts", None)
        if callable(restore_counts):
            restore_counts(reads=operation_counts["reads"], writes=operation_counts["writes"])
    return {
        "format": raw["format"],
        "adapter_id": adapter_id,
        "config": config,
        "base_fingerprint": base_fingerprint,
        "source_episode_ids": source_episode_ids,
        "created_at": created_at,
        "update_count": raw["update_count"],
        "tags": tags,
        "importance": importance,
        "operation_counts": operation_counts,
        "metadata": metadata,
        "state_schema_sha256": raw["state_schema_sha256"],
        "state_content_sha256": raw["state_content_sha256"],
        "metadata_sha256": raw["metadata_sha256"],
        "state_dict": checked,
    }


# Short aliases keep integration convenient while the explicit names make the
# adapter-specific boundary clear to callers and reviewers.
save_adapter_checkpoint = save_plastic_memory_checkpoint
load_adapter_checkpoint = load_plastic_memory_checkpoint
save_checkpoint = save_plastic_memory_checkpoint
load_checkpoint = load_plastic_memory_checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "load_adapter_checkpoint",
    "load_checkpoint",
    "load_plastic_memory_checkpoint",
    "save_adapter_checkpoint",
    "save_checkpoint",
    "save_plastic_memory_checkpoint",
    "state_content_sha256",
    "state_schema_sha256",
]
