"""Weights-only checkpoints for the factorized explicit-slot model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .concept_model import ConceptVocabulary
from .explicit_slot_checkpoint import (
    DATASET_VERSION, SLOT_CLASS_COUNT, SLOT_ORDER, _atomic_save, _canonical,
    _finite_state, _model_config, _vocab_from_dict, _vocab_to_dict,
)
from .factorized_slot_model import FactorizedSlotModel
from .tensorizer import SemanticTensorizer
from .toy_checkpoint import BYTE_SPEC
from .toy_experiment import ToyModel

CHECKPOINT_VERSION = "norishio-factorized-slot-checkpoint-1.0"


@dataclass(frozen=True)
class LoadedFactorizedSlotCheckpoint:
    model: FactorizedSlotModel
    tensorizer: SemanticTensorizer
    vocabulary: ConceptVocabulary
    dataset_version: str


def _tensor_digest(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        # Avoid a NumPy dependency in the CPU-first runtime.
        h.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return h.hexdigest()


def _manifest(config: Mapping[str, Any], vocabulary: Mapping[str, Any],
              tensorizer: Mapping[str, Any], state: Mapping[str, torch.Tensor]) -> dict[str, str]:
    metadata = _canonical({"config": config, "vocabulary": vocabulary, "tensorizer": tensorizer})
    return {"sha256": hashlib.sha256(metadata).hexdigest(), "state_sha256": _tensor_digest(state)}


def _validate_model(model: Any, tensorizer: SemanticTensorizer,
                    vocabulary: ConceptVocabulary) -> dict[str, Any]:
    if not isinstance(model, FactorizedSlotModel):
        raise TypeError("model must be a FactorizedSlotModel")
    if _vocab_to_dict(getattr(model, "vocabulary", None)) != _vocab_to_dict(vocabulary):
        raise ValueError("model and checkpoint vocabulary mismatch")
    config = _model_config(model, tensorizer)
    projection = model.decoder.concept_projection
    if (type(projection).__name__ != "FactorizedProjection" or
            getattr(projection, "in_features", None) != 43 or
            getattr(projection, "out_features", None) != 32):
        raise ValueError("model must use a factorized 43-to-32 projection")
    for name in ("base_proj", "participant_proj", "time_proj"):
        if not hasattr(projection, name):
            raise ValueError("factorized projection blocks are incomplete")
    if (projection.base_proj.in_features, projection.base_proj.out_features) != (33, 32) or \
            projection.base_proj.bias is None:
        raise ValueError("invalid factorized base projection")
    for name in ("participant_proj", "time_proj"):
        branch = getattr(projection, name)
        if (branch.in_features, branch.out_features) != (5, 32) or branch.bias is not None:
            raise ValueError("invalid factorized slot projection")
    return config


def save_factorized_checkpoint(path: str | Path, model: FactorizedSlotModel,
                               tensorizer: SemanticTensorizer,
                               vocabulary: ConceptVocabulary, *, dataset_version: str) -> None:
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version!r}")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("factorized checkpoints currently support float32 models only")
    config = _validate_model(model, tensorizer, vocabulary)
    vocab = _vocab_to_dict(vocabulary)
    tensorizer_dict = tensorizer.to_dict()
    state = _finite_state(model)
    payload = {
        "format": CHECKPOINT_VERSION, "dataset_version": dataset_version,
        "byte_spec": dict(BYTE_SPEC), "config": config, "vocabulary": vocab,
        "tensorizer": tensorizer_dict, "factorized_state": state,
        "manifest": _manifest(config, vocab, tensorizer_dict, state),
    }
    _atomic_save(path, payload)


def _check_config(config: Any, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("invalid factorized model config")
    required = {"pathway", "hidden_dim", "vocab_size", "conditioning_mode", "encoder"}
    if set(config) != required or config["pathway"] != "C" or config["hidden_dim"] != 32:
        raise ValueError("invalid factorized model config")
    if config["vocab_size"] != BYTE_SPEC["size"] or config["conditioning_mode"] != "per_step_additive":
        raise ValueError("invalid factorized model config")
    encoder = config["encoder"]
    if not isinstance(encoder, dict) or encoder.get("hidden_dim") != 32:
        raise ValueError("invalid factorized encoder config")
    if encoder.get("vocab_sizes") != tensorizer.vocab_sizes:
        raise ValueError("factorized encoder config mismatch")
    return config


def load_factorized_checkpoint(path: str | Path, *, expected_dataset_version: str | None = None,
                               expected_tensorizer: SemanticTensorizer | None = None,
                               expected_vocabulary: ConceptVocabulary | None = None) -> LoadedFactorizedSlotCheckpoint:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "dataset_version", "byte_spec", "config", "vocabulary", "tensorizer", "factorized_state", "manifest"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid factorized checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION or raw["dataset_version"] != DATASET_VERSION:
        raise ValueError("unsupported checkpoint or dataset version")
    if expected_dataset_version is not None and raw["dataset_version"] != expected_dataset_version:
        raise ValueError("dataset version mismatch")
    if _canonical(raw["byte_spec"]) != _canonical(BYTE_SPEC):
        raise ValueError("byte specification mismatch")
    config, vocab_raw, tensor_raw, state = raw["config"], raw["vocabulary"], raw["tensorizer"], raw["factorized_state"]
    vocab = _vocab_from_dict(vocab_raw)
    tensorizer = SemanticTensorizer.from_dict(tensor_raw)
    if expected_tensorizer is not None and tensorizer.to_dict() != expected_tensorizer.to_dict():
        raise ValueError("tensorizer mismatch")
    if expected_vocabulary is not None and _vocab_to_dict(vocab) != _vocab_to_dict(expected_vocabulary):
        raise ValueError("vocabulary mismatch")
    config = _check_config(config, tensorizer)
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("factorized_state must contain only string keys and tensors")
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not bool(torch.isfinite(v).all()))
           for v in state.values()):
        raise ValueError("factorized_state must contain finite float32 tensors")
    manifest = raw["manifest"]
    if not isinstance(manifest, dict) or manifest != _manifest(config, vocab_raw, tensor_raw, state):
        raise ValueError("checkpoint metadata or state integrity digest mismatch")
    base = ToyModel("C", tensorizer, vocab, hidden_dim=32, conditioning_mode="per_step_additive")
    model = FactorizedSlotModel(base, vocab)
    if _model_config(model, tensorizer) != config:
        raise ValueError("factorized model config does not match reconstructed model")
    expected = model.state_dict()
    if set(state) != set(expected) or any(v.dtype != expected[k].dtype or tuple(v.shape) != tuple(expected[k].shape) for k, v in state.items()):
        raise ValueError("checkpoint factorized_state does not match model config")
    try:
        model.load_state_dict({k: v.detach().cpu() for k, v in state.items()}, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint factorized_state does not match model config") from exc
    return LoadedFactorizedSlotCheckpoint(model.eval(), tensorizer, vocab, raw["dataset_version"])


__all__ = ["DATASET_VERSION", "CHECKPOINT_VERSION", "LoadedFactorizedSlotCheckpoint", "save_factorized_checkpoint", "load_factorized_checkpoint"]
