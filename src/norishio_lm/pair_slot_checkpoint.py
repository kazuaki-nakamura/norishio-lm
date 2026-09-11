"""Strict weights-only checkpoints for :class:`PairSlotModel`."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .concept_model import ConceptVocabulary
from .explicit_slot_checkpoint import DATASET_VERSION, _atomic_save, _canonical, _vocab_from_dict, _vocab_to_dict
from .local_slot_checkpoint import _config as _local_config, _finite_state, _state_digest
from .pair_slot_model import PairSlotModel
from .local_slot_model import _kept_indices
from .tensorizer import SemanticTensorizer
from .toy_checkpoint import BYTE_SPEC
from .toy_experiment import ToyModel

CHECKPOINT_VERSION = "norishio-pair-slot-checkpoint-1.0"


@dataclass(frozen=True)
class LoadedPairSlotCheckpoint:
    model: PairSlotModel
    tensorizer: SemanticTensorizer
    vocabulary: ConceptVocabulary
    dataset_version: str


def _config(model: PairSlotModel, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    config = _local_config(model, tensorizer)
    if model.decoder.local is not True:
        raise ValueError("pair model requires local causal conditioning")
    if model.decoder.concept_projection.kept_old_indices != _kept_indices(model.vocabulary):
        raise ValueError("pair model requires canonical retained concept indices")
    if not isinstance(model, PairSlotModel) or getattr(model, "pair_seed", None) != 28:
        raise ValueError("model must use PairSlotModel with pair seed 28")
    if not isinstance(getattr(model, "pair_head", None), torch.nn.Linear) or \
            (model.pair_head.in_features, model.pair_head.out_features) != (32, 25):
        raise ValueError("pair model must use Linear(32, 25) pair head")
    config["pair_seed"] = 28
    config["pair_head"] = {"in_features": 32, "out_features": 25, "bias": True}
    return config


def _metadata(model: PairSlotModel) -> dict[str, Any]:
    projection = model.decoder.concept_projection
    return {
        "decoder_class": type(model.decoder).__name__,
        "projection_class": type(projection).__name__,
        "pair_head_class": type(model.pair_head).__name__,
        "pair_shape": [5, 5],
        "pair_seed": 28,
        "independent_heads": ["participant", "time"],
    }


def _metadata_digest(config: Mapping[str, Any], vocabulary: Mapping[str, Any],
                     tensorizer: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"config": config, "vocabulary": vocabulary,
                                      "tensorizer": tensorizer, "metadata": metadata})).hexdigest()


def save_pair_checkpoint(path: str | Path, model: PairSlotModel,
                         tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
                         *, dataset_version: str) -> None:
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version!r}")
    if _vocab_to_dict(getattr(model, "vocabulary", None)) != _vocab_to_dict(vocabulary):
        raise ValueError("model and checkpoint vocabulary mismatch")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("pair checkpoints currently support float32 models only")
    config, metadata = _config(model, tensorizer), _metadata(model)
    vocab, tensorizer_dict = _vocab_to_dict(vocabulary), tensorizer.to_dict()
    state = _finite_state(model)
    payload = {"format": CHECKPOINT_VERSION, "dataset_version": dataset_version,
               "byte_spec": dict(BYTE_SPEC), "config": config, "vocabulary": vocab,
               "tensorizer": tensorizer_dict, "metadata": metadata,
               "manifest": {"sha256": _metadata_digest(config, vocab, tensorizer_dict, metadata),
                             "state_sha256": _state_digest(state)}, "state_dict": state}
    _atomic_save(path, payload)


def load_pair_checkpoint(path: str | Path, *, expected_dataset_version: str | None = None,
                         expected_tensorizer: SemanticTensorizer | None = None,
                         expected_vocabulary: ConceptVocabulary | None = None) -> LoadedPairSlotCheckpoint:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "dataset_version", "byte_spec", "config", "vocabulary", "tensorizer",
                "metadata", "manifest", "state_dict"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid pair checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION or raw["dataset_version"] != DATASET_VERSION:
        raise ValueError("unsupported checkpoint or dataset version")
    if expected_dataset_version is not None and raw["dataset_version"] != expected_dataset_version:
        raise ValueError("dataset version mismatch")
    if _canonical(raw["byte_spec"]) != _canonical(BYTE_SPEC):
        raise ValueError("byte specification mismatch")
    config, vocab_raw, tensor_raw, metadata = raw["config"], raw["vocabulary"], raw["tensorizer"], raw["metadata"]
    vocab, tensorizer = _vocab_from_dict(vocab_raw), SemanticTensorizer.from_dict(tensor_raw)
    if expected_tensorizer is not None and tensorizer.to_dict() != expected_tensorizer.to_dict():
        raise ValueError("tensorizer mismatch")
    if expected_vocabulary is not None and _vocab_to_dict(vocab) != _vocab_to_dict(expected_vocabulary):
        raise ValueError("vocabulary mismatch")
    manifest = raw["manifest"]
    if not isinstance(manifest, dict) or set(manifest) != {"sha256", "state_sha256"}:
        raise ValueError("invalid checkpoint manifest")
    state = raw["state_dict"]
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("state_dict must contain only string keys and tensors")
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not bool(torch.isfinite(v).all())) for v in state.values()):
        raise ValueError("state_dict must contain finite float32 tensors")
    if (manifest["sha256"] != _metadata_digest(config, vocab_raw, tensor_raw, metadata)
            or manifest["state_sha256"] != _state_digest(state)):
        raise ValueError("checkpoint metadata or state integrity digest mismatch")
    if not isinstance(config, dict) or config.get("pair_seed") != 28:
        raise ValueError("invalid pair model config")
    with torch.random.fork_rng(devices=[]):
        initial = ToyModel(config["pathway"], tensorizer, vocab, hidden_dim=config["hidden_dim"],
                           conditioning_mode=config["conditioning_mode"])
        model = PairSlotModel(initial, vocab, new_seed=20, pair_seed=28)
    if _config(model, tensorizer) != config or _metadata(model) != metadata:
        raise ValueError("checkpoint architecture metadata does not match reconstructed model")
    expected = model.state_dict()
    if set(state) != set(expected) or any(v.dtype != expected[k].dtype or tuple(v.shape) != tuple(expected[k].shape) for k, v in state.items()):
        raise ValueError("checkpoint state_dict does not match model config")
    try:
        model.load_state_dict({k: v.detach().cpu() for k, v in state.items()}, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict does not match model config") from exc
    return LoadedPairSlotCheckpoint(model.eval(), tensorizer, vocab, raw["dataset_version"])


__all__ = ["DATASET_VERSION", "CHECKPOINT_VERSION", "LoadedPairSlotCheckpoint",
           "save_pair_checkpoint", "load_pair_checkpoint"]
