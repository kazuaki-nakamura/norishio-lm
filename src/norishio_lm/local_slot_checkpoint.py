"""Weights-only checkpoints for the local duplicate-removed slot model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from .concept_model import ConceptVocabulary
from .explicit_slot_checkpoint import (
    DATASET_VERSION, _atomic_save, _canonical, _vocab_from_dict, _vocab_to_dict,
)
from .tensorizer import SemanticTensorizer
from .toy_checkpoint import BYTE_SPEC
from .toy_experiment import ToyModel

CHECKPOINT_VERSION = "norishio-local-slot-checkpoint-1.0"
LOCAL_RULE_VERSION = "duplicate-removed-local-1.0"
_PATHWAYS = {"C"}


@dataclass(frozen=True)
class LoadedLocalSlotCheckpoint:
    model: Any
    tensorizer: SemanticTensorizer
    vocabulary: ConceptVocabulary
    dataset_version: str


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def _metadata_digest(config: Mapping[str, Any], vocabulary: Mapping[str, Any],
                     tensorizer: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"config": config, "vocabulary": vocabulary,
                                      "tensorizer": tensorizer, "metadata": metadata})).hexdigest()


def _finite_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if not isinstance(state, dict) or not state:
        raise ValueError("model state_dict must not be empty")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("state_dict must contain only string keys and tensors")
        if value.is_floating_point() and (value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError("model state must contain finite float32 tensors")
        result[key] = value.detach().cpu().clone()
    return result


def _retained_indices(model: Any) -> tuple[int, ...]:
    projection = getattr(getattr(model, "decoder", None), "concept_projection", None)
    values = getattr(projection, "kept_indices",
                      getattr(projection, "kept_old_indices",
                              getattr(projection, "keptindices",
                                      getattr(model, "kept_old_indices", None))))
    if not isinstance(values, tuple) or any(type(item) is not int for item in values):
        raise ValueError("local projection must expose immutable tuple kept_indices")
    if len(set(values)) != len(values) or any(item < 0 or item >= 43 for item in values):
        raise ValueError("invalid local projection kept_indices")
    return values


def _config(model: Any, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    decoder = getattr(model, "decoder", None)
    encoder = getattr(model, "encoder", None)
    if getattr(model, "pathway", None) not in _PATHWAYS or decoder is None or encoder is None:
        raise ValueError("local slot model must be an encoded C or D model")
    if getattr(decoder, "embedding", None) is None or decoder.embedding.num_embeddings != BYTE_SPEC["size"]:
        raise ValueError("local decoder must use the 260-token byte vocabulary")
    local = getattr(decoder, "local", None)
    if type(local) is not bool:
        raise ValueError("local decoder flag must be bool")
    cfg = encoder.config
    if cfg.vocab_sizes != tensorizer.vocab_sizes:
        raise ValueError("model encoder vocabulary sizes do not match tensorizer")
    projection = getattr(decoder, "concept_projection", None)
    if (getattr(projection, "in_features", None) != 43 or
            getattr(projection, "out_features", None) != 32):
        raise ValueError("local projection must expose 43-to-32 transport")
    base = getattr(projection, "base_proj", None)
    slots = [getattr(projection, name, None) for name in ("participant_proj", "time_proj")]
    if (base is None or (base.in_features, base.out_features) != (21, 32) or base.bias is None or
            any(branch is None or (branch.in_features, branch.out_features) != (5, 32) or branch.bias is not None
                for branch in slots)):
        raise ValueError("invalid local projection blocks")
    rule_version = getattr(model, "local_rule_version", getattr(decoder, "local_rule_version", LOCAL_RULE_VERSION))
    if not isinstance(rule_version, str) or not rule_version:
        raise ValueError("local rule version must be a non-empty string")
    return {
        "pathway": model.pathway, "hidden_dim": int(getattr(decoder, "hidden_dim", 32)),
        "vocab_size": decoder.embedding.num_embeddings,
        "conditioning_mode": getattr(decoder, "conditioning_mode", None),
        "encoder": {"embedding_dim": cfg.embedding_dim, "hidden_dim": cfg.hidden_dim,
                    "vocab_sizes": dict(cfg.vocab_sizes)},
        "local": local, "rule_version": rule_version,
        **({"gradient_routing": getattr(model, "gradient_routing")}
           if getattr(model, "gradient_routing", "end_to_end") != "end_to_end" else {}),
    }


def _metadata(model: Any) -> dict[str, Any]:
    projection = model.decoder.concept_projection
    indices = _retained_indices(model)
    from .local_slot_model import PREFIXES
    result = {
        "kept_indices": list(indices),
        "kept_indices_type": "tuple",
        "decoder_class": type(model.decoder).__name__,
        "projection_class": type(projection).__name__,
        "causal_prefixes": [list(prefix) for prefix in PREFIXES],
        "slot_width": 6,
    }
    if getattr(model, "gradient_routing", "end_to_end") != "end_to_end":
        result["gradient_routing"] = model.gradient_routing
    return result


def save_local_checkpoint(path: str | Path, model: Any, tensorizer: SemanticTensorizer,
                          vocabulary: ConceptVocabulary, *, dataset_version: str) -> None:
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version!r}")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("local checkpoints currently support float32 models only")
    model_vocab = getattr(model, "vocabulary", None)
    if model_vocab is None or _vocab_to_dict(model_vocab) != _vocab_to_dict(vocabulary):
        raise ValueError("model and checkpoint vocabulary mismatch")
    config = _config(model, tensorizer)
    metadata = _metadata(model)
    vocab, tensorizer_dict = _vocab_to_dict(vocabulary), tensorizer.to_dict()
    state = _finite_state(model)
    payload = {"format": CHECKPOINT_VERSION, "dataset_version": dataset_version,
               "byte_spec": dict(BYTE_SPEC), "config": config, "vocabulary": vocab,
               "tensorizer": tensorizer_dict, "metadata": metadata,
               "manifest": {"sha256": _metadata_digest(config, vocab, tensorizer_dict, metadata),
                             "state_sha256": _state_digest(state)}, "state_dict": state}
    _atomic_save(path, payload)


def _reconstruct(config: Mapping[str, Any], tensorizer: SemanticTensorizer,
                 vocabulary: ConceptVocabulary) -> Any:
    from .local_slot_model import DuplicateRemovedModel
    pathway = config["pathway"]
    with torch.random.fork_rng(devices=[]):
        initial = ToyModel(pathway, tensorizer, vocabulary, hidden_dim=config["hidden_dim"],
                           conditioning_mode=config["conditioning_mode"])
        return DuplicateRemovedModel(initial, vocabulary, local=config["local"],
                                     gradient_routing=config.get("gradient_routing", "end_to_end"))


def load_local_checkpoint(path: str | Path, *, expected_dataset_version: str | None = None,
                          expected_tensorizer: SemanticTensorizer | None = None,
                          expected_vocabulary: ConceptVocabulary | None = None) -> LoadedLocalSlotCheckpoint:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "dataset_version", "byte_spec", "config", "vocabulary", "tensorizer",
                "metadata", "manifest", "state_dict"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid local checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION or raw["dataset_version"] != DATASET_VERSION:
        raise ValueError("unsupported checkpoint or dataset version")
    if expected_dataset_version is not None and raw["dataset_version"] != expected_dataset_version:
        raise ValueError("dataset version mismatch")
    if _canonical(raw["byte_spec"]) != _canonical(BYTE_SPEC):
        raise ValueError("byte specification mismatch")
    config, vocab_raw, tensor_raw, metadata = raw["config"], raw["vocabulary"], raw["tensorizer"], raw["metadata"]
    if not isinstance(metadata, dict) or metadata.get("kept_indices_type") != "tuple":
        raise ValueError("invalid local metadata")
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
    if manifest["sha256"] != _metadata_digest(config, vocab_raw, tensor_raw, metadata) or manifest["state_sha256"] != _state_digest(state):
        raise ValueError("checkpoint metadata or state integrity digest mismatch")
    if (not isinstance(config, dict) or type(config.get("local")) is not bool or
            config.get("pathway") not in _PATHWAYS):
        raise ValueError("invalid local model config")
    model = _reconstruct(config, tensorizer, vocab)
    if _config(model, tensorizer) != config or _metadata(model) != metadata:
        raise ValueError("checkpoint architecture metadata does not match reconstructed model")
    expected = model.state_dict()
    if set(state) != set(expected) or any(v.dtype != expected[k].dtype or tuple(v.shape) != tuple(expected[k].shape) for k, v in state.items()):
        raise ValueError("checkpoint state_dict does not match model config")
    try:
        model.load_state_dict({k: v.detach().cpu() for k, v in state.items()}, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict does not match model config") from exc
    return LoadedLocalSlotCheckpoint(model.eval(), tensorizer, vocab, raw["dataset_version"])


__all__ = ["DATASET_VERSION", "CHECKPOINT_VERSION", "LoadedLocalSlotCheckpoint",
           "save_local_checkpoint", "load_local_checkpoint"]
