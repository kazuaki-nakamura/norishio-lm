"""Weights-only checkpoints for the explicit participant/time slot model."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .concept_model import ConceptVocabulary
from .tensorizer import SemanticTensorizer
from .toy_checkpoint import BYTE_SPEC, _canonical, _vocab_from_dict, _vocab_to_dict
from .toy_experiment import ToyModel

DATASET_VERSION = "norishio-toy-1.0"
CHECKPOINT_VERSION = "norishio-explicit-slot-checkpoint-1.0"
SLOT_ORDER = ("participant", "time")
SLOT_CLASS_COUNT = 5


@dataclass(frozen=True)
class LoadedExplicitSlotCheckpoint:
    model: Any
    tensorizer: SemanticTensorizer
    vocabulary: ConceptVocabulary
    dataset_version: str


def _digest(config: Mapping[str, Any], vocabulary: Mapping[str, Any],
            tensorizer: Mapping[str, Any], slots: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"config": config, "vocabulary": vocabulary,
                                      "tensorizer": tensorizer, "slots": slots})).hexdigest()


def _finite_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if not isinstance(state, dict) or not state:
        raise ValueError("model state_dict must not be empty")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("state_dict must contain only string keys and tensors")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("model state contains non-finite tensor")
        result[key] = value.detach().cpu().clone()
    return result


def _slot_metadata(vocabulary: ConceptVocabulary) -> dict[str, Any]:
    # Vocabulary IDs are 1..5; heads use zero-based IDs 0..4. Keep both
    # mappings explicit in the integrity-covered metadata.
    mappings: dict[str, dict[str, int]] = {}
    head_mappings: dict[str, dict[str, int]] = {}
    for name in SLOT_ORDER:
        values = vocabulary.fields.get(name, {})
        if set(values.values()) != set(range(1, SLOT_CLASS_COUNT + 1)):
            raise ValueError(f"{name} vocabulary must define exactly five IDs 1..5")
        mappings[name] = {repr(value): int(identifier) for value, identifier in
                          sorted(values.items(), key=lambda item: item[1])}
        head_mappings[name] = {repr(value): int(identifier) - 1 for value, identifier in
                               sorted(values.items(), key=lambda item: item[1])}
    return {"order": list(SLOT_ORDER), "class_count": SLOT_CLASS_COUNT,
            "class_ids": mappings, "head_ids": head_mappings}


def _model_config(model: Any, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    decoder = model.decoder
    hidden = getattr(decoder, "hidden_dim", None)
    if type(hidden) is not int or hidden < 1:
        raise ValueError("invalid explicit model hidden dimension")
    if decoder.embedding.num_embeddings != BYTE_SPEC["size"]:
        raise ValueError("explicit decoder must use the 260-token byte vocabulary")
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("explicit slot model requires an encoder")
    cfg = encoder.config
    if cfg.vocab_sizes != tensorizer.vocab_sizes:
        raise ValueError("model encoder vocabulary sizes do not match tensorizer")
    projection = getattr(decoder, "concept_projection", None)
    concept_dim = sum(len(values) + 1 for values in model.decoder.bottleneck.vocabulary.fields.values())
    if concept_dim != 33 or projection is None or projection.in_features != concept_dim + 2 * SLOT_CLASS_COUNT:
        raise ValueError("explicit decoder projection must accept 43 concept/slot values")
    if getattr(decoder, "conditioning_mode", None) != "per_step_additive" or hidden != 32:
        raise ValueError("explicit model must use hidden_dim=32 and per_step_additive conditioning")
    heads = getattr(model, "slot_heads", None)
    if heads is None or tuple(getattr(heads, "keys", lambda: ())()) != SLOT_ORDER:
        raise ValueError("explicit model slot heads must be participant,time")
    if any(getattr(heads[name], "out_features", None) != SLOT_CLASS_COUNT for name in SLOT_ORDER):
        raise ValueError("explicit slot heads must have five classes")
    return {"pathway": "C", "hidden_dim": hidden, "vocab_size": decoder.embedding.num_embeddings,
            "conditioning_mode": getattr(decoder, "conditioning_mode", "per_step_additive"),
            "encoder": {"embedding_dim": cfg.embedding_dim, "hidden_dim": cfg.hidden_dim,
                        "vocab_sizes": dict(cfg.vocab_sizes)}}


def _atomic_save(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
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


def save_explicit_checkpoint(path: str | os.PathLike[str], model: Any,
                             tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
                             *, dataset_version: str) -> None:
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version!r}")
    model_vocab = getattr(getattr(model.decoder, "bottleneck", None), "vocabulary", None)
    if model_vocab is None or _vocab_to_dict(model_vocab) != _vocab_to_dict(vocabulary):
        raise ValueError("model and checkpoint vocabulary mismatch")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("explicit checkpoints currently support float32 models only")
    config = _model_config(model, tensorizer)
    vocab = _vocab_to_dict(vocabulary)
    tensorizer_dict = tensorizer.to_dict()
    slots = _slot_metadata(vocabulary)
    payload = {"format": CHECKPOINT_VERSION, "dataset_version": dataset_version,
               "byte_spec": dict(BYTE_SPEC), "config": config, "vocabulary": vocab,
               "tensorizer": tensorizer_dict, "slots": slots,
               "manifest": {"sha256": _digest(config, vocab, tensorizer_dict, slots)},
               "state_dict": _finite_state(model)}
    _atomic_save(path, payload)


def _check_config(config: Any, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    if not isinstance(config, dict) or set(config) != {"pathway", "hidden_dim", "vocab_size", "conditioning_mode", "encoder"}:
        raise ValueError("invalid explicit model config")
    if config["pathway"] != "C" or type(config["hidden_dim"]) is not int or config["hidden_dim"] < 1:
        raise ValueError("invalid explicit model config")
    if (config["vocab_size"] != BYTE_SPEC["size"] or
            config["conditioning_mode"] != "per_step_additive" or config["hidden_dim"] != 32):
        raise ValueError("invalid explicit model config")
    enc = config["encoder"]
    if not isinstance(enc, dict) or set(enc) != {"embedding_dim", "hidden_dim", "vocab_sizes"}:
        raise ValueError("invalid explicit encoder config")
    if (type(enc["embedding_dim"]) is not int or type(enc["hidden_dim"]) is not int or
            enc["embedding_dim"] < 1 or enc["hidden_dim"] < 1 or
            not isinstance(enc["vocab_sizes"], dict)):
        raise ValueError("invalid explicit encoder dimensions")
    if enc["hidden_dim"] != config["hidden_dim"] or enc["vocab_sizes"] != tensorizer.vocab_sizes:
        raise ValueError("explicit encoder config mismatch")
    return config


def load_explicit_checkpoint(path: str | os.PathLike[str], *, expected_dataset_version: str | None = None,
                             expected_tensorizer: SemanticTensorizer | None = None,
                             expected_vocabulary: ConceptVocabulary | None = None) -> LoadedExplicitSlotCheckpoint:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format", "dataset_version", "byte_spec", "config", "vocabulary", "tensorizer", "slots", "manifest", "state_dict"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("invalid explicit checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION or raw["dataset_version"] != DATASET_VERSION:
        raise ValueError("unsupported checkpoint or dataset version")
    if expected_dataset_version is not None and raw["dataset_version"] != expected_dataset_version:
        raise ValueError("dataset version mismatch")
    if _canonical(raw["byte_spec"]) != _canonical(BYTE_SPEC):
        raise ValueError("byte specification mismatch")
    config, vocab_raw, tensor_raw, slots = raw["config"], raw["vocabulary"], raw["tensorizer"], raw["slots"]
    if not isinstance(raw["manifest"], dict) or raw["manifest"].get("sha256") != _digest(config, vocab_raw, tensor_raw, slots):
        raise ValueError("checkpoint metadata integrity digest mismatch")
    vocab = _vocab_from_dict(vocab_raw)
    tensorizer = SemanticTensorizer.from_dict(tensor_raw)
    if expected_tensorizer is not None and tensorizer.to_dict() != expected_tensorizer.to_dict():
        raise ValueError("tensorizer mismatch")
    if expected_vocabulary is not None and _vocab_to_dict(vocab) != _vocab_to_dict(expected_vocabulary):
        raise ValueError("vocabulary mismatch")
    if slots != _slot_metadata(vocab) or slots.get("class_count") != SLOT_CLASS_COUNT:
        raise ValueError("invalid slot metadata")
    config = _check_config(config, tensorizer)
    if slots.get("order") != list(SLOT_ORDER):
        raise ValueError("invalid slot order")
    from .explicit_slot_model import ExplicitSlotModel
    with torch.random.fork_rng(devices=[]):
        base = ToyModel("C", tensorizer, vocab, hidden_dim=config["hidden_dim"],
                        conditioning_mode="per_step_additive")
        model = ExplicitSlotModel(base, vocab)
    if model.encoder.config.embedding_dim != config["encoder"]["embedding_dim"]:
        raise ValueError("explicit encoder embedding dimension mismatch")
    state = raw["state_dict"]
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("state_dict must contain only string keys and tensors")
    expected = model.state_dict()
    if set(state) != set(expected) or any(v.dtype != expected[k].dtype or tuple(v.shape) != tuple(expected[k].shape)
                                          for k, v in state.items()):
        raise ValueError("checkpoint state_dict does not match explicit model config")
    if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("checkpoint state contains non-finite tensor")
    try:
        model.load_state_dict({k: v.detach().cpu() for k, v in state.items()}, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict does not match explicit model config") from exc
    return LoadedExplicitSlotCheckpoint(model.eval(), tensorizer, vocab, raw["dataset_version"])


__all__ = ["DATASET_VERSION", "CHECKPOINT_VERSION", "SLOT_ORDER", "SLOT_CLASS_COUNT",
           "LoadedExplicitSlotCheckpoint", "save_explicit_checkpoint", "load_explicit_checkpoint"]
