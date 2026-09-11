"""Safe, CPU-only inference checkpoints for the authored toy model.

The checkpoint is a plain mapping containing tensors and primitive containers.
It never serializes a Python model object or relies on pickle reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from .concept_model import CONCEPT_FIELDS, ConceptVocabulary
from .tensorizer import SemanticTensorizer
from .toy_experiment import ToyModel


DATASET_VERSION = "norishio-toy-1.0"
CHECKPOINT_VERSION = "norishio-toy-checkpoint-1.0"
BYTE_SPEC = {
    "version": "norishio-byte-1.0", "PAD": 0, "BOS": 1, "EOS": 2,
    "SEP": 3, "offset": 4, "size": 260,
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: ToyModel
    tensorizer: SemanticTensorizer
    vocabulary: ConceptVocabulary
    constant: torch.Tensor | None
    dataset_version: str


def _tag(value: Any) -> Any:
    """Encode values used as ConceptVocabulary keys without type loss."""
    if value is None or isinstance(value, str):
        return value
    if type(value) is bool:
        return {"__type__": "bool", "value": value}
    if type(value) is int:
        return {"__type__": "int", "value": value}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_tag(item) for item in value]}
    if isinstance(value, list):
        return {"__type__": "list", "items": [_tag(item) for item in value]}
    raise ValueError(f"unsupported vocabulary value type: {type(value).__name__}")


def _untag(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, dict) or set(value) - {"__type__", "value", "items"}:
        raise ValueError("invalid tagged vocabulary value")
    kind = value.get("__type__")
    if kind == "bool" and set(value) == {"__type__", "value"} and type(value["value"]) is bool:
        return value["value"]
    if kind == "int" and set(value) == {"__type__", "value"} and type(value["value"]) is int:
        return value["value"]
    if kind in {"tuple", "list"} and set(value) == {"__type__", "items"} and isinstance(value["items"], list):
        items = [_untag(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    raise ValueError("invalid tagged vocabulary value")


def _vocab_to_dict(vocab: ConceptVocabulary) -> dict[str, Any]:
    def field_map(values: Mapping[Any, int]) -> list[list[Any]]:
        return [[_tag(value), identifier] for value, identifier in sorted(values.items(), key=lambda p: p[1])]
    return {
        "fields": {name: field_map(vocab.fields[name]) for name in CONCEPT_FIELDS},
        "senses": [[value, identifier] for value, identifier in sorted(vocab.senses.items(), key=lambda p: p[1])],
        "sememes": [[value, identifier] for value, identifier in sorted(vocab.sememes.items(), key=lambda p: p[1])],
    }


def _validate_ids(entries: Any, *, name: str, start: int) -> None:
    if not isinstance(entries, list) or any(not isinstance(row, list) or len(row) != 2 for row in entries):
        raise ValueError(f"{name} must be [value, id] entries")
    ids = [row[1] for row in entries]
    if any(type(identifier) is not int for identifier in ids) or sorted(ids) != list(range(start, start + len(ids))):
        raise ValueError(f"{name} IDs must be contiguous")


def _vocab_from_dict(raw: Any) -> ConceptVocabulary:
    if not isinstance(raw, dict) or set(raw) != {"fields", "senses", "sememes"}:
        raise ValueError("invalid vocabulary schema")
    fields_raw = raw["fields"]
    if not isinstance(fields_raw, dict) or set(fields_raw) != set(CONCEPT_FIELDS):
        raise ValueError("vocabulary fields must match concept schema")
    fields: dict[str, dict[Any, int]] = {}
    for name in CONCEPT_FIELDS:
        entries = fields_raw[name]
        _validate_ids(entries, name=f"field {name}", start=1)
        decoded = { _untag(row[0]): row[1] for row in entries }
        if len(decoded) != len(entries):
            raise ValueError(f"field {name} contains duplicate values")
        fields[name] = decoded
    for name in ("senses", "sememes"):
        _validate_ids(raw[name], name=name, start=(1 if name == "senses" else 0))
        if any(type(row[0]) is not str for row in raw[name]):
            raise ValueError(f"{name} values must be strings")
        if len({row[0] for row in raw[name]}) != len(raw[name]):
            raise ValueError(f"{name} contains duplicate values")
    return ConceptVocabulary(fields, {row[0]: row[1] for row in raw["senses"]},
                             {row[0]: row[1] for row in raw["sememes"]})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(config: Mapping[str, Any], vocabulary: Mapping[str, Any], tensorizer: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"config": config, "vocabulary": vocabulary,
                                      "tensorizer": tensorizer})).hexdigest()


def _config(model: ToyModel, tensorizer: SemanticTensorizer) -> dict[str, Any]:
    if model.pathway not in {"A", "B", "C"}:
        raise ValueError("model pathway must be A, B, or C")
    decoder = model.decoder
    if decoder.embedding.num_embeddings != BYTE_SPEC["size"]:
        raise ValueError("toy decoder must use the 260-token byte vocabulary")
    result: dict[str, Any] = {"pathway": model.pathway, "hidden_dim": decoder.hidden_dim,
                              "vocab_size": decoder.embedding.num_embeddings,
                              "conditioning_mode": decoder.conditioning_mode}
    if model.encoder is not None:
        cfg = model.encoder.config
        result["encoder"] = {"embedding_dim": cfg.embedding_dim, "hidden_dim": cfg.hidden_dim,
                              "vocab_sizes": dict(cfg.vocab_sizes)}
        if cfg.vocab_sizes != tensorizer.vocab_sizes:
            raise ValueError("model encoder vocabulary sizes do not match tensorizer")
    else:
        result["encoder"] = None
    return result


def _validate_state(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("state_dict must contain only string keys and tensors")
    return {key: value.detach().cpu() for key, value in state.items()}


def save_checkpoint(path: str | os.PathLike[str], model: ToyModel, tensorizer: SemanticTensorizer,
                    vocabulary: ConceptVocabulary, *, dataset_version: str,
                    constant: torch.Tensor | None = None) -> None:
    """Write an exclusive, atomically published CPU checkpoint."""
    if dataset_version != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version: {dataset_version!r}")
    if not isinstance(model, ToyModel):
        raise TypeError("model must be a ToyModel")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("toy checkpoints currently support float32 models only")
    config = _config(model, tensorizer)
    vocab = _vocab_to_dict(vocabulary)
    tensorizer_dict = tensorizer.to_dict()
    payload: dict[str, Any] = {
        "format": CHECKPOINT_VERSION, "dataset_version": dataset_version,
        "byte_spec": dict(BYTE_SPEC), "config": config, "vocabulary": vocab,
        "tensorizer": tensorizer_dict,
        "manifest": {"sha256": _digest(config, vocab, tensorizer_dict)},
        "metadata": {"condition": "constant" if constant is not None else "predicted"},
        "state_dict": _validate_state(model.state_dict()),
        "constant": None,
    }
    if constant is not None:
        if not isinstance(constant, torch.Tensor) or constant.ndim != 2 or constant.shape[0] != 1:
            raise ValueError("constant must have shape [1, concept_dim]")
        if constant.dtype != model.decoder.embedding.weight.dtype or not constant.is_floating_point() or not bool(torch.isfinite(constant).all()):
            raise ValueError("constant must be finite floating point data")
        if model.decoder.concept_projection is None or constant.shape[1] != model.decoder.concept_projection.in_features:
            raise ValueError("constant conditioning dimension does not match vocabulary")
        payload["constant"] = constant.detach().cpu().clone()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(str(target))
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
        # Windows rename is atomic and refuses to replace an existing target.
        # The explicit existence check gives a stable error before serialization;
        # rename closes the race without exposing a partially written file.
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            # POSIX rename replaces targets; link provides exclusive publication.
            os.link(temporary, target)
            os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(path: str | os.PathLike[str], *, expected_dataset_version: str | None = None,
                    expected_tensorizer: SemanticTensorizer | None = None,
                    expected_vocabulary: ConceptVocabulary | None = None) -> LoadedCheckpoint:
    """Load and validate a CPU checkpoint using weights-only deserialization."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != {"format", "dataset_version", "byte_spec", "config", "vocabulary", "tensorizer", "manifest", "metadata", "state_dict", "constant"}:
        raise ValueError("invalid checkpoint keys")
    if raw["format"] != CHECKPOINT_VERSION or raw["dataset_version"] != DATASET_VERSION:
        raise ValueError("unsupported checkpoint or dataset version")
    if expected_dataset_version is not None and raw["dataset_version"] != expected_dataset_version:
        raise ValueError("dataset version mismatch")
    if not isinstance(raw["byte_spec"], dict) or _canonical(raw["byte_spec"]) != _canonical(BYTE_SPEC):
        raise ValueError("byte specification mismatch")
    config, vocab_raw = raw["config"], raw["vocabulary"]
    if not isinstance(raw["manifest"], dict) or set(raw["manifest"]) != {"sha256"} or raw["manifest"].get("sha256") != _digest(config, vocab_raw, raw["tensorizer"]):
        raise ValueError("checkpoint metadata integrity digest mismatch")
    vocabulary = _vocab_from_dict(vocab_raw)
    tensorizer = SemanticTensorizer.from_dict(raw["tensorizer"])
    if expected_tensorizer is not None and tensorizer.to_dict() != expected_tensorizer.to_dict():
        raise ValueError("tensorizer mismatch")
    if expected_vocabulary is not None and _vocab_to_dict(vocabulary) != _vocab_to_dict(expected_vocabulary):
        raise ValueError("vocabulary mismatch")
    required_config_keys = {"pathway", "hidden_dim", "vocab_size", "encoder"}
    allowed_config_keys = required_config_keys | {"conditioning_mode"}
    if (not isinstance(config, dict) or not required_config_keys <= set(config) or
            not set(config) <= allowed_config_keys or config.get("pathway") not in {"A", "B", "C"} or
            type(config.get("hidden_dim")) is not int or config["hidden_dim"] < 1):
        raise ValueError("invalid model config")
    conditioning_mode = config.get("conditioning_mode", "initial_only")
    if conditioning_mode not in {"initial_only", "per_step_additive"}:
        raise ValueError("invalid conditioning mode")
    if config["pathway"] != "C" and conditioning_mode != "initial_only":
        raise ValueError("conditioning mode is only supported for pathway C")
    encoder_config = config.get("encoder")
    if config["pathway"] == "A" and encoder_config is not None:
        raise ValueError("A pathway cannot have encoder config")
    if config["pathway"] != "A" and not isinstance(encoder_config, dict):
        raise ValueError("encoded pathway requires encoder config")
    if config.get("vocab_size") != BYTE_SPEC["size"]:
        raise ValueError("invalid byte vocabulary size")
    if encoder_config is not None:
        if set(encoder_config) != {"embedding_dim", "hidden_dim", "vocab_sizes"}:
            raise ValueError("invalid encoder config keys")
        if type(encoder_config.get("embedding_dim")) is not int or type(encoder_config.get("hidden_dim")) is not int or encoder_config["embedding_dim"] < 1 or encoder_config["hidden_dim"] < 1:
            raise ValueError("invalid encoder dimensions")
        if encoder_config.get("vocab_sizes") != tensorizer.vocab_sizes:
            raise ValueError("encoder and tensorizer vocabulary sizes differ")
        if encoder_config["hidden_dim"] != config["hidden_dim"]:
            raise ValueError("encoder and decoder hidden dimensions differ")
    model = ToyModel(config["pathway"], tensorizer, vocabulary, hidden_dim=config["hidden_dim"],
                     conditioning_mode=conditioning_mode)
    if encoder_config is not None and encoder_config.get("embedding_dim") != model.encoder.config.embedding_dim:
        raise ValueError("encoder embedding dimension mismatch")
    state = _validate_state(raw["state_dict"])
    expected_state = model.state_dict()
    if any(key in expected_state and value.dtype != expected_state[key].dtype for key, value in state.items()):
        raise ValueError("checkpoint state_dict dtype does not match model config")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint state_dict does not match model config") from exc
    constant = raw["constant"]
    if constant is not None:
        if not isinstance(constant, torch.Tensor) or constant.ndim != 2 or constant.shape[0] != 1:
            raise ValueError("invalid constant conditioning vector")
        if constant.dtype != model.decoder.embedding.weight.dtype or not constant.is_floating_point() or not bool(torch.isfinite(constant).all()):
            raise ValueError("constant must be finite floating point data")
        if model.decoder.concept_projection is None or constant.shape[1] != model.decoder.concept_projection.in_features:
            raise ValueError("constant conditioning dimension does not match vocabulary")
        constant = constant.detach().cpu().clone()
    if not isinstance(raw["metadata"], dict) or set(raw["metadata"]) != {"condition"} or raw["metadata"].get("condition") != ("constant" if constant is not None else "predicted"):
        raise ValueError("condition metadata does not match checkpoint")
    return LoadedCheckpoint(model.eval(), tensorizer, vocabulary, constant, raw["dataset_version"])


__all__ = ["DATASET_VERSION", "BYTE_SPEC", "LoadedCheckpoint", "save_checkpoint", "load_checkpoint"]
