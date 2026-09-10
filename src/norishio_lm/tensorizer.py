"""Deterministic conversion of validated semantic records to channel tensors."""
from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .schema import Provenance, SemanticRecord

try:  # Keep importing schema utilities possible on machines without torch.
    import torch
except ImportError:  # pragma: no cover - exercised only in torch-less installs
    torch = None  # type: ignore[assignment]

CHANNELS = ("surface", "tokens", "morphemes", "characters", "subcharacters",
            "etymology", "senses", "sememes", "concepts", "relations")
CHECKPOINT_VERSION = "1"


@dataclass(frozen=True)
class Feature:
    value: str
    path: str
    provenance: tuple[Provenance, ...]
    sense_id: str | None = None


@dataclass(frozen=True)
class ChannelBatch:
    ids: torch.Tensor
    mask: torch.Tensor
    features: tuple[tuple[Feature, ...], ...]
    # Record-level declarations, one tuple per row even when no features exist.
    # Empty outer tuple means not supplied by a manual/legacy constructor.
    layer_provenance: tuple[tuple[Provenance, ...], ...] = ()

    def to(self, device: str | torch.device) -> ChannelBatch:
        return ChannelBatch(self.ids.to(device), self.mask.to(device), self.features,
                            self.layer_provenance)


@dataclass(frozen=True)
class SemanticBatch:
    channels: dict[str, ChannelBatch]

    def to(self, device: str | torch.device) -> SemanticBatch:
        return SemanticBatch({name: batch.to(device) for name, batch in self.channels.items()})


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _vocab_key(value: str) -> str:
    """Encode observed strings so reserved checkpoint keys remain unambiguous."""
    return _json(value)


class SemanticTensorizer:
    """Fit fixed, independently ablatable vocabularies and encode records."""

    def __init__(self, vocabularies: Mapping[str, Mapping[str, int]]) -> None:
        self._vocabularies = self._validate_vocabularies(vocabularies)

    @staticmethod
    def _validate_vocabularies(vocabularies: Mapping[str, Mapping[str, int]]) -> Mapping[str, Mapping[str, int]]:
        if not isinstance(vocabularies, Mapping):
            raise ValueError("checkpoint channels must be a mapping")
        if set(vocabularies) != set(CHANNELS):
            raise ValueError("checkpoint must contain exactly all tensorizer channels")
        result = {}
        for channel in CHANNELS:
            raw = vocabularies[channel]
            if not isinstance(raw, Mapping):
                raise ValueError(f"vocabulary for {channel} must be a mapping")
            if raw.get("<PAD>") != 0 or raw.get("<UNK>") != 1:
                raise ValueError(f"{channel} must reserve <PAD>=0 and <UNK>=1")
            ids = tuple(raw.values())
            if any(type(v) is not int or v < 0 for v in ids):
                raise ValueError(f"invalid IDs in {channel} vocabulary")
            if len(set(ids)) != len(raw):
                raise ValueError(f"duplicate IDs in {channel} vocabulary")
            if set(ids) != set(range(len(raw))):
                raise ValueError(f"{channel} vocabulary IDs must be contiguous")
            if any(not isinstance(k, str) for k in raw):
                raise ValueError(f"{channel} vocabulary values must be strings")
            for key in raw:
                if key in ("<PAD>", "<UNK>"):
                    continue
                try:
                    decoded = json.loads(key)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{channel}: expected JSON string feature key") from exc
                if not isinstance(decoded, str) or _vocab_key(decoded) != key:
                    raise ValueError(f"{channel}: expected canonical JSON string feature key")
            result[channel] = MappingProxyType(dict(raw))
        return MappingProxyType(result)

    @classmethod
    def fit(cls, records: Sequence[SemanticRecord]) -> SemanticTensorizer:
        if not records:
            raise ValueError("cannot fit on an empty batch")
        values = {channel: set() for channel in CHANNELS}
        for record in records:
            for channel, features in cls._features_for(cls._validated(record)).items():
                values[channel].update(_vocab_key(feature.value) for feature in features)
        vocabs = {}
        for channel in CHANNELS:
            ordered = sorted(values[channel])
            vocabs[channel] = {"<PAD>": 0, "<UNK>": 1}
            vocabs[channel].update({value: i + 2 for i, value in enumerate(ordered)})
        return cls(vocabs)

    @staticmethod
    def _validated(record: SemanticRecord) -> SemanticRecord:
        return SemanticRecord.from_dict(record.to_dict())

    @classmethod
    def _features_for(cls, record: SemanticRecord) -> dict[str, tuple[Feature, ...]]:
        """Extract features from a validated snapshot; preserve candidate scope."""
        excluded = set(record.excluded_layers)
        result: dict[str, tuple[Feature, ...]] = {}
        def prov(layer: str) -> tuple[Provenance, ...]:
            return tuple(record.provenance.get(layer, (Provenance(),)))
        if "surface" not in excluded:
            result["surface"] = (Feature(record.surface, "$.surface", prov("surface")),)
        for layer in ("tokens", "morphemes", "characters"):
            result[layer] = () if layer in excluded else tuple(
                Feature(value, f"$.{layer}[{i}]", prov(layer))
                for i, value in enumerate(getattr(record, layer)))
        result["subcharacters"] = () if "subcharacters" in excluded else tuple(
            Feature(_json([character, component]), f"$.subcharacters[{character!r}][{i}]", prov("subcharacters"))
            for character, components in record.subcharacters.items() for i, component in enumerate(components))
        result["etymology"] = () if "etymology_notes" in excluded else tuple(
            Feature(_json([key, note]), f"$.etymology_notes[{key!r}]", prov("etymology_notes"))
            for key, note in record.etymology_notes.items())
        result["senses"] = () if "senses" in excluded else tuple(
            Feature(sense.sense_id, f"$.senses[{i}].sense_id", sense.provenance["sense"], sense.sense_id)
            for i, sense in enumerate(record.senses))
        result["sememes"] = () if "sememes" in excluded else tuple(
            Feature(value, f"$.senses[{i}].sememes[{j}]", sense.provenance["sememes"], sense.sense_id)
            for i, sense in enumerate(record.senses) for j, value in enumerate(sense.sememes))
        result["concepts"] = () if "concepts" in excluded else tuple(
            Feature(value, f"$.senses[{i}].concepts[{j}]", sense.provenance["concepts"], sense.sense_id)
            for i, sense in enumerate(record.senses) for j, value in enumerate(sense.concepts))
        result["relations"] = () if "relations" in excluded else tuple(
            Feature(_json(list(value)), f"$.relations[{i}]", prov("relations"))
            for i, value in enumerate(record.relations))
        return result

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {channel: len(vocab) for channel, vocab in self._vocabularies.items()}

    @property
    def vocabularies(self) -> dict[str, dict[str, int]]:
        return {channel: dict(vocab) for channel, vocab in self._vocabularies.items()}

    def encode(self, records: Sequence[SemanticRecord]) -> SemanticBatch:
        if torch is None:
            raise RuntimeError("SemanticTensorizer.encode requires torch")
        if not records:
            raise ValueError("cannot encode an empty batch")
        snapshots = [self._validated(record) for record in records]
        all_features = [self._features_for(record) for record in snapshots]
        channels = {}
        for channel in CHANNELS:
            rows = [features[channel] for features in all_features]
            length = max(1, max((len(row) for row in rows), default=0))
            ids = torch.zeros((len(rows), length), dtype=torch.long)
            mask = torch.zeros((len(rows), length), dtype=torch.bool)
            vocab = self._vocabularies[channel]
            for row_index, row in enumerate(rows):
                for col, feature in enumerate(row):
                    ids[row_index, col] = vocab.get(_vocab_key(feature.value), 1)
                    mask[row_index, col] = True
            layer = "etymology_notes" if channel == "etymology" else channel
            layer_provenance = tuple(record.provenance[layer] for record in snapshots)
            channels[channel] = ChannelBatch(ids, mask, tuple(rows), layer_provenance)
        return SemanticBatch(channels)

    def to_dict(self) -> dict[str, Any]:
        return {"version": CHECKPOINT_VERSION, "channels": self.vocabularies}

    @classmethod
    def from_dict(cls, value: Any) -> SemanticTensorizer:
        if not isinstance(value, dict) or set(value) != {"version", "channels"}:
            raise ValueError("invalid tensorizer checkpoint keys")
        if value["version"] != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported tensorizer checkpoint version: {value.get('version')!r}")
        return cls(value["channels"])
