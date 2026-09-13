"""Small, source-only models for the frozen benchmark-v2 tournament.

The module deliberately contains no benchmark row handling.  A caller supplies
already encoded UTF-8 byte ids and, during training, supplies decoder labels and
factor targets as separate tensors.  In particular, targets and metadata never
enter source encoding or factor prediction.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

VOCAB_SIZE = 260
PAD_ID, BOS_ID, EOS_ID, SEP_ID, BYTE_OFFSET = 0, 1, 2, 3, 4
SOURCE_EMBEDDING_DIM = 16
LATENT_DIM = 32
DECODER_EMBEDDING_DIM = 32
HIDDEN_DIM = 32
FACTOR_ORDER = ("participant", "time", "event", "operator")
FACTOR_SIZES = {"participant": 6, "time": 6, "event": 4, "operator": 4}
ARM_IDS = ("A_G0", "A_G1", "B", "C", "D", "E")
FROZEN_DERANGEMENTS = {
    "participant": (2, 3, 4, 5, 0, 1),
    "time": (3, 4, 5, 0, 1, 2),
    "event": (2, 3, 0, 1),
    "operator": (3, 0, 1, 2),
}

FactorLogits: TypeAlias = dict[str, Tensor]
DerangementInput: TypeAlias = Mapping[str, Sequence[int] | Mapping[int, int] | Tensor]


def _seeded(seed: int, constructor: Any) -> nn.Module:
    """Construct a module with a private, named RNG stream."""
    if type(seed) is not int:
        raise TypeError("initialization seeds must be integers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructor()


def _check_ids(ids: Tensor, name: str, *, allow_ignore: bool = False) -> Tensor:
    if not isinstance(ids, Tensor) or ids.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 integer tensor")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                         torch.uint8):
        raise TypeError(f"{name} must contain integer token ids")
    values = ids.long()
    if allow_ignore:
        valid = values != -100
        if bool(valid.any()) and (values[valid] < 0).any():
            raise ValueError(f"{name} contains an invalid ignored-token value")
        if bool(valid.any()) and (values[valid] >= VOCAB_SIZE).any():
            raise ValueError(f"{name} contains an out-of-range token id")
    elif (values < 0).any() or (values >= VOCAB_SIZE).any():
        raise ValueError(f"{name} contains an out-of-range token id")
    return values


def _check_mask(mask: Tensor | None, source_ids: Tensor) -> Tensor:
    if mask is None:
        return source_ids.ne(PAD_ID)
    if not isinstance(mask, Tensor) or mask.shape != source_ids.shape:
        raise ValueError("source_mask must have the same shape as source_ids")
    if mask.dtype is not torch.bool:
        raise TypeError("source_mask must be boolean")
    return mask


def _check_factor_targets(targets: Tensor, batch: int, device: torch.device) -> Tensor:
    if not isinstance(targets, Tensor) or targets.ndim != 2:
        raise ValueError("factor_targets must have shape [batch, 4]")
    if targets.shape != (batch, len(FACTOR_ORDER)):
        raise ValueError("factor_targets must have shape [batch, 4]")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                             torch.uint8):
        raise TypeError("factor_targets must contain integer class ids")
    result = targets.to(device=device, dtype=torch.long)
    for index, field in enumerate(FACTOR_ORDER):
        if (result[:, index] < 0).any() or (result[:, index] >= FACTOR_SIZES[field]).any():
            raise ValueError(f"factor_targets contains an invalid {field} class")
    return result


def _normalize_derangements(mappings: DerangementInput) -> dict[str, Tensor]:
    """Validate explicit zero-based permutations used by arm E.

    A sequence is interpreted as ``source_class -> destination_class``.  A
    mapping object uses the same integer keys and values.  Requiring a complete
    permutation and rejecting fixed points keeps the broken-semantics control
    explicit and prevents an accidental identity control.
    """
    if not isinstance(mappings, Mapping) or set(mappings) != set(FACTOR_ORDER):
        raise ValueError("E requires one derangement for every factor")
    result: dict[str, Tensor] = {}
    for field in FACTOR_ORDER:
        width = FACTOR_SIZES[field]
        raw = mappings[field]
        if isinstance(raw, Mapping):
            if set(raw) == set(range(width)):
                values = [raw[index] for index in range(width)]
            elif len(raw) == width and all(isinstance(key, str) for key in raw):
                # The frozen manifest names factors symbolically.  Preserve
                # its insertion order when callers pass that manifest form.
                labels = list(raw)
                positions = {label: index for index, label in enumerate(labels)}
                if not all(value in positions for value in raw.values()):
                    raise ValueError(f"{field} derangement values must use the same labels")
                values = [positions[raw[label]] for label in labels]
            else:
                raise ValueError(f"{field} derangement keys must be 0..{width - 1}")
        elif isinstance(raw, Tensor):
            values = raw.detach().cpu().tolist()
        else:
            values = list(raw)
        if len(values) != width or any(type(value) is not int for value in values):
            raise ValueError(f"{field} derangement must contain {width} integer values")
        if set(values) != set(range(width)) or any(index == value for index, value in enumerate(values)):
            raise ValueError(f"{field} derangement must be a fixed-point-free permutation")
        result[field] = torch.tensor(values, dtype=torch.long)
    return result


class SourceByteEncoder(nn.Module):
    """260-way byte embedding, masked mean, and 16-to-32 projection."""

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()

        def build() -> nn.Module:
            return nn.ModuleDict({
                "embedding": nn.Embedding(VOCAB_SIZE, SOURCE_EMBEDDING_DIM,
                                           padding_idx=PAD_ID),
                "projection": nn.Linear(SOURCE_EMBEDDING_DIM, LATENT_DIM),
            })

        modules = _seeded(seed, build)
        assert isinstance(modules, nn.ModuleDict)
        self.embedding = modules["embedding"]
        self.projection = modules["projection"]
        self.seed = seed

    def forward(self, source_ids: Tensor, source_mask: Tensor | None = None) -> Tensor:
        ids = _check_ids(source_ids, "source_ids")
        mask = _check_mask(source_mask, ids)
        embedded = self.embedding(ids)
        weights = mask.unsqueeze(-1).to(dtype=embedded.dtype)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (embedded * weights).sum(dim=1) / count
        return self.projection(mean)


class CausalByteDecoder(nn.Module):
    """One-layer 32-dimensional causal byte GRU and byte LM head."""

    def __init__(self, *, seed: int = 1000) -> None:
        super().__init__()

        def build() -> nn.Module:
            return nn.ModuleDict({
                "embedding": nn.Embedding(VOCAB_SIZE, DECODER_EMBEDDING_DIM),
                "gru": nn.GRU(DECODER_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True),
                "lm_head": nn.Linear(HIDDEN_DIM, VOCAB_SIZE),
            })

        modules = _seeded(seed, build)
        assert isinstance(modules, nn.ModuleDict)
        self.embedding = modules["embedding"]
        self.gru = modules["gru"]
        self.lm_head = modules["lm_head"]
        self.seed = seed

    def run(self, input_ids: Tensor, *, h0: Tensor | None = None,
            conditioning: Tensor | None = None) -> tuple[Tensor, Tensor]:
        ids = _check_ids(input_ids, "decoder_input_ids")
        embedded = self.embedding(ids)
        if conditioning is not None:
            if conditioning.ndim == 2 and conditioning.shape == (ids.shape[0], HIDDEN_DIM):
                embedded = embedded + conditioning.unsqueeze(1)
            elif conditioning.ndim == 3 and conditioning.shape == (*ids.shape, HIDDEN_DIM):
                embedded = embedded + conditioning
            else:
                raise ValueError("decoder conditioning must have shape [B, 32] or [B, T, 32]")
        if h0 is None:
            initial = torch.zeros((1, ids.shape[0], HIDDEN_DIM), dtype=embedded.dtype,
                                  device=embedded.device)
        elif h0.shape == (ids.shape[0], HIDDEN_DIM):
            initial = h0.unsqueeze(0)
        elif h0.shape == (1, ids.shape[0], HIDDEN_DIM):
            initial = h0
        else:
            raise ValueError("h0 must have shape [B, 32] or [1, B, 32]")
        hidden, final = self.gru(embedded, initial)
        return self.lm_head(hidden), final

    def forward(self, input_ids: Tensor, *, h0: Tensor | None = None,
                conditioning: Tensor | None = None) -> Tensor:
        return self.run(input_ids, h0=h0, conditioning=conditioning)[0]


class ModelOutput(dict[str, Any]):
    """Typed-by-convention mapping with convenient tensor properties."""

    @property
    def logits(self) -> Tensor | None:
        return self.get("logits")

    @property
    def factor_logits(self) -> FactorLogits:
        return self["factor_logits"]

    @property
    def latent(self) -> Tensor:
        return self["latent"]


class BenchmarkV2Model(nn.Module):
    """One of the frozen Issue #34 source-only tournament arms."""

    def __init__(self, arm: str = "A_G0", *, seed: int = 0,
                 derangement_mappings: DerangementInput | None = None) -> None:
        super().__init__()
        if arm not in ARM_IDS:
            raise ValueError(f"unknown benchmark-v2 arm: {arm}")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        if arm == "E" and derangement_mappings is None:
            raise ValueError("arm E requires explicitly supplied derangement mappings")
        self.arm = arm
        self.seed = seed
        self.source_seed = seed
        self.decoder_seed = seed + 1000
        self.factor_seed = seed + 2000
        self.encoder = SourceByteEncoder(seed=self.source_seed)
        self.decoder = CausalByteDecoder(seed=self.decoder_seed)
        self.derangement_mappings = (_normalize_derangements(derangement_mappings)
                                     if derangement_mappings is not None else None)
        if arm == "E" and any(
            tuple(self.derangement_mappings[field].tolist()) != FROZEN_DERANGEMENTS[field]
            for field in FACTOR_ORDER
        ):
            raise ValueError("arm E mappings differ from the frozen manifest derangements")

        if arm in {"A_G0", "A_G1", "E"}:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.factor_seed)
                self.factor_heads = nn.ModuleDict({
                    field: nn.Linear(LATENT_DIM, FACTOR_SIZES[field])
                    for field in FACTOR_ORDER
                })
                self.factor_projection = nn.Linear(20, HIDDEN_DIM)
        elif arm == "C":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.factor_seed)
                self.factor_heads = nn.ModuleDict({
                    field: nn.Linear(LATENT_DIM, FACTOR_SIZES[field])
                    for field in FACTOR_ORDER
                })
                self.symbol_tables = nn.ModuleDict({
                    field: nn.Embedding(FACTOR_SIZES[field], 8) for field in FACTOR_ORDER
                })
                self.symbol_composer = nn.Linear(32, HIDDEN_DIM)
        elif arm == "D":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.factor_seed)
                self.latent_adapter = nn.Sequential(
                    nn.Linear(LATENT_DIM, 24), nn.Tanh(), nn.Linear(24, HIDDEN_DIM)
                )
        elif arm == "B":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.factor_seed)
                self.factor_heads = nn.ModuleDict({
                    field: nn.Sequential(nn.Linear(LATENT_DIM, 8), nn.Tanh())
                    for field in FACTOR_ORDER
                })
                self.factor_class_heads = nn.ModuleDict({
                    field: nn.Linear(8, FACTOR_SIZES[field]) for field in FACTOR_ORDER
                })
                self.factor_projections = nn.ModuleDict({
                    field: nn.Linear(FACTOR_SIZES[field], HIDDEN_DIM, bias=False)
                    for field in FACTOR_ORDER
                })
                self.factor_projection_bias = nn.Parameter(torch.empty(HIDDEN_DIM))
                nn.init.uniform_(self.factor_projection_bias,
                                 -1.0 / math.sqrt(20), 1.0 / math.sqrt(20))

    @property
    def supports_factor_loss(self) -> bool:
        return self.arm != "D"

    def _factor_logits(self, latent: Tensor) -> FactorLogits:
        if self.arm == "D":
            return {}
        if self.arm == "B":
            return {field: self.factor_class_heads[field](self.factor_heads[field](latent))
                    for field in FACTOR_ORDER}
        return {field: self.factor_heads[field](latent) for field in FACTOR_ORDER}

    def _factor_probabilities(self, factor_logits: FactorLogits) -> dict[str, Tensor]:
        return {field: F.softmax(factor_logits[field], dim=-1) for field in FACTOR_ORDER}

    def _mapped_probabilities(self, probabilities: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.arm != "E":
            return probabilities
        assert self.derangement_mappings is not None
        result: dict[str, Tensor] = {}
        for field in FACTOR_ORDER:
            mapping = self.derangement_mappings[field].to(probabilities[field].device)
            mapped = torch.zeros_like(probabilities[field])
            result[field] = mapped.scatter_add(
                1, mapping.unsqueeze(0).expand(probabilities[field].shape[0], -1),
                probabilities[field]
            )
        return result

    def _conditioning(self, latent: Tensor, factor_logits: FactorLogits,
                      *, detach_factor_path: bool = False) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.arm == "D":
            condition = self.latent_adapter(latent)
            return condition, condition, {}
        probabilities = self._mapped_probabilities(self._factor_probabilities(factor_logits))
        if self.arm == "B":
            raise ValueError("arm B requires prefix-local gates")
        if self.arm == "C":
            symbols = torch.stack([factor_logits[field].argmax(dim=-1) for field in FACTOR_ORDER], dim=1)
            condition = self.symbol_composer(torch.cat([
                self.symbol_tables[field](symbols[:, index])
                for index, field in enumerate(FACTOR_ORDER)
            ], dim=-1))
        else:
            flat = torch.cat([probabilities[field] for field in FACTOR_ORDER], dim=-1)
            if detach_factor_path:
                flat = flat.detach()
            condition = self.factor_projection(flat)
        return condition, condition, probabilities

    def _local_conditioning(self, factor_logits: FactorLogits, local_gates: Tensor,
                            steps: int) -> tuple[Tensor, dict[str, Tensor]]:
        if self.arm != "B":
            raise ValueError("local factor gates are specific to arm B")
        if (not isinstance(local_gates, Tensor) or local_gates.ndim != 3 or
                local_gates.shape[1:] != (steps, len(FACTOR_ORDER))):
            raise ValueError("local_gates must have shape [B, T, 4]")
        if local_gates.dtype is not torch.bool:
            raise TypeError("local_gates must be boolean")
        probabilities = self._factor_probabilities(factor_logits)
        batch = next(iter(probabilities.values())).shape[0]
        if local_gates.shape[0] != batch:
            raise ValueError("local_gates batch differs from source batch")
        gates = local_gates.to(device=next(iter(probabilities.values())).device)
        condition = self.factor_projection_bias.view(1, 1, -1).expand(batch, steps, -1)
        for index, field in enumerate(FACTOR_ORDER):
            projected = self.factor_projections[field](probabilities[field]).unsqueeze(1)
            condition = condition + projected * gates[:, :, index:index + 1].to(projected.dtype)
        return condition, probabilities

    def encode_source(self, source_ids: Tensor, source_mask: Tensor | None = None) -> Tensor:
        """Encode only source bytes; metadata and gold fields are impossible here."""
        return self.encoder(source_ids, source_mask)

    def source_predictions(self, source_ids: Tensor, *,
                           source_mask: Tensor | None = None) -> ModelOutput:
        latent = self.encode_source(source_ids, source_mask)
        return ModelOutput(latent=latent, factor_logits=self._factor_logits(latent))

    def source_only_intermediate_predictions(self, source_ids: Tensor, *,
                                             source_mask: Tensor | None = None) -> ModelOutput:
        return self.source_predictions(source_ids, source_mask=source_mask)

    def predict_factors(self, source_ids: Tensor, *,
                        source_mask: Tensor | None = None) -> FactorLogits:
        """Return source-only factor logits without accepting gold metadata."""
        return self.source_predictions(source_ids, source_mask=source_mask).factor_logits

    def source_factor_probabilities(self, source_ids: Tensor, *,
                                    source_mask: Tensor | None = None) -> dict[str, Tensor]:
        predictions = self.source_predictions(source_ids, source_mask=source_mask)
        return self._factor_probabilities(predictions.factor_logits)

    def factor_loss(self, factor_logits: FactorLogits | Tensor,
                    factor_targets: Tensor) -> Tensor:
        if not self.supports_factor_loss:
            raise ValueError("arm D has no factor heads or factor loss")
        if isinstance(factor_logits, Tensor):
            factor_logits = self.source_predictions(factor_logits).factor_logits
        if not isinstance(factor_logits, Mapping) or set(factor_logits) != set(FACTOR_ORDER):
            raise ValueError("factor_logits must contain participant, time, event, and operator")
        first = factor_logits[FACTOR_ORDER[0]]
        targets = _check_factor_targets(factor_targets, first.shape[0], first.device)
        if self.arm == "E":
            assert self.derangement_mappings is not None
        losses = []
        for index, field in enumerate(FACTOR_ORDER):
            target = targets[:, index]
            if self.arm == "E":
                target = self.derangement_mappings[field].to(target.device)[target]
            logits = factor_logits[field]
            if logits.shape != (targets.shape[0], FACTOR_SIZES[field]):
                raise ValueError(f"{field} logits have the wrong shape")
            losses.append(F.cross_entropy(logits, target))
        return sum(losses)

    def forward(self, source_ids: Tensor, decoder_input_ids: Tensor | None = None, *,
                source_mask: Tensor | None = None, labels: Tensor | None = None,
                factor_targets: Tensor | None = None,
                local_gates: Tensor | None = None) -> ModelOutput:
        latent = self.encode_source(source_ids, source_mask)
        factor_logits = self._factor_logits(latent)
        output = ModelOutput(latent=latent, factor_logits=factor_logits)
        if self.arm == "B":
            if decoder_input_ids is None:
                if local_gates is not None:
                    raise ValueError("local_gates require decoder_input_ids")
                h0, condition = None, None
                probabilities = self._factor_probabilities(factor_logits)
            else:
                if local_gates is None:
                    raise ValueError("arm B decoder requires prefix-local gates")
                h0 = None
                condition, probabilities = self._local_conditioning(
                    factor_logits, local_gates, decoder_input_ids.shape[1]
                )
        elif self.arm == "D":
            h0, condition, probabilities = self._conditioning(latent, factor_logits)
        else:
            h0, condition, probabilities = self._conditioning(
                latent, factor_logits, detach_factor_path=self.arm == "A_G1"
            )
        output["factor_probs"] = probabilities
        if self.arm == "C":
            output["symbol_indices"] = torch.stack([
                factor_logits[field].argmax(dim=-1) for field in FACTOR_ORDER
            ], dim=1)
        if decoder_input_ids is not None:
            logits, _ = self.decoder.run(decoder_input_ids, h0=h0, conditioning=condition)
            output["logits"] = logits
            if labels is not None:
                checked_labels = _check_ids(labels, "labels", allow_ignore=True).to(logits.device)
                if checked_labels.shape != logits.shape[:2]:
                    raise ValueError("labels must have the same [B, T] shape as decoder_input_ids")
                output["lm_loss"] = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), checked_labels.reshape(-1), ignore_index=-100
                )
        elif labels is not None:
            raise ValueError("labels require decoder_input_ids")
        if factor_targets is not None:
            output["factor_loss"] = self.factor_loss(factor_logits, factor_targets)
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "schema": "norishio.issue34.model.v1",
            "arm": self.arm,
            "seed": self.seed,
            "source_seed": self.source_seed,
            "decoder_seed": self.decoder_seed,
            "factor_seed": self.factor_seed,
            "derangements": (
                {field: list(FROZEN_DERANGEMENTS[field]) for field in FACTOR_ORDER}
                if self.arm == "E" else None
            ),
        }


class A_G0Model(BenchmarkV2Model):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__("A_G0", seed=seed)


class A_G1Model(BenchmarkV2Model):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__("A_G1", seed=seed)


class CModel(BenchmarkV2Model):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__("C", seed=seed)


class DModel(BenchmarkV2Model):
    def __init__(self, *, seed: int = 0) -> None:
        super().__init__("D", seed=seed)


class EModel(BenchmarkV2Model):
    def __init__(self, *, seed: int = 0,
                 derangement_mappings: DerangementInput) -> None:
        super().__init__("E", seed=seed, derangement_mappings=derangement_mappings)


def build_model(arm: str, *, seed: int = 0,
                derangement_mappings: DerangementInput | None = None) -> BenchmarkV2Model:
    return BenchmarkV2Model(arm, seed=seed, derangement_mappings=derangement_mappings)


def count_trainable_parameters(model: nn.Module) -> int:
    """Count exactly the tensors included by the frozen tournament budget."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


# Short aliases keep the public surface usable by small benchmark runners while
# retaining descriptive class names in tracebacks and documentation.
ByteSourceEncoder = SourceByteEncoder
ByteDecoder = CausalByteDecoder
build_benchmark_v2_model = build_model


__all__ = [
    "ARM_IDS", "A_G0Model", "A_G1Model", "BenchmarkV2Model", "BYTE_OFFSET",
    "BOS_ID", "ByteDecoder", "ByteSourceEncoder", "CModel", "CausalByteDecoder", "DModel", "DerangementInput",
    "EModel", "EOS_ID", "FACTOR_ORDER", "FACTOR_SIZES", "FROZEN_DERANGEMENTS", "ModelOutput",
    "PAD_ID", "SEP_ID", "SourceByteEncoder", "VOCAB_SIZE", "build_benchmark_v2_model",
    "build_model", "count_trainable_parameters",
]
