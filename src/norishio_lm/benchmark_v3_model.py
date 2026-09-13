"""CPU-small shared model for the benchmark-v3 Phase 1 arms.

The four ``H*L*`` arms share every parameterized component and differ only in
the factor-head activation (H0: linear, H1: tanh) and whether per-step factor
gates are applied (L0: no gates, L1: caller-supplied causal gates).  D_AUX
retains the same four heads and CE loss while conditioning its decoder only on
the source latent.  NO_INPUT replaces every source row with the same BOS/SEP
tensor before encoding.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, TypeAlias

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .benchmark_v2_model import (
    BOS_ID,
    BYTE_OFFSET,
    CausalByteDecoder,
    EOS_ID,
    LATENT_DIM,
    PAD_ID,
    SEP_ID,
    SourceByteEncoder,
    VOCAB_SIZE,
)

FACTOR_ORDER = ("participant", "time", "event", "operator")
FACTOR_SIZES = {"participant": 6, "time": 6, "event": 4, "operator": 4}
ARM_IDS = ("H0L0", "H0L1", "H1L0", "H1L1", "D_AUX", "NO_INPUT")
MAIN_ARMS = ("H0L0", "H0L1", "H1L0", "H1L1")
FACTOR_HIDDEN_DIM = 16
HIDDEN_DIM = 32
FACTOR_METADATA = {
    "code_space": "canonical_factor_class_indices",
    "canonical_space": "canonical_semantic_space",
    "factor_order": list(FACTOR_ORDER),
    "probability_semantics": "softmax over canonical factor logits",
}

FactorLogits: TypeAlias = dict[str, Tensor]
FactorProbabilities: TypeAlias = dict[str, Tensor]


def _seeded(seed: int, constructor: Any) -> nn.Module:
    if type(seed) is not int:
        raise TypeError("initialization seeds must be integers")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return constructor()


def _check_ids(ids: Tensor, name: str, *, allow_ignore: bool = False) -> Tensor:
    if not isinstance(ids, Tensor) or ids.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 integer tensor")
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} must contain integer token ids")
    result = ids.long()
    valid = result != -100 if allow_ignore else torch.ones_like(result, dtype=torch.bool)
    if bool(valid.any()) and ((result[valid] < 0).any() or (result[valid] >= VOCAB_SIZE).any()):
        raise ValueError(f"{name} contains an out-of-range token id")
    return result


def _source_mask(mask: Tensor | None, source_ids: Tensor) -> Tensor:
    if mask is None:
        return source_ids.ne(PAD_ID)
    if not isinstance(mask, Tensor) or mask.shape != source_ids.shape:
        raise ValueError("source_mask must have the same shape as source_ids")
    if mask.dtype is not torch.bool:
        raise TypeError("source_mask must be boolean")
    return mask


def _factor_targets(targets: Tensor, batch: int, device: torch.device) -> Tensor:
    if not isinstance(targets, Tensor) or targets.shape != (batch, len(FACTOR_ORDER)):
        raise ValueError("factor_targets must have shape [batch, 4]")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("factor_targets must contain integer class ids")
    result = targets.to(device=device, dtype=torch.long)
    for column, field in enumerate(FACTOR_ORDER):
        if bool((result[:, column] < 0).any()) or bool((result[:, column] >= FACTOR_SIZES[field]).any()):
            raise ValueError(f"factor_targets contains an invalid {field} class")
    return result


def _check_gates(gates: Tensor, batch: int, steps: int, device: torch.device) -> Tensor:
    if not isinstance(gates, Tensor) or gates.shape != (batch, steps, len(FACTOR_ORDER)):
        raise ValueError("local_gates must have shape [batch, decoder_steps, 4]")
    if gates.dtype is not torch.bool:
        raise TypeError("local_gates must be boolean")
    return gates.to(device=device)


def _check_probability_map(probabilities: Mapping[str, Tensor], batch: int | None = None) -> FactorProbabilities:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(FACTOR_ORDER):
        raise ValueError("factor probabilities must contain exactly the four canonical factors")
    checked: FactorProbabilities = {}
    for field in FACTOR_ORDER:
        value = probabilities[field]
        if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != FACTOR_SIZES[field]:
            raise ValueError(f"{field} probabilities must have shape [batch, {FACTOR_SIZES[field]}]")
        if batch is not None and value.shape[0] != batch:
            raise ValueError("factor probability batches must match")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{field} probabilities must be finite floating-point values")
        if bool((value < 0).any()) or not bool(torch.allclose(value.sum(-1), torch.ones(value.shape[0], device=value.device, dtype=value.dtype), atol=1e-6, rtol=1e-6)):
            raise ValueError(f"{field} probabilities must be nonnegative rows summing to one")
        checked[field] = value
    return checked


def _validate_one_hot(one_hot: Tensor, field: str, batch: int, device: torch.device) -> Tensor:
    if not isinstance(one_hot, Tensor) or one_hot.ndim not in (1, 2):
        raise ValueError("artificial one-hot must have shape [classes] or [batch, classes]")
    width = FACTOR_SIZES[field]
    if one_hot.ndim == 1:
        one_hot = one_hot.unsqueeze(0).expand(batch, -1)
    if one_hot.shape != (batch, width):
        raise ValueError(f"artificial one-hot for {field} must have width {width}")
    value = one_hot.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool((value > 1).any()):
        raise ValueError("artificial one-hot must be finite and in [0, 1]")
    if not bool(torch.all(value.sum(-1) == 1)):
        raise ValueError("artificial one-hot rows must sum to one")
    if not bool(torch.all((value == 0) | (value == 1))) or not bool((value == 1).sum(-1).eq(1).all()):
        raise ValueError("artificial one-hot must contain exactly one one per row")
    return value


class _FactorHead(nn.Module):
    def __init__(self, width: int, *, nonlinear: bool) -> None:
        super().__init__()
        self.hidden = nn.Linear(LATENT_DIM, FACTOR_HIDDEN_DIM)
        self.activation = nn.Tanh() if nonlinear else nn.Identity()
        self.classifier = nn.Linear(FACTOR_HIDDEN_DIM, width)

    def forward(self, latent: Tensor) -> Tensor:
        return self.classifier(self.activation(self.hidden(latent)))


class ModelOutput(dict[str, Any]):
    @property
    def logits(self) -> Tensor | None:
        return self.get("logits")

    @property
    def factor_logits(self) -> FactorLogits:
        return self["factor_logits"]

    @property
    def factor_probs(self) -> FactorProbabilities:
        return self["factor_probs"]

    @property
    def latent(self) -> Tensor:
        return self["latent"]

    @property
    def source_ids(self) -> Tensor:
        return self["source_ids"]


class BenchmarkV3Model(nn.Module):
    def __init__(self, arm: str = "H0L0", *, seed: int = 0) -> None:
        super().__init__()
        if arm not in ARM_IDS:
            raise ValueError(f"unknown benchmark-v3 arm: {arm}")
        if type(seed) is not int:
            raise TypeError("seed must be an integer")
        self.arm, self.seed = arm, seed
        self.source_seed, self.decoder_seed = seed, seed + 1000
        self.h0_seed, self.factor_seed, self.projection_seed = seed + 1500, seed + 2000, seed + 2500
        self.encoder = SourceByteEncoder(seed=self.source_seed)
        self.decoder = CausalByteDecoder(seed=self.decoder_seed)
        self.h0 = _seeded(self.h0_seed, lambda: nn.Linear(LATENT_DIM, HIDDEN_DIM))
        nonlinear = arm.startswith("H1")
        self.factor_heads = _seeded(
            self.factor_seed,
            lambda: nn.ModuleDict({field: _FactorHead(FACTOR_SIZES[field], nonlinear=nonlinear)
                                   for field in FACTOR_ORDER}),
        )
        if not isinstance(self.factor_heads, nn.ModuleDict):
            raise AssertionError("factor head construction failed")
        if arm in MAIN_ARMS or arm == "NO_INPUT":
            self.factor_projections = _seeded(
                self.projection_seed,
                lambda: nn.ModuleDict({field: nn.Linear(FACTOR_SIZES[field], HIDDEN_DIM, bias=False)
                                       for field in FACTOR_ORDER}),
            )
            self.factor_bias = nn.Parameter(torch.zeros(HIDDEN_DIM))
        self.arm_metadata = {
            "arm": arm, "head_activation": "tanh" if nonlinear else "linear",
            "gate_mode": "causal_per_step" if arm.endswith("L1") else "disabled",
            "decoder_factor_conditioning": arm != "D_AUX",
            "source_mode": "fixed_no_input" if arm == "NO_INPUT" else "source_bytes",
            **FACTOR_METADATA,
        }

    @property
    def supports_factor_loss(self) -> bool:
        return True

    @property
    def uses_local_gates(self) -> bool:
        return self.arm.endswith("L1")

    def _source(self, source_ids: Tensor, source_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        ids = _check_ids(source_ids, "source_ids")
        if self.arm == "NO_INPUT":
            ids = torch.tensor([[BOS_ID, SEP_ID]], dtype=torch.long, device=ids.device).expand(ids.shape[0], -1).clone()
            mask = torch.ones_like(ids, dtype=torch.bool)
        else:
            mask = _source_mask(source_mask, ids)
        return ids, mask

    def encode_source(self, source_ids: Tensor, source_mask: Tensor | None = None) -> Tensor:
        ids, mask = self._source(source_ids, source_mask)
        return self.encoder(ids, mask)

    def _factor_logits(self, latent: Tensor) -> FactorLogits:
        return {field: self.factor_heads[field](latent) for field in FACTOR_ORDER}

    @staticmethod
    def canonical_factor_probabilities(factor_logits: Mapping[str, Tensor]) -> FactorProbabilities:
        if not isinstance(factor_logits, Mapping) or set(factor_logits) != set(FACTOR_ORDER):
            raise ValueError("factor_logits must contain exactly the four canonical factors")
        result: FactorProbabilities = {}
        for field in FACTOR_ORDER:
            logits = factor_logits[field]
            if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[1] != FACTOR_SIZES[field]:
                raise ValueError(f"{field} logits have the wrong shape")
            result[field] = F.softmax(logits, dim=-1)
        return result

    def source_factor_probabilities(self, source_ids: Tensor, *, source_mask: Tensor | None = None) -> FactorProbabilities:
        return self.canonical_factor_probabilities(self._factor_logits(self.encode_source(source_ids, source_mask)))

    def factor_loss(self, factor_logits: Mapping[str, Tensor], factor_targets: Tensor) -> Tensor:
        if not isinstance(factor_logits, Mapping) or set(factor_logits) != set(FACTOR_ORDER):
            raise ValueError("factor_logits must contain exactly the four canonical factors")
        first = factor_logits[FACTOR_ORDER[0]]
        targets = _factor_targets(factor_targets, first.shape[0], first.device)
        return sum(F.cross_entropy(factor_logits[field], targets[:, index])
                   for index, field in enumerate(FACTOR_ORDER))

    def _factor_condition(self, probabilities: FactorProbabilities, *, steps: int,
                          local_gates: Tensor | None, batch: int, device: torch.device) -> Tensor:
        probs = _check_probability_map(probabilities, batch)
        condition = self.factor_bias.view(1, 1, -1).expand(batch, steps, -1)
        for index, field in enumerate(FACTOR_ORDER):
            contribution = self.factor_projections[field](probs[field]).unsqueeze(1)
            if local_gates is not None:
                contribution = contribution * local_gates[:, :, index:index + 1].to(contribution.dtype)
            condition = condition + contribution
        return condition

    def intervene_probabilities(self, probabilities: Mapping[str, Tensor], field: str,
                                artificial_one_hot: Tensor) -> FactorProbabilities:
        if field not in FACTOR_ORDER:
            raise ValueError(f"unknown factor {field!r}")
        checked = _check_probability_map(probabilities)
        one_hot = _validate_one_hot(artificial_one_hot, field, checked[field].shape[0], checked[field].device)
        result = {name: value.clone() for name, value in checked.items()}
        result[field] = one_hot.to(dtype=checked[field].dtype)
        return result

    def intervene_factor_probabilities(self, probabilities: Mapping[str, Tensor], field: str,
                                       artificial_one_hot: Tensor) -> FactorProbabilities:
        return self.intervene_probabilities(probabilities, field, artificial_one_hot)

    def forward_with_intervention(self, source_ids: Tensor, decoder_input_ids: Tensor,
                                  field: str, artificial_one_hot: Tensor, *,
                                  source_mask: Tensor | None = None, local_gates: Tensor | None = None) -> ModelOutput:
        output = self(source_ids, decoder_input_ids, source_mask=source_mask, local_gates=local_gates)
        probabilities = self.intervene_probabilities(output["factor_probs"], field, artificial_one_hot)
        output["intervened_factor_probs"] = probabilities
        condition = None if self.arm == "D_AUX" else self._factor_condition(
            probabilities, steps=decoder_input_ids.shape[1], local_gates=(
                _check_gates(local_gates, decoder_input_ids.shape[0], decoder_input_ids.shape[1], decoder_input_ids.device)
                if self.uses_local_gates else None), batch=decoder_input_ids.shape[0], device=decoder_input_ids.device)
        h0 = self.h0(output["latent"])
        output["logits"] = self.decoder(decoder_input_ids, h0=h0, conditioning=condition)
        return output

    def forward(self, source_ids: Tensor, decoder_input_ids: Tensor | None = None, *,
                source_mask: Tensor | None = None, labels: Tensor | None = None,
                factor_targets: Tensor | None = None, local_gates: Tensor | None = None) -> ModelOutput:
        raw_source, _ = self._source(source_ids, source_mask)
        latent = self.encoder(raw_source, torch.ones_like(raw_source, dtype=torch.bool)) if self.arm == "NO_INPUT" else self.encode_source(source_ids, source_mask)
        factor_logits = self._factor_logits(latent)
        probabilities = self.canonical_factor_probabilities(factor_logits)
        output = ModelOutput(latent=latent, factor_logits=factor_logits, factor_probs=probabilities,
                             canonical_factor_probabilities=probabilities,
                             factor_metadata=dict(self.arm_metadata), source_ids=raw_source)
        if decoder_input_ids is not None:
            decoder_ids = _check_ids(decoder_input_ids, "decoder_input_ids")
            if decoder_ids.shape[0] != raw_source.shape[0]:
                raise ValueError("decoder and source batches must match")
            gates = None
            if self.uses_local_gates:
                if local_gates is None:
                    raise ValueError("L1 arms require caller-supplied local_gates")
                gates = _check_gates(local_gates, decoder_ids.shape[0], decoder_ids.shape[1], decoder_ids.device)
            condition = None
            if self.arm in MAIN_ARMS or self.arm == "NO_INPUT":
                condition = self._factor_condition(probabilities, steps=decoder_ids.shape[1], local_gates=gates,
                                                   batch=decoder_ids.shape[0], device=decoder_ids.device)
            logits = self.decoder(decoder_ids, h0=self.h0(latent), conditioning=condition)
            output["logits"] = logits
            if labels is not None:
                checked = _check_ids(labels, "labels", allow_ignore=True).to(logits.device)
                if checked.shape != logits.shape[:2]:
                    raise ValueError("labels must have the same [batch, steps] shape as decoder_input_ids")
                output["lm_loss"] = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), checked.reshape(-1), ignore_index=-100)
        elif labels is not None:
            raise ValueError("labels require decoder_input_ids")
        if factor_targets is not None:
            output["factor_loss"] = self.factor_loss(factor_logits, factor_targets)
        losses = [output[key] for key in ("lm_loss", "factor_loss") if key in output]
        if losses:
            output["loss"] = sum(losses)
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_count(self) -> int:
        """Return the authoritative count of trainable parameters."""
        return self.parameter_count()

def build_model(arm: str, *, seed: int = 0) -> BenchmarkV3Model:
    return BenchmarkV3Model(arm, seed=seed)


__all__ = ["ARM_IDS", "MAIN_ARMS", "FACTOR_ORDER", "FACTOR_SIZES", "FACTOR_METADATA",
           "BOS_ID", "BYTE_OFFSET", "EOS_ID", "PAD_ID", "SEP_ID", "VOCAB_SIZE",
           "BenchmarkV3Model", "ModelOutput", "build_model"]
