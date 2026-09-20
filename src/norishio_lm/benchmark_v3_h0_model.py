"""Future-only H0-path variants for the factor-path confirmation experiment.

The historical :mod:`benchmark_v3_model` arms and their checkpoint contract
remain unchanged.  This module exposes two future-only external arm IDs while
keeping the exact H1L1 parameterized module tree.  ``H1L1_ANCHOR`` changes only
the latent used to initialise the decoder; factor heads and factor projections
still consume the row's source latent.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .benchmark_v3_model import (
    BOS_ID,
    SEP_ID,
    FACTOR_ORDER,
    FACTOR_SIZES,
    BenchmarkV3Model,
    ModelOutput,
    _check_gates,
    _check_ids,
    _check_probability_map,
)


H0_ARM_IDS = ("H1L1", "H1L1_ANCHOR")
ANCHOR_ARM = "H1L1_ANCHOR"
PARAMETER_COUNT = 32_120
FactorProbabilityMap = dict[str, Tensor]


class FutureH0PathModel(BenchmarkV3Model):
    """H1L1 and parameter-matched source-independent-h0 future arms.

    ``arm`` and ``arm_id`` are the external future-only arm identities;
    ``internal_arm`` records the inherited H1L1 implementation.  No module,
    parameter, buffer, or state-dict key is added for the anchor mode.
    """

    arm_id: str
    internal_arm: str
    h0_mode: str

    def __init__(self, arm: str = "H1L1", *, seed: int = 0) -> None:
        if arm not in H0_ARM_IDS:
            raise ValueError(f"unknown future H0-path arm: {arm}")
        super().__init__("H1L1", seed=seed)
        self.internal_arm = "H1L1"
        self.arm = arm
        self.arm_id = arm
        self.h0_mode = "anchor_bos_sep" if arm == ANCHOR_ARM else "source_latent"
        if arm == ANCHOR_ARM:
            self.arm_metadata = {
                **self.arm_metadata,
                "arm": arm,
                "internal_arm": self.internal_arm,
                "external_arm": arm,
                "h0_mode": self.h0_mode,
            }
        if self.parameter_count() != PARAMETER_COUNT:
            raise AssertionError("future H0-path parameter count changed")

    @property
    def uses_local_gates(self) -> bool:
        """Both future arms use the inherited causal L1 gates."""

        return True

    def _anchor_latent(self, batch: int, device: torch.device) -> Tensor:
        anchor_ids = torch.tensor(
            [[BOS_ID, SEP_ID]], dtype=torch.long, device=device,
        ).expand(batch, -1)
        anchor_mask = torch.ones_like(anchor_ids, dtype=torch.bool)
        return self.encoder(anchor_ids, anchor_mask)

    def _decoder_h0(self, source_latent: Tensor) -> Tensor:
        if self.arm_id == ANCHOR_ARM:
            source_latent = self._anchor_latent(
                source_latent.shape[0], source_latent.device,
            )
        return self.h0(source_latent)

    def decoder_initial_state(
        self, source_ids: Tensor, *, source_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the decoder h0 state used by the selected future arm."""

        latent = self.encode_source(source_ids, source_mask)
        return self._decoder_h0(latent)

    def _decoder_condition(
        self, probabilities: Mapping[str, Tensor], decoder_ids: Tensor,
        local_gates: Tensor | None,
    ) -> Tensor:
        gates = None
        if self.uses_local_gates:
            if local_gates is None:
                raise ValueError("H1L1 future arms require caller-supplied local_gates")
            gates = _check_gates(
                local_gates, decoder_ids.shape[0], decoder_ids.shape[1],
                decoder_ids.device,
            )
        return self._factor_condition(
            probabilities, steps=decoder_ids.shape[1], local_gates=gates,
            batch=decoder_ids.shape[0], device=decoder_ids.device,
        )

    def forward(
        self, source_ids: Tensor, decoder_input_ids: Tensor | None = None, *,
        source_mask: Tensor | None = None, labels: Tensor | None = None,
        factor_targets: Tensor | None = None,
        local_gates: Tensor | None = None,
    ) -> ModelOutput:
        """Run the shared H1L1 path with an optional anchor-only decoder h0."""

        raw_source, _ = self._source(source_ids, source_mask)
        latent = self.encode_source(source_ids, source_mask)
        factor_logits = self._factor_logits(latent)
        probabilities = self.canonical_factor_probabilities(factor_logits)
        output = ModelOutput(
            latent=latent,
            factor_logits=factor_logits,
            factor_probs=probabilities,
            canonical_factor_probabilities=probabilities,
            factor_metadata=dict(self.arm_metadata),
            source_ids=raw_source,
        )

        if decoder_input_ids is not None:
            decoder_ids = _check_ids(decoder_input_ids, "decoder_input_ids")
            if decoder_ids.shape[0] != raw_source.shape[0]:
                raise ValueError("decoder and source batches must match")
            condition = self._decoder_condition(probabilities, decoder_ids, local_gates)
            logits = self.decoder(
                decoder_ids,
                h0=self._decoder_h0(latent),
                conditioning=condition,
            )
            output["logits"] = logits
            if labels is not None:
                checked = _check_ids(labels, "labels", allow_ignore=True).to(logits.device)
                if checked.shape != logits.shape[:2]:
                    raise ValueError(
                        "labels must have the same [batch, steps] shape as decoder_input_ids"
                    )
                output["lm_loss"] = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    checked.reshape(-1),
                    ignore_index=-100,
                )
        elif labels is not None:
            raise ValueError("labels require decoder_input_ids")

        if factor_targets is not None:
            output["factor_loss"] = self.factor_loss(factor_logits, factor_targets)
        losses = [output[key] for key in ("lm_loss", "factor_loss") if key in output]
        if losses:
            output["loss"] = sum(losses)
        return output

    def forward_with_intervention(
        self, source_ids: Tensor, decoder_input_ids: Tensor, field: str,
        artificial_one_hot: Tensor, *, source_mask: Tensor | None = None,
        local_gates: Tensor | None = None,
    ) -> ModelOutput:
        """Apply one factor intervention while preserving the selected h0 path."""

        output = self(
            source_ids, decoder_input_ids, source_mask=source_mask,
            local_gates=local_gates,
        )
        probabilities = self.intervene_probabilities(
            output["factor_probs"], field, artificial_one_hot,
        )
        output["intervened_factor_probs"] = probabilities
        condition = self._decoder_condition(
            probabilities, decoder_input_ids, local_gates,
        )
        output["logits"] = self.decoder(
            decoder_input_ids,
            h0=self._decoder_h0(output["latent"]),
            conditioning=condition,
        )
        return output

    def replace_factor_probability(
        self,
        probabilities: Mapping[str, Tensor],
        field: str,
        replacement: Tensor,
    ) -> FactorProbabilityMap:
        """Replace exactly one factor row or batch with a probability simplex."""

        if field not in FACTOR_ORDER:
            raise ValueError(f"unknown factor {field!r}")
        checked = _check_probability_map(probabilities)
        batch_sizes = {value.shape[0] for value in checked.values()}
        if len(batch_sizes) != 1:
            raise ValueError("factor probability batches must match")
        batch = checked[field].shape[0]
        width = FACTOR_SIZES[field]
        if not isinstance(replacement, Tensor) or replacement.ndim not in (1, 2):
            raise ValueError(
                "probability replacement must have shape [classes] or [batch, classes]"
            )
        if not replacement.is_floating_point():
            raise TypeError("probability replacement must be floating-point")
        if replacement.shape[-1] != width:
            raise ValueError(
                f"probability replacement for {field} must have width {width}"
            )
        if replacement.ndim == 2 and replacement.shape[0] != batch:
            raise ValueError("probability replacement batch does not match the map")
        value = replacement.to(
            device=checked[field].device, dtype=checked[field].dtype,
        )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("probability replacement must be finite")
        if bool((value < 0).any()):
            raise ValueError("probability replacement must be nonnegative")
        rows = value.shape[0] if value.ndim == 2 else 1
        sums = value if value.ndim == 2 else value.unsqueeze(0)
        if not bool(torch.allclose(
            sums.sum(-1), torch.ones(rows, device=value.device, dtype=value.dtype),
            atol=1e-6, rtol=1e-6,
        )):
            raise ValueError("probability replacement rows must sum to one")
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(batch, -1).clone()
        else:
            value = value.clone()
        result = {name: tensor.clone() for name, tensor in checked.items()}
        result[field] = value
        return result

    def forward_with_probability_replacement(
        self,
        source_ids: Tensor,
        decoder_input_ids: Tensor,
        field: str,
        replacement: Tensor,
        *,
        source_mask: Tensor | None = None,
        local_gates: Tensor | None = None,
    ) -> ModelOutput:
        """Forward with one factor probability vector replaced by a simplex."""

        output = self(
            source_ids, decoder_input_ids, source_mask=source_mask,
            local_gates=local_gates,
        )
        probabilities = self.replace_factor_probability(
            output["factor_probs"], field, replacement,
        )
        output["replaced_factor_probs"] = probabilities
        condition = self._decoder_condition(
            probabilities, decoder_input_ids, local_gates,
        )
        output["logits"] = self.decoder(
            decoder_input_ids,
            h0=self._decoder_h0(output["latent"]),
            conditioning=condition,
        )
        return output


def build_future_h0_model(arm: str, *, seed: int = 0) -> FutureH0PathModel:
    """Build one of the two future-only H0-path arms."""

    return FutureH0PathModel(arm, seed=seed)


__all__ = [
    "ANCHOR_ARM",
    "FactorProbabilityMap",
    "H0_ARM_IDS",
    "PARAMETER_COUNT",
    "FutureH0PathModel",
    "build_future_h0_model",
]
