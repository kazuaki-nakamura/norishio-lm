"""Bounded factor-only source-head runner for the frozen Issue #48 design.

No fixture is loaded here. The caller supplies a paired, already-initialized
H1L1 model, prepared rows, and the descriptor-bound schedule. Decoder
parameters remain present but are absent from the loss computation graph.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import time
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from . import benchmark_v3_role_fixture as fixture
from .benchmark_v3_head_runner import evaluate_head_outputs
from .benchmark_v3_model import BenchmarkV3Model, BOS_ID, BYTE_OFFSET, EOS_ID, FACTOR_ORDER, VOCAB_SIZE
from .benchmark_v3_runner import FactorVocabulary, PreparedRow, collate_rows

STEPS = 600
BATCH_SIZE = 16
PARAMETER_COUNT = 32_120


def prepare_role_row(row: Mapping[str, Any], source_ids: Sequence[int],
                     vocabulary: FactorVocabulary) -> PreparedRow:
    """Convert authored source/target into explicit trainer inputs and labels."""
    if set(row["inputs"]) != {"context", "text"}:
        raise ValueError("model input must contain only context and text")
    if row["inputs"]["context"] != "":
        raise ValueError("role fixture context must remain empty")
    frame = row["targets"]["frame"]
    target_text = row["targets"]["text"]
    tokens = tuple(BYTE_OFFSET + byte for byte in target_text.encode("utf-8"))
    source = tuple(source_ids)
    if source != tuple(fixture.source_ids(row)):
        raise ValueError("prepared source tokens differ from authored fixture row")
    decoder = (BOS_ID, *tokens)
    labels = (*tokens, EOS_ID)
    if not source or any(type(token) is not int or not 0 <= token < VOCAB_SIZE
                         for token in (*source, *decoder, *labels)):
        raise ValueError("token outside frozen byte vocabulary")
    return PreparedRow(source, decoder, labels, vocabulary.encode(frame),
                       target_text, {factor: str(frame[factor]) for factor in FACTOR_ORDER})


def train_factor_only(model: BenchmarkV3Model, seed: int,
                      train_rows: Sequence[PreparedRow],
                      schedule: Sequence[Sequence[int]],
                      descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Run exactly the supplied 600 updates; retain diagnostics, not weights."""
    if seed not in (7, 17, 29) or model.arm != "H1L1":
        raise ValueError("seed or arm differs from preregistered Issue #48 design")
    if len(train_rows) != 384 or len(schedule) != STEPS or any(
        len(batch) != BATCH_SIZE or any(type(index) is not int or not 0 <= index < 384
                                       for index in batch) for batch in schedule
    ):
        raise ValueError("train rows or schedule differ from freeze")
    if trainable_parameter_count(model) != PARAMETER_COUNT:
        raise ValueError("H1L1 parameter count differs from freeze")
    training = descriptor["training"]
    if (training["updates_per_run"] != STEPS or training["batch_size"] != BATCH_SIZE
            or training["optimizer"] != {"name": "Adam", "learning_rate": 0.003,
                                            "betas": [0.9, 0.999], "eps": 1e-8,
                                            "weight_decay": 0.0}
            or training["gradient_clip"] != 1.0
            or training["torch_num_threads"] != 1):
        raise ValueError("training hyperparameters differ from freeze")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    initial_sha = state_sha256(model.state_dict())
    losses: list[float] = []
    start = time.perf_counter()
    for indices in schedule:
        batch = collate_rows([train_rows[index] for index in indices])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.source_ids, source_mask=batch.source_mask,
                       factor_targets=batch.factor_targets)
        loss = output["factor_loss"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite factor-only loss")
        loss.backward()
        if any(parameter.grad is not None for parameter in model.decoder.parameters()):
            raise ValueError("factor-only training reached decoder parameters")
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("non-finite saved factor loss")
        losses.append(value)
    wall = time.perf_counter() - start
    evaluated = evaluate_head_outputs(model, train_rows, descriptor=descriptor,
                                      split="train", objective="FACTOR_ONLY", seed=seed)
    return {
        "model": model,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": state_sha256(model.state_dict()),
        "training_wall_seconds": wall,
        "optimizer_updates": len(losses),
        "losses": losses,
        "train_head_records": evaluated["head_records"],
    }


__all__ = ["PARAMETER_COUNT", "STEPS", "prepare_role_row", "train_factor_only"]
