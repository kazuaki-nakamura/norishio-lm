"""Reusable H1L1 head-objective training for the Issue #46 experiment.

Callers provide prepared rows, a frozen schedule, and its descriptor.  No
fixture is loaded here.  JOINT trains with LM plus factor loss; FACTOR_ONLY
trains with factor loss alone.  Evaluation retains only factor-head outputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from .benchmark_v3_fsm import FrozenLocalPrefixFSM
from .benchmark_v3_model import FACTOR_ORDER, BenchmarkV3Model, build_model
from .benchmark_v3_runner import (
    ADAM_BETAS, ADAM_EPS, BATCH_SIZE, GRADIENT_CLIP, LEARNING_RATE,
    PreparedRow, collate_rows, teacher_forced_gates,
)

HEAD_RUNNER_SCHEMA = "norishio.issue46.head-runner.v1"
OBJECTIVES = ("JOINT", "FACTOR_ONLY")
Objective: TypeAlias = Literal["JOINT", "FACTOR_ONLY"]
SEEDS = (7, 17, 29)
TRAIN_STEPS = 600
PARAMETER_COUNT = 32_120


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def descriptor_sha256(descriptor: Mapping[str, Any] | None) -> str:
    value = {} if descriptor is None else descriptor
    if not isinstance(value, Mapping):
        raise TypeError("descriptor must be a mapping")
    try:
        return hashlib.sha256(_canonical_bytes(dict(value))).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError("descriptor must be JSON-serializable") from exc


def _validate_schedule(schedule: Sequence[Sequence[int]], *, row_count: int | None) -> list[list[int]]:
    if (not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes))
            or not schedule or len(schedule) > TRAIN_STEPS):
        raise ValueError("schedule must contain between 1 and 600 steps")
    normalized: list[list[int]] = []
    for batch in schedule:
        if (not isinstance(batch, Sequence) or isinstance(batch, (str, bytes))
                or len(batch) != BATCH_SIZE):
            raise ValueError("every schedule batch must contain 16 row indices")
        values = list(batch)
        if any(type(index) is not int or index < 0 or
               (row_count is not None and index >= row_count) for index in values):
            raise ValueError("schedule contains an invalid training row index")
        normalized.append(values)
    return normalized


def schedule_sha256(schedule: Sequence[Sequence[int]]) -> str:
    """Hash a supplied schedule; short schedules are accepted by tiny tests."""
    normalized = _validate_schedule(schedule, row_count=None)
    return hashlib.sha256(_canonical_bytes(normalized)).hexdigest()


def _checked_rows(rows: Sequence[PreparedRow], name: str) -> tuple[PreparedRow, ...]:
    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows):
        raise ValueError(f"{name} must contain at least one prepared row")
    if not all(isinstance(row, PreparedRow) for row in rows):
        raise TypeError(f"{name} must contain PreparedRow values")
    return tuple(rows)


def _padded_local_gates(rows: Sequence[PreparedRow], width: int,
                        fsm: FrozenLocalPrefixFSM) -> Tensor:
    gates = torch.zeros((len(rows), width, len(FACTOR_ORDER)), dtype=torch.bool)
    for index, row in enumerate(rows):
        gates[index, :row.decoder_length] = teacher_forced_gates(row, fsm)
    return gates


def _objective(value: str) -> Objective:
    if value not in OBJECTIVES:
        raise ValueError(f"unknown objective: {value!r}")
    return value  # type: ignore[return-value]


def _head_records(output: Mapping[str, Any], rows: Sequence[PreparedRow], *,
                  split: str, row_metadata: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    if row_metadata is not None and len(row_metadata) != len(rows):
        raise ValueError("row_metadata must match prepared row count")
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        vectors: dict[str, list[float]] = {}
        argmax: dict[str, int] = {}
        for field in FACTOR_ORDER:
            value = output["factor_probs"][field][index].detach().cpu()
            vectors[field] = [float(item) for item in value.tolist()]
            argmax[field] = int(value.argmax().item())
        record: dict[str, Any] = {
            "row_index": index,
            "factor_probability_vectors": vectors,
            "factor_argmax": argmax,
            "factor_targets": list(row.factor_targets),
            "target_frame": dict(row.target_frame),
            "split": split,
            "available": True,
        }
        if row_metadata is not None:
            metadata = row_metadata[index]
            if not isinstance(metadata, Mapping):
                raise TypeError("row_metadata entries must be mappings")
            record.update(dict(metadata))
            record["split"] = str(metadata.get("split", split))
        records.append(record)
    return records


def _validate_probability_records(records: Sequence[Mapping[str, Any]]) -> None:
    for record in records:
        vectors, argmax = record["factor_probability_vectors"], record["factor_argmax"]
        if set(vectors) != set(FACTOR_ORDER) or set(argmax) != set(FACTOR_ORDER):
            raise ValueError("head record must contain all canonical factors")
        for field in FACTOR_ORDER:
            values = vectors[field]
            tensor = torch.tensor(values, dtype=torch.float64)
            if (not values or not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any())
                    or not bool(torch.allclose(tensor.sum(), torch.tensor(1.0, dtype=tensor.dtype), atol=1e-6, rtol=1e-6))):
                raise ValueError(f"invalid {field} probability vector")
            if argmax[field] != int(tensor.argmax().item()):
                raise ValueError(f"invalid {field} probability argmax")


@dataclass(frozen=True)
class PairedInitialization:
    seed: int
    trainable_parameters: int
    initial_state_sha256: str
    bitwise_equal: bool

    def to_record(self) -> dict[str, Any]:
        return {"schema": HEAD_RUNNER_SCHEMA, "seed": self.seed,
                "trainable_parameters": self.trainable_parameters,
                "initial_state_sha256": self.initial_state_sha256,
                "bitwise_equal": self.bitwise_equal}


def paired_initialization_preflight(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    joint, factor_only = build_model("H1L1", seed=seed), build_model("H1L1", seed=seed)
    left, right = joint.state_dict(), factor_only.state_dict()
    if tuple(left) != tuple(right) or any(not torch.equal(left[key], right[key]) for key in left):
        raise ValueError("paired objective initial state_dict tensors differ")
    count = trainable_parameter_count(joint)
    if count != PARAMETER_COUNT or trainable_parameter_count(factor_only) != PARAMETER_COUNT:
        raise ValueError(f"H1L1 parameter count changed: expected {PARAMETER_COUNT}, got {count}")
    digest = state_sha256(left)
    if digest != state_sha256(right):
        raise ValueError("paired objective initial state digest differs")
    return PairedInitialization(seed, count, digest, True).to_record()


def _training_step(model: BenchmarkV3Model, optimizer: torch.optim.Optimizer,
                   batch: Any, objective: Objective,
                   fsm: FrozenLocalPrefixFSM) -> float:
    model.train(); optimizer.zero_grad(set_to_none=True)
    if objective == "JOINT":
        gates = _padded_local_gates(batch.rows, batch.decoder_input_ids.shape[1], fsm)
        output = model(batch.source_ids, batch.decoder_input_ids,
                       source_mask=batch.source_mask, labels=batch.labels,
                       factor_targets=batch.factor_targets, local_gates=gates)
        loss = output["lm_loss"] + output["factor_loss"]
    else:
        output = model(batch.source_ids, source_mask=batch.source_mask,
                       factor_targets=batch.factor_targets)
        loss = output["factor_loss"]
    if not torch.isfinite(loss):
        raise ValueError("non-finite head training loss")
    loss.backward(); clip_grad_norm_(model.parameters(), GRADIENT_CLIP); optimizer.step()
    return float(loss.detach().cpu())


@dataclass
class HeadTrainingResult:
    objective: Objective
    seed: int
    model: BenchmarkV3Model
    descriptor_sha256: str
    schedule_sha256: str
    initial_state_sha256: str
    final_state_sha256: str
    losses: list[float]
    train_head_records: list[dict[str, Any]]
    training_wall_seconds: float

    @property
    def trainable_parameters(self) -> int:
        return trainable_parameter_count(self.model)

    @property
    def optimizer_updates(self) -> int:
        return len(self.losses)

    def to_record(self) -> dict[str, Any]:
        vectors = [deepcopy(item["factor_probability_vectors"]) for item in self.train_head_records]
        argmax = [deepcopy(item["factor_argmax"]) for item in self.train_head_records]
        return {"schema": HEAD_RUNNER_SCHEMA, "objective": self.objective, "seed": self.seed,
                "arm": "H1L1", "trainable_parameters": self.trainable_parameters,
                "descriptor_sha256": self.descriptor_sha256, "schedule_sha256": self.schedule_sha256,
                "initial_state_sha256": self.initial_state_sha256, "final_state_sha256": self.final_state_sha256,
                "optimizer_updates": self.optimizer_updates, "losses": list(self.losses),
                "training_wall_seconds": self.training_wall_seconds,
                "train_head_records": deepcopy(self.train_head_records),
                "train_head_probability_vectors": vectors, "train_head_argmax": argmax}


def train_head_mode(objective: Objective, seed: int, train_rows: Sequence[PreparedRow],
                    schedule: Sequence[Sequence[int]], descriptor: Mapping[str, Any] | None = None,
                    *, fsm: FrozenLocalPrefixFSM | None = None,
                    train_metadata: Sequence[Mapping[str, Any]] | None = None) -> HeadTrainingResult:
    objective = _objective(objective)
    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    rows = _checked_rows(train_rows, "train_rows")
    normalized = _validate_schedule(schedule, row_count=len(rows))
    digest = descriptor_sha256(descriptor)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    preflight = paired_initialization_preflight(seed)
    model = build_model("H1L1", seed=seed)
    if state_sha256(model.state_dict()) != preflight["initial_state_sha256"]:
        raise ValueError("run initialization differs from paired preflight")
    if trainable_parameter_count(model) != PARAMETER_COUNT:
        raise ValueError("H1L1 parameter count changed")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS,
                                 eps=ADAM_EPS, weight_decay=0.0)
    machine = FrozenLocalPrefixFSM() if fsm is None else fsm
    initial = state_sha256(model.state_dict()); losses: list[float] = []
    started = time.perf_counter()
    for indices in normalized:
        losses.append(_training_step(model, optimizer, collate_rows([rows[i] for i in indices]), objective, machine))
    training_wall_seconds = time.perf_counter() - started
    model.eval()
    train = evaluate_head_outputs(model, rows, descriptor=descriptor, split="train",
                                  row_metadata=train_metadata)
    return HeadTrainingResult(objective, seed, model, digest, schedule_sha256(normalized), initial,
                              state_sha256(model.state_dict()), losses, train["head_records"],
                              training_wall_seconds)


@torch.no_grad()
def evaluate_head_outputs(model: BenchmarkV3Model, rows: Sequence[PreparedRow],
                          descriptor: Mapping[str, Any] | None = None, *, split: str = "confirmation",
                          objective: Objective | None = None, seed: int | None = None,
                          row_metadata: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if not isinstance(model, BenchmarkV3Model) or model.arm != "H1L1":
        raise ValueError("head evaluation requires an H1L1 BenchmarkV3Model")
    checked = _checked_rows(rows, "rows"); batch = collate_rows(checked); model.eval()
    output = model(batch.source_ids, source_mask=batch.source_mask)
    records = _head_records(output, checked, split=split, row_metadata=row_metadata)
    _validate_probability_records(records)
    return {"schema": HEAD_RUNNER_SCHEMA, "arm": "H1L1",
            "objective": None if objective is None else _objective(objective), "seed": seed,
            "split": str(split), "descriptor_sha256": descriptor_sha256(descriptor),
            "row_count": len(records), "head_records": records}


def run_head_mode(objective: Objective, seed: int, train_rows: Sequence[PreparedRow],
                  confirmation_rows: Sequence[PreparedRow], schedule: Sequence[Sequence[int]],
                  descriptor: Mapping[str, Any] | None = None, *, fsm: FrozenLocalPrefixFSM | None = None,
                  train_metadata: Sequence[Mapping[str, Any]] | None = None,
                  confirmation_metadata: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    training = train_head_mode(objective, seed, train_rows, schedule, descriptor, fsm=fsm,
                               train_metadata=train_metadata)
    confirmation = evaluate_head_outputs(training.model, confirmation_rows, descriptor=descriptor,
                                         split="confirmation", objective=training.objective, seed=seed,
                                         row_metadata=confirmation_metadata)
    record = training.to_record(); records = confirmation["head_records"]
    record["confirmation_head_records"] = records; record["confirmation_row_count"] = len(records)
    record["confirmation_head_probability_vectors"] = [deepcopy(item["factor_probability_vectors"]) for item in records]
    record["confirmation_head_argmax"] = [deepcopy(item["factor_argmax"]) for item in records]
    return record


def training_step(model: BenchmarkV3Model, optimizer: torch.optim.Optimizer, batch: Any,
                  objective: Objective, *, fsm: FrozenLocalPrefixFSM | None = None) -> float:
    return _training_step(model, optimizer, batch, _objective(objective),
                          FrozenLocalPrefixFSM() if fsm is None else fsm)


train_benchmark_v3_head = train_head_mode
evaluate_benchmark_v3_head = evaluate_head_outputs
evaluate_head_mode = evaluate_head_outputs
run_benchmark_v3_head = run_head_mode

__all__ = ["BATCH_SIZE", "HEAD_RUNNER_SCHEMA", "HeadTrainingResult", "OBJECTIVES",
           "PARAMETER_COUNT", "SEEDS", "TRAIN_STEPS", "descriptor_sha256",
           "evaluate_benchmark_v3_head", "evaluate_head_outputs", "evaluate_head_mode",
           "paired_initialization_preflight", "run_benchmark_v3_head", "run_head_mode",
           "schedule_sha256", "training_step", "train_benchmark_v3_head", "train_head_mode"]
