"""CPU runner for the future-only Issue #44 h0-bypass confirmation.

This runner reads only the new authored train/confirmation fixture.  It does
not import historical result, marker, checkpoint, or final-split artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import time
from typing import Any
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from . import benchmark_v3_h0_fixture as fixture
from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from .benchmark_v3_fsm import FrozenLocalPrefixFSM
from .benchmark_v3_h0_model import H0_ARM_IDS, PARAMETER_COUNT, FutureH0PathModel, build_future_h0_model
from .benchmark_v3_metrics import score_benchmark_v3
from .benchmark_v3_model import BOS_ID, BYTE_OFFSET, EOS_ID, FACTOR_ORDER, PAD_ID, SEP_ID, VOCAB_SIZE
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_runner import (
    ADAM_BETAS,
    ADAM_EPS,
    BATCH_SIZE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_NEW_TOKENS,
    TRAIN_STEPS,
    FactorVocabulary,
    PreparedRow,
    collate_rows,
    greedy_generate_batch,
    teacher_forced_gates,
    training_step,
)


SEEDS = (7, 17, 29)
WEIGHT_DECAY = 0.0


@lru_cache(maxsize=1)
def validated_h0_bundle() -> dict[str, list[dict[str, Any]]]:
    bundle = fixture.build()
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    return deepcopy(bundle)


@lru_cache(maxsize=1)
def h0_factor_vocabulary() -> FactorVocabulary:
    spec = fixture.spec_data()
    values = {
        "participant": tuple(item["id"] for item in spec["participants"]),
        "time": tuple(item["id"] for item in spec["times"]),
        "event": tuple(item["id"] for item in spec["events"]),
        "operator": tuple(spec["operators"]),
    }
    expected = {"participant": 6, "time": 6, "event": 4, "operator": 4}
    if {field: len(values[field]) for field in FACTOR_ORDER} != expected:
        raise ValueError("h0 fixture factor widths changed")
    if any(len(set(values[field])) != len(values[field]) for field in FACTOR_ORDER):
        raise ValueError("h0 fixture factor vocabulary contains duplicates")
    return FactorVocabulary(values)


def prepare_h0_row(row: Mapping[str, Any]) -> PreparedRow:
    """Tensorize one new-fixture row without passing evaluator fields to the model."""

    source = tuple(fixture.source_ids(row))
    targets = row.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("h0 row is missing targets")
    frame = targets.get("frame")
    text = targets.get("text")
    if not isinstance(frame, Mapping) or not isinstance(text, str):
        raise ValueError("h0 row is missing target frame/text")
    target_bytes = tuple(BYTE_OFFSET + value for value in text.encode("utf-8"))
    decoder = (BOS_ID, *target_bytes)
    labels = (*target_bytes, EOS_ID)
    if any(not 0 <= value < VOCAB_SIZE for value in (*source, *decoder, *labels)):
        raise ValueError("h0 row contains an out-of-range token")
    return PreparedRow(
        source_ids=source,
        decoder_input_ids=decoder,
        labels=labels,
        factor_targets=h0_factor_vocabulary().encode(frame),
        target_text=text,
        target_frame={field: str(frame[field]) for field in FACTOR_ORDER},
    )


def load_h0_train_rows() -> list[dict[str, Any]]:
    rows = deepcopy(validated_h0_bundle()["train"])
    if len(rows) != 384:
        raise ValueError("h0 fixture requires exactly 384 train rows")
    return rows


def load_h0_confirmation_rows() -> list[dict[str, Any]]:
    report = fixture.validate(validated_h0_bundle())
    rows = fixture.confirmation_rows(
        validated_h0_bundle(), manifest_digest=report["content_digest_sha256"],
    )
    if len(rows) != 96:
        raise ValueError("h0 fixture requires exactly 96 confirmation rows")
    return rows


def paired_initialization_preflight(seed: int) -> dict[str, Any]:
    """Prove paired arms start from bitwise-identical parameter tensors."""

    if seed not in SEEDS:
        raise ValueError("seed is not preregistered")
    ordinary = build_future_h0_model("H1L1", seed=seed)
    anchor = build_future_h0_model("H1L1_ANCHOR", seed=seed)
    left = ordinary.state_dict()
    right = anchor.state_dict()
    if tuple(left) != tuple(right) or any(not torch.equal(left[key], right[key]) for key in left):
        raise ValueError("paired initial state_dict tensors differ")
    if trainable_parameter_count(ordinary) != PARAMETER_COUNT or trainable_parameter_count(anchor) != PARAMETER_COUNT:
        raise ValueError("paired arm parameter count changed")
    left_digest = state_sha256(left)
    right_digest = state_sha256(right)
    if left_digest != right_digest:
        raise ValueError("paired initial state digest differs")
    return {
        "seed": seed,
        "arms": list(H0_ARM_IDS),
        "trainable_parameters": PARAMETER_COUNT,
        "initial_state_sha256": left_digest,
        "bitwise_equal": True,
    }


@dataclass
class H0TrainingResult:
    arm: str
    seed: int
    model: FutureH0PathModel
    initial_state_sha256: str
    final_state_sha256: str
    schedule_sha256: str
    losses: list[float]
    optimizer_updates: int
    wall_seconds: float


def train_h0_arm(arm: str, seed: int) -> H0TrainingResult:
    """Train one preregistered arm for exactly 600 deterministic CPU updates."""

    if arm not in H0_ARM_IDS or seed not in SEEDS:
        raise ValueError("arm or seed is not preregistered")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    preflight = paired_initialization_preflight(seed)
    model = build_future_h0_model(arm, seed=seed)
    if state_sha256(model.state_dict()) != preflight["initial_state_sha256"]:
        raise ValueError("run initialization differs from paired preflight")
    schedule = batch_schedule(seed=seed, train_rows=384, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)
    prepared = [prepare_h0_row(row) for row in load_h0_train_rows()]
    fsm = FrozenLocalPrefixFSM(fixture.spec_data())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    started = time.monotonic()
    losses: list[float] = []
    for indices in schedule:
        losses.append(training_step(model, optimizer, collate_rows([prepared[index] for index in indices]), fsm=fsm))
    elapsed = time.monotonic() - started
    model.eval()
    return H0TrainingResult(
        arm=arm,
        seed=seed,
        model=model,
        initial_state_sha256=str(preflight["initial_state_sha256"]),
        final_state_sha256=state_sha256(model.state_dict()),
        schedule_sha256=schedule_sha256(schedule),
        losses=losses,
        optimizer_updates=len(schedule),
        wall_seconds=elapsed,
    )


def _padded_gates(rows: Sequence[PreparedRow], width: int, fsm: FrozenLocalPrefixFSM) -> Tensor:
    result = torch.zeros((len(rows), width, len(FACTOR_ORDER)), dtype=torch.bool)
    for index, row in enumerate(rows):
        result[index, :row.decoder_length] = teacher_forced_gates(row, fsm)
    return result


def _source_identity(source_ids: Sequence[int]) -> str:
    payload = json.dumps(list(source_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def evaluate_h0_confirmation(
    model: FutureH0PathModel, *, constant_source_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate all 96 rows, optionally with the frozen OOD constant source.

    The constant-source path is an inference diagnostic for a trained H1L1
    model.  It is not another learned arm and is never evidence of a learned
    factor path.
    """

    if model.arm_id not in H0_ARM_IDS:
        raise ValueError("model is not an h0 confirmation arm")
    if constant_source_diagnostic and model.arm_id != "H1L1":
        raise ValueError("constant-source diagnostic is frozen to the H1L1 arm")
    authored_rows = load_h0_confirmation_rows()
    prepared = [prepare_h0_row(row) for row in authored_rows]
    batch = collate_rows(prepared)
    fsm = FrozenLocalPrefixFSM(fixture.spec_data())
    gates = _padded_gates(prepared, batch.decoder_input_ids.shape[1], fsm)
    model.eval()
    output = model(
        batch.source_ids, batch.decoder_input_ids, source_mask=batch.source_mask,
        labels=batch.labels, local_gates=gates,
    )
    evaluation_sources = (
        [(BOS_ID, SEP_ID)] * len(prepared)
        if constant_source_diagnostic
        else [row.source_ids for row in prepared]
    )
    if constant_source_diagnostic:
        source_width = max(len(values) for values in evaluation_sources)
        source_ids = torch.full((len(evaluation_sources), source_width), PAD_ID, dtype=torch.long)
        source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
        for index, values in enumerate(evaluation_sources):
            source_ids[index, :len(values)] = torch.tensor(values, dtype=torch.long)
            source_mask[index, :len(values)] = True
        output = model(
            source_ids, batch.decoder_input_ids, source_mask=source_mask,
            labels=batch.labels, local_gates=gates,
        )
    generated = greedy_generate_batch(
        model, evaluation_sources, max_new_tokens=MAX_NEW_TOKENS, fsm=fsm,
    )
    vocabulary = h0_factor_vocabulary()
    score_rows: list[dict[str, Any]] = []
    raw_confirmation_rows: list[dict[str, Any]] = []
    generated_text_counts: dict[str, int] = {}
    for item in generated:
        if isinstance(item.get("text"), str):
            generated_text_counts[item["text"]] = generated_text_counts.get(item["text"], 0) + 1
    for index, row in enumerate(prepared):
        factor_ids = [int(output.factor_logits[field][index].argmax().item()) for field in FACTOR_ORDER]
        factor_argmax = {field: factor_ids[position] for position, field in enumerate(FACTOR_ORDER)}
        factor_probabilities = {
            field: [float(value) for value in output.factor_probs[field][index].detach().cpu().tolist()]
            for field in FACTOR_ORDER
        }
        predicted = output["logits"][index, :row.decoder_length].argmax(dim=-1).tolist()
        matched = sum(actual == wanted for actual, wanted in zip(predicted, row.labels))
        parsed = fixture.parse_target_text(generated[index]["text"]) if isinstance(generated[index].get("text"), str) else None
        unique = not isinstance(generated[index].get("text"), str) or generated_text_counts[generated[index]["text"]] == 1
        teacher_forced = {
            "exact": matched == row.decoder_length,
            "matched_bytes": matched,
            "expected_bytes": row.decoder_length,
        }
        score_rows.append({
            "target_frame": row.target_frame,
            "target_text": row.target_text,
            "intermediate_frame": vocabulary.decode(factor_ids),
            "generation": {**generated[index], "unique_output": unique},
            "teacher_forced_bytes": teacher_forced,
        })
        if not generated[index]["valid_utf8"]:
            parse_failure_reason = "invalid_utf8"
        elif not generated[index]["ended_eos"]:
            parse_failure_reason = "maximum_length_without_eos"
        elif parsed is None:
            parse_failure_reason = "unparseable_generation"
        else:
            parse_failure_reason = None
        authored = authored_rows[index]
        raw_confirmation_rows.append({
            "arm": model.arm_id,
            "seed": int(model.seed),
            "row_id": str(authored["id"]),
            "support_group": str(authored["metadata"]["support_class"]),
            "source_identity": _source_identity(evaluation_sources[index]),
            "target_frame": row.target_frame,
            "target_text": row.target_text,
            "factor_probability_vectors": factor_probabilities,
            "factor_argmax": factor_argmax,
            "intermediate_frame": vocabulary.decode(factor_ids),
            "generated_token_ids": list(generated[index]["tokens"]),
            "generated_text": generated[index]["text"],
            "eos": bool(generated[index]["ended_eos"]),
            "utf8_valid": bool(generated[index]["valid_utf8"]),
            "unique": unique,
            "parsed_frame": parsed,
            "parse_failure_reason": parse_failure_reason,
            "teacher_forced_byte_counts": teacher_forced,
        })
    report = score_benchmark_v3(
        score_rows,
        fixture.parse_target_text,
        train_support=fixture.train_support({"train": load_h0_train_rows()}),
    )
    report["arm"] = model.arm_id
    report["confirmation_manifest_sha256"] = fixture.validate(validated_h0_bundle())["content_digest_sha256"]
    report["raw_confirmation_rows"] = raw_confirmation_rows
    report["input_ablation"] = "constant_bos_sep" if constant_source_diagnostic else None
    report["claim_scope"] = (
        "out_of_distribution_inference_diagnostic_not_learned_path_evidence"
        if constant_source_diagnostic
        else "ordinary_confirmation"
    )
    return report


def evaluate_h0_constant_source(model: FutureH0PathModel) -> dict[str, Any]:
    """Run the preregistered constant ``[BOS, SEP]`` OOD inference diagnostic."""

    return evaluate_h0_confirmation(model, constant_source_diagnostic=True)


def run_h0_arm(arm: str, seed: int) -> dict[str, Any]:
    training = train_h0_arm(arm, seed)
    return {"training": training, "ordinary_confirmation": evaluate_h0_confirmation(training.model)}


__all__ = [
    "H0TrainingResult",
    "SEEDS",
    "evaluate_h0_constant_source",
    "evaluate_h0_confirmation",
    "h0_factor_vocabulary",
    "load_h0_confirmation_rows",
    "load_h0_train_rows",
    "paired_initialization_preflight",
    "prepare_h0_row",
    "run_h0_arm",
    "train_h0_arm",
    "validated_h0_bundle",
]
