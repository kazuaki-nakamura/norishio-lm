"""Frozen CPU training and development evaluation for benchmark-v3.

This module is intentionally a small execution layer around the contracts in
``benchmark_v3_model``, ``benchmark_v3_fsm``, ``benchmark_v3_protocol`` and
``benchmark_v3_metrics``.  It owns row preparation, strict padding, the fixed
schedule, greedy generation, and scorer-compatible diagnostics.  It does not
open the final holdout and it does not run at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from .benchmark_v3_contract import INTERVENTION_RULE
from . import benchmark_v3_data as benchmark
from .benchmark_v3_audit import audit_frozen_bundle
from .benchmark_v3_fsm import FrozenLocalPrefixFSM
from .benchmark_v3_metrics import score_benchmark_v3
from .benchmark_v3_model import (
    ARM_IDS,
    BOS_ID,
    BYTE_OFFSET,
    EOS_ID,
    FACTOR_ORDER,
    PAD_ID,
    SEP_ID,
    VOCAB_SIZE,
    BenchmarkV3Model,
    build_model,
)
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_tournament import ARM_COUNTS, BENCHMARK_DIGEST, SEEDS


TRAIN_STEPS = 600
BATCH_SIZE = 16
MAX_NEW_TOKENS = 96
GRADIENT_CLIP = 1.0
LEARNING_RATE = 0.003
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0

_DEFAULT_VOCAB: FactorVocabulary | None = None


@lru_cache(maxsize=1)
def _fixture_module() -> Any:
    """Return the committed v3 generator through its public module API."""

    return benchmark


@lru_cache(maxsize=1)
def _validated_bundle() -> dict[str, list[dict[str, Any]]]:
    benchmark = _fixture_module()
    bundle = benchmark.build()
    report = benchmark.validate(bundle)
    benchmark.check_expected(report)
    audit_frozen_bundle(bundle)
    if report.get("content_digest_sha256") != BENCHMARK_DIGEST:
        raise ValueError("benchmark fixture digest differs from the frozen tournament")
    return bundle


@dataclass(frozen=True)
class FactorVocabulary:
    """Committed factor ID order used to tensorize frame labels."""

    values: dict[str, tuple[str, ...]]

    @property
    def ids(self) -> dict[str, dict[str, int]]:
        return {field: {value: index for index, value in enumerate(values)}
                for field, values in self.values.items()}

    def encode(self, frame: Mapping[str, Any]) -> tuple[int, int, int, int]:
        mapping = self.ids
        result: list[int] = []
        for field in FACTOR_ORDER:
            if field not in frame or frame[field] not in mapping[field]:
                raise ValueError(f"frame contains unknown {field} value")
            result.append(mapping[field][frame[field]])
        return tuple(result)  # type: ignore[return-value]

    def decode(self, values: Sequence[int]) -> dict[str, str]:
        if len(values) != len(FACTOR_ORDER):
            raise ValueError("factor values must contain four classes")
        result: dict[str, str] = {}
        for index, field in enumerate(FACTOR_ORDER):
            value = values[index]
            if type(value) is not int or not 0 <= value < len(self.values[field]):
                raise ValueError(f"invalid {field} class")
            result[field] = self.values[field][value]
        return result


def factor_vocabulary(spec: Mapping[str, Any] | None = None) -> FactorVocabulary:
    """Derive factor classes from the committed authored specification."""

    global _DEFAULT_VOCAB
    if spec is None and _DEFAULT_VOCAB is not None:
        return _DEFAULT_VOCAB
    source = _fixture_module().spec_data() if spec is None else spec
    values = {
        "participant": tuple(entry["id"] for entry in source["participants"]),
        "time": tuple(entry["id"] for entry in source["times"]),
        "event": tuple(entry["id"] for entry in source["events"]),
        "operator": tuple(source["operators"]),
    }
    expected = {"participant": 6, "time": 6, "event": 4, "operator": 4}
    if {field: len(values[field]) for field in FACTOR_ORDER} != expected:
        raise ValueError("factor vocabulary differs from frozen benchmark")
    if any(len(set(values[field])) != len(values[field]) for field in FACTOR_ORDER):
        raise ValueError("factor vocabulary contains duplicates")
    result = FactorVocabulary(values)
    if spec is None:
        _DEFAULT_VOCAB = result
    return result


@dataclass(frozen=True)
class PreparedRow:
    """A train or development row after source/decoder tokenization."""

    source_ids: tuple[int, ...]
    decoder_input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    factor_targets: tuple[int, int, int, int]
    target_text: str
    target_frame: dict[str, str]

    @property
    def source_length(self) -> int:
        return len(self.source_ids)

    @property
    def decoder_length(self) -> int:
        return len(self.decoder_input_ids)


def prepare_row(row: Mapping[str, Any], vocabulary: FactorVocabulary | None = None) -> PreparedRow:
    """Prepare one generated benchmark row without exposing hidden fields."""

    benchmark = _fixture_module()
    vocab = factor_vocabulary() if vocabulary is None else vocabulary
    source = benchmark.source_ids(dict(row))
    frame = row.get("targets", {}).get("frame")
    target_text = row.get("targets", {}).get("text")
    if not isinstance(frame, Mapping) or not isinstance(target_text, str):
        raise ValueError("benchmark row is missing target frame/text")
    target_bytes = tuple(BYTE_OFFSET + value for value in target_text.encode("utf-8"))
    # The decoder is a target-only causal LM.  Source bytes are consumed by
    # the source encoder and never appear in decoder history, preserving the
    # exact BOS self-history distribution used by greedy generation.
    input_ids = (BOS_ID, *target_bytes)
    labels = (*target_bytes, EOS_ID)
    if len(input_ids) != len(labels) or not input_ids:
        raise ValueError("decoder inputs and labels must have equal length")
    if any(value < 0 or value >= VOCAB_SIZE for value in input_ids):
        raise ValueError("decoder input contains an out-of-range token")
    if any(value != -100 and (value < 0 or value >= VOCAB_SIZE) for value in labels):
        raise ValueError("decoder label contains an out-of-range token")
    return PreparedRow(
        source_ids=tuple(int(value) for value in source),
        decoder_input_ids=input_ids,
        labels=labels,
        factor_targets=vocab.encode(frame),
        target_text=target_text,
        target_frame={field: str(frame[field]) for field in FACTOR_ORDER},
    )


@dataclass
class PaddedBatch:
    """Strictly padded CPU tensors and lengths for a mini-batch."""

    source_ids: Tensor
    source_mask: Tensor
    decoder_input_ids: Tensor
    decoder_mask: Tensor
    labels: Tensor
    factor_targets: Tensor
    rows: tuple[PreparedRow, ...]


def _pad_rows(rows: Sequence[PreparedRow]) -> PaddedBatch:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    source_width = max(row.source_length for row in rows)
    decoder_width = max(row.decoder_length for row in rows)
    source = torch.full((len(rows), source_width), PAD_ID, dtype=torch.long)
    source_mask = torch.zeros((len(rows), source_width), dtype=torch.bool)
    decoder = torch.full((len(rows), decoder_width), PAD_ID, dtype=torch.long)
    decoder_mask = torch.zeros((len(rows), decoder_width), dtype=torch.bool)
    labels = torch.full((len(rows), decoder_width), -100, dtype=torch.long)
    factors = torch.empty((len(rows), len(FACTOR_ORDER)), dtype=torch.long)
    for index, row in enumerate(rows):
        source[index, :row.source_length] = torch.tensor(row.source_ids, dtype=torch.long)
        source_mask[index, :row.source_length] = True
        decoder[index, :row.decoder_length] = torch.tensor(row.decoder_input_ids, dtype=torch.long)
        decoder_mask[index, :row.decoder_length] = True
        labels[index, :row.decoder_length] = torch.tensor(row.labels, dtype=torch.long)
        factors[index] = torch.tensor(row.factor_targets, dtype=torch.long)
    # Padding is structural, never a decoder supervision target.
    if bool(labels[~decoder_mask].ne(-100).any()):
        raise AssertionError("decoder padding leaked into labels")
    if bool(source[source_mask].eq(PAD_ID).any()):
        raise AssertionError("source padding appeared inside a live source")
    return PaddedBatch(source, source_mask, decoder, decoder_mask, labels, factors,
                       tuple(rows))


def collate_rows(rows: Sequence[PreparedRow]) -> PaddedBatch:
    """Public adapter used by tests and callers needing exact padding."""

    return _pad_rows(rows)


def load_train_rows() -> list[dict[str, Any]]:
    """Load only the generated training rows from the committed fixture."""

    rows = deepcopy(_validated_bundle()["train"])
    if len(rows) != 384:
        raise ValueError("benchmark-v3 train row count changed")
    return rows


def load_development_rows() -> list[dict[str, Any]]:
    """Load the preregistered diagnostic-validation split only."""

    rows = deepcopy(_validated_bundle()["diagnostic-validation"])
    if len(rows) != 384:
        raise ValueError("benchmark-v3 development row count changed")
    return rows


def teacher_forced_gates(row: PreparedRow, fsm: FrozenLocalPrefixFSM | None = None) -> Tensor:
    """Return prefix-local gates from consumed target bytes only."""

    machine = FrozenLocalPrefixFSM() if fsm is None else fsm
    result = torch.zeros((row.decoder_length, len(FACTOR_ORDER)), dtype=torch.bool)
    for position in range(row.decoder_length):
        # The decoder input at position zero is BOS.  Every later position
        # contains the next target byte in the consumed prefix.
        history = row.decoder_input_ids[:position + 1]
        result[position] = torch.tensor(machine.gates(history), dtype=torch.bool)
    return result


def _padded_local_gates(batch: PaddedBatch, fsm: FrozenLocalPrefixFSM) -> Tensor:
    gates = torch.zeros((len(batch.rows), batch.decoder_input_ids.shape[1], 4), dtype=torch.bool)
    for index, row in enumerate(batch.rows):
        gates[index, :row.decoder_length] = teacher_forced_gates(row, fsm)
    return gates


def training_step(
    model: BenchmarkV3Model,
    optimizer: torch.optim.Optimizer,
    batch: PaddedBatch,
    *,
    fsm: FrozenLocalPrefixFSM | None = None,
) -> float:
    """Run one optimizer update; this is the intentionally configurable test helper."""

    model.train()
    local_gates = None
    if model.uses_local_gates:
        local_gates = _padded_local_gates(batch, FrozenLocalPrefixFSM() if fsm is None else fsm)
    optimizer.zero_grad(set_to_none=True)
    kwargs: dict[str, Any] = {
        "source_mask": batch.source_mask,
        "labels": batch.labels,
        "local_gates": local_gates,
    }
    kwargs["factor_targets"] = batch.factor_targets
    output = model(batch.source_ids, batch.decoder_input_ids, **kwargs)
    loss = output["lm_loss"]
    loss = loss + output["factor_loss"]
    if not torch.isfinite(loss):
        raise ValueError("non-finite benchmark-v3 training loss")
    loss.backward()
    clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
    optimizer.step()
    return float(loss.detach().cpu())


# Private spelling retained as a small, explicit hook for focused tests.
_training_step = training_step


@dataclass
class TrainingResult:
    arm: str
    seed: int
    model: BenchmarkV3Model
    initial_state_sha256: str
    final_state_sha256: str
    schedule_sha256: str
    losses: list[float]

    @property
    def trainable_parameters(self) -> int:
        return trainable_parameter_count(self.model)


def train_benchmark_v3(
    arm: str,
    seed: int,
) -> TrainingResult:
    """Train one frozen arm for exactly 600 scheduled CPU updates."""

    if arm not in ARM_IDS or seed not in SEEDS:
        raise ValueError("arm or seed is not preregistered")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = build_model(arm, seed=seed)
    expected = ARM_COUNTS[arm]
    actual = trainable_parameter_count(model)
    if actual != expected:
        raise ValueError(f"parameter preflight failed: expected {expected}, got {actual}")
    schedule = batch_schedule(seed=seed)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
    )
    # The optimizer is created only after the exact parameter preflight.
    vocabulary = factor_vocabulary()
    train_rows = [prepare_row(row, vocabulary) for row in load_train_rows()]
    initial_digest = state_sha256(model.state_dict())
    fsm = FrozenLocalPrefixFSM() if model.uses_local_gates else None
    losses: list[float] = []
    for indices in schedule:
        batch = _pad_rows([train_rows[index] for index in indices])
        losses.append(training_step(model, optimizer, batch, fsm=fsm))
    model.eval()
    return TrainingResult(
        arm=arm, seed=seed, model=model,
        initial_state_sha256=initial_digest,
        final_state_sha256=state_sha256(model.state_dict()),
        schedule_sha256=schedule_sha256(schedule), losses=losses,
    )


FACTOR_PREDICTION_SPACE_SCHEMA = "norishio.issue36.factor-prediction-space.v1"


def factor_prediction_metadata(arm: str) -> dict[str, Any]:
    """Describe raw factor-logit codes and their canonical semantic decoding."""

    if arm not in ARM_IDS:
        raise ValueError(f"unknown benchmark-v3 arm: {arm}")
    raw_to_canonical = {
        field: list(range(len(factor_vocabulary().values[field])))
        for field in FACTOR_ORDER
    }
    return {
        "schema": FACTOR_PREDICTION_SPACE_SCHEMA,
        "arm": arm,
        "raw_code_space": "canonical_factor_class_indices",
        "canonical_space": "authored_structural_factor_label_space",
        "factor_loss_target_space": "authored_structural_factor_label_space",
        "canonicalization": "identity",
        "raw_to_canonical_indices": raw_to_canonical,
    }


def decode_factor_logits(
    logits: Mapping[str, Tensor], index: int, vocab: FactorVocabulary, *, arm: str,
) -> dict[str, Any]:
    """Decode raw factor logits while exposing raw and canonical spaces explicitly."""

    metadata = factor_prediction_metadata(arm)
    raw_indices = tuple(int(logits[field][index].argmax(dim=-1).item())
                        for field in FACTOR_ORDER)
    canonical_indices = tuple(
        metadata["raw_to_canonical_indices"][field][raw_indices[position]]
        for position, field in enumerate(FACTOR_ORDER)
    )
    return {
        "raw_code_indices": raw_indices,
        "canonical_indices": canonical_indices,
        "raw_code_frame": vocab.decode(raw_indices),
        "canonical_frame": vocab.decode(canonical_indices),
        "metadata": metadata,
    }


def _factor_frame(logits: Mapping[str, Tensor], index: int, vocab: FactorVocabulary,
                  *, arm: str) -> dict[str, str]:
    return decode_factor_logits(logits, index, vocab, arm=arm)["canonical_frame"]


def _decode_generated(tokens: Sequence[int]) -> dict[str, Any]:
    generated = list(tokens)
    ended_eos = bool(generated and generated[-1] == EOS_ID)
    if ended_eos:
        generated = generated[:-1]
    invalid = [token for token in generated if token < BYTE_OFFSET or token >= VOCAB_SIZE]
    raw = bytes(token - BYTE_OFFSET for token in generated if BYTE_OFFSET <= token < VOCAB_SIZE)
    try:
        text = raw.decode("utf-8") if not invalid else None
        valid_utf8 = text is not None
    except UnicodeDecodeError:
        text = None
        valid_utf8 = False
    return {
        "text": text,
        "tokens": list(tokens),
        "ended_eos": ended_eos,
        "valid_utf8": valid_utf8,
        "invalid_special_tokens": invalid,
        "unique_output": True,
    }


def _source_batch(source_rows: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]:
    if not source_rows:
        raise ValueError("source_rows cannot be empty")
    width = max(len(row) for row in source_rows)
    source = torch.full((len(source_rows), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(source_rows), width), dtype=torch.bool)
    for index, row in enumerate(source_rows):
        values = list(row)
        if not values or any(type(value) is not int or not 0 <= value < VOCAB_SIZE for value in values):
            raise ValueError("source_rows must contain valid nonempty token sequences")
        source[index, :len(values)] = torch.tensor(values, dtype=torch.long)
        mask[index, :len(values)] = True
    return source, mask


@torch.no_grad()
def greedy_generate_batch(
    model: BenchmarkV3Model,
    source_rows: Sequence[Sequence[int] | Tensor],
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    fsm: FrozenLocalPrefixFSM | None = None,
) -> list[dict[str, Any]]:
    """Batched BOS self-history generation with strict source padding."""

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("benchmark-v3 generation length is frozen at 96")
    values = [row.detach().cpu().tolist() if isinstance(row, Tensor) else list(row)
              for row in source_rows]
    source, source_mask = _source_batch(values)
    machine = FrozenLocalPrefixFSM() if model.uses_local_gates and fsm is None else fsm
    model.eval()
    generated = [[] for _ in values]
    active = list(range(len(values)))
    decoder = torch.full((len(values), 1), BOS_ID, dtype=torch.long)
    for _ in range(MAX_NEW_TOKENS):
        if not active:
            break
        active_source = source[active]
        active_mask = source_mask[active]
        # ``decoder`` is kept in the same compact order as ``active`` after
        # completed rows are removed; indexing it by global row IDs would
        # misalign (or overrun) the remaining histories.
        active_decoder = decoder
        gates = None
        if model.uses_local_gates:
            assert machine is not None
            gates = machine.batch_gates(active_decoder, as_tensor=True)
        output = model(active_source, active_decoder, source_mask=active_mask, local_gates=gates)
        next_tokens = output["logits"][:, -1].argmax(dim=-1).tolist()
        next_active: list[int] = []
        next_decoder: list[list[int]] = []
        for local_index, row_index in enumerate(active):
            token = int(next_tokens[local_index])
            generated[row_index].append(token)
            if token != EOS_ID:
                next_active.append(row_index)
                next_decoder.append(active_decoder[local_index].tolist() + [token])
        active = next_active
        if active:
            decoder = torch.tensor(next_decoder, dtype=torch.long)
    return [_decode_generated(tokens) for tokens in generated]


@torch.no_grad()
def greedy_generate(
    model: BenchmarkV3Model,
    source_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    fsm: FrozenLocalPrefixFSM | None = None,
) -> dict[str, Any]:
    """Generate one row from BOS using only the model's own decoder history."""

    return greedy_generate_batch(model, [source_ids], max_new_tokens=max_new_tokens, fsm=fsm)[0]


@torch.no_grad()
def greedy_generate_intervened(
    model: BenchmarkV3Model,
    source_ids: Sequence[int] | Tensor,
    factor: str,
    artificial_one_hot: Tensor,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    fsm: FrozenLocalPrefixFSM | None = None,
) -> dict[str, Any]:
    """Generate from the same BOS prefix while replacing one factor distribution."""

    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError("benchmark-v3 generation length is frozen at 96")
    values = source_ids.detach().cpu().tolist() if isinstance(source_ids, Tensor) else list(source_ids)
    source, source_mask = _source_batch([values])
    machine = FrozenLocalPrefixFSM() if model.uses_local_gates and fsm is None else fsm
    decoder = torch.tensor([[BOS_ID]], dtype=torch.long)
    generated: list[int] = []
    model.eval()
    for _ in range(MAX_NEW_TOKENS):
        gates = None
        if model.uses_local_gates:
            assert machine is not None
            gates = machine.batch_gates(decoder, as_tensor=True)
        output = model.forward_with_intervention(
            source, decoder, factor, artificial_one_hot,
            source_mask=source_mask, local_gates=gates,
        )
        token = int(output["logits"][:, -1].argmax(dim=-1).item())
        generated.append(token)
        if token == EOS_ID:
            break
        decoder = torch.tensor([[BOS_ID, *generated]], dtype=torch.long)
    return _decode_generated(generated)


def _source_identity(source_ids: Sequence[int]) -> str:
    payload = json.dumps(list(source_ids), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _probability_lists(probabilities: Mapping[str, Tensor], index: int = 0) -> dict[str, list[float]]:
    return {field: [float(value) for value in probabilities[field][index].detach().cpu().tolist()]
            for field in FACTOR_ORDER}


def alternate_intervention_one_hot(
    probabilities: Mapping[str, Tensor], factor: str, *, index: int = 0,
) -> tuple[Tensor, int, int]:
    """Select a deterministic class after the baseline argmax for intervention.

    The selected class is fixed by the pre-result rule
    ``(baseline_argmax + 1) % width``.  Returning the baseline and selected
    indices alongside the one-hot makes the intervention auditable without
    consulting target labels.
    """

    if factor not in FACTOR_ORDER:
        raise ValueError(f"unknown factor {factor!r}")
    selected = probabilities.get(factor)
    if not isinstance(selected, Tensor) or selected.ndim != 2:
        raise ValueError("factor probabilities must be rank-2 tensors")
    if type(index) is not int or not 0 <= index < selected.shape[0]:
        raise ValueError("probability row index is out of range")
    width = selected.shape[1]
    if width < 2:
        raise ValueError("intervention requires at least two factor classes")
    baseline_class = int(selected[index].argmax(dim=-1).item())
    intervention_class = (baseline_class + 1) % width
    one_hot = torch.zeros((1, width), dtype=torch.float32, device=selected.device)
    one_hot[0, intervention_class] = 1.0
    return one_hot, baseline_class, intervention_class


def _development_intervention_pairs(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Select one deterministic source-side pair for each factor.

    The first rows in committed diagnostic order are paired with the first
    row sharing every non-target factor and differing in the requested factor.
    Gold frames choose diagnostic pairings only; model inputs remain source
    ``context``/``text`` fields and no target is passed to the model.
    """

    result: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for factor in FACTOR_ORDER:
        found: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
        for baseline in rows:
            frame = baseline.get("targets", {}).get("frame", {})
            for changed in rows:
                other = changed.get("targets", {}).get("frame", {})
                if frame.get(factor) == other.get(factor):
                    continue
                if all(frame.get(field) == other.get(field)
                       for field in FACTOR_ORDER if field != factor):
                    found = (baseline, changed)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"no development intervention pair for {factor}")
        result.append((factor, found[0], found[1]))
    return result


@torch.no_grad()
def _evaluate_rows(model: BenchmarkV3Model, raw_rows: Sequence[Mapping[str, Any]],
                   *, intervention_pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score already-authorized frozen rows; public callers use the dev wrapper."""
    raw_rows = list(raw_rows)
    vocabulary = factor_vocabulary()
    prepared = [prepare_row(row, vocabulary) for row in raw_rows]
    train_support = _fixture_module().train_support({"train": load_train_rows()})
    batch = _pad_rows(prepared)
    fsm = FrozenLocalPrefixFSM() if model.uses_local_gates else None
    gates = _padded_local_gates(batch, fsm) if fsm is not None else None
    model.eval()
    output = model(batch.source_ids, batch.decoder_input_ids, source_mask=batch.source_mask,
                   labels=batch.labels, local_gates=gates)
    generations = greedy_generate_batch(model, [row.source_ids for row in prepared], fsm=fsm)
    rows_out: list[dict[str, Any]] = []
    for index, row in enumerate(prepared):
        intermediate = _factor_frame(output.factor_logits, index, vocabulary, arm=model.arm)
        expected = [label for label in row.labels if label != -100]
        predicted = output["logits"][index, :len(expected)].argmax(dim=-1).tolist()
        matched = sum(int(actual == wanted) for actual, wanted in zip(predicted, expected))
        rows_out.append({
            "target_frame": row.target_frame,
            "target_text": row.target_text,
            "intermediate_frame": intermediate,
            "generation": generations[index],
            "teacher_forced_bytes": {
                "exact": matched == len(expected),
                "matched_bytes": matched,
                "expected_bytes": len(expected),
            },
        })
    # Development-only locality diagnostic: four deterministic source-side
    # one-factor swaps, selected from the same diagnostic-validation rows.
    try:
        pairs = _development_intervention_pairs(intervention_pool)
    except ValueError:
        raise ValueError("evaluation split cannot supply frozen intervention pairs")
    benchmark = _fixture_module()
    intervention_sources = [
        benchmark.source_ids(dict(baseline)) for _, baseline, _ in pairs
    ] + [
        benchmark.source_ids(dict(changed)) for _, _, changed in pairs
    ]
    intervention_generations = greedy_generate_batch(model, intervention_sources, fsm=fsm)
    source_side_swaps = []
    probability_interventions = []
    for index, (factor, baseline_row, changed_row) in enumerate(pairs):
        source_side_swaps.append({
            "target_factor": factor,
            "baseline": intervention_generations[index],
            "changed": intervention_generations[len(pairs) + index],
            "baseline_target_frame": baseline_row["targets"]["frame"],
            "changed_target_frame": changed_row["targets"]["frame"],
        })
        baseline_prepared = prepare_row(baseline_row, vocabulary)
        source_tensor, source_mask = _source_batch([baseline_prepared.source_ids])
        baseline_output = model(source_tensor, source_mask=source_mask)
        effective_source = baseline_output.source_ids[0].detach().cpu().tolist()
        baseline_probabilities = baseline_output.factor_probs
        one_hot, baseline_class, intervention_class = alternate_intervention_one_hot(
            baseline_probabilities, factor,
        )
        intervened_probabilities = model.intervene_probabilities(
            baseline_probabilities, factor, one_hot,
        )
        probability_interventions.append({
            "factor": factor,
            "oracle": False,
            "intervention_rule": INTERVENTION_RULE,
            "baseline_argmax_class": baseline_class,
            "intervention_class": intervention_class,
            "baseline_source_identity": _source_identity(effective_source),
            "intervened_source_identity": _source_identity(effective_source),
            "baseline_decoder_prefix": [BOS_ID],
            "intervened_decoder_prefix": [BOS_ID],
            "baseline_probabilities": _probability_lists(baseline_probabilities),
            "intervened_probabilities": _probability_lists(intervened_probabilities),
            "intervened_one_hot": [float(value) for value in one_hot[0].tolist()],
            "baseline": intervention_generations[index],
            "intervened": greedy_generate_intervened(
                model, baseline_prepared.source_ids, factor, one_hot, fsm=fsm,
            ),
        })
    parser = benchmark.parse_target_text
    report = score_benchmark_v3(rows_out, parser, train_support=train_support,
                                source_side_swaps=source_side_swaps,
                                intermediate_probability_interventions=probability_interventions)
    report["intermediate_prediction_metadata"] = factor_prediction_metadata(model.arm)
    return report


@torch.no_grad()
def evaluate_benchmark_v3(
    model: BenchmarkV3Model,
    *,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate only authenticated diagnostic-validation rows."""

    frozen = load_development_rows()
    if rows is None:
        selected = frozen
    else:
        lookup = {row["id"]: row for row in frozen}
        selected = list(rows)
        benchmark = _fixture_module()
        for row in selected:
            identifier = row.get("id") if isinstance(row, Mapping) else None
            if identifier not in lookup or benchmark.canonical(dict(row)) != benchmark.canonical(lookup[identifier]):
                raise PermissionError("runner accepts only unchanged diagnostic-validation rows")
    return _evaluate_rows(model, selected, intervention_pool=frozen)


def run_benchmark_v3(
    arm: str,
    seed: int,
) -> dict[str, Any]:
    """Train one frozen run and score only its diagnostic-validation rows."""

    training = train_benchmark_v3(arm, seed)
    report = evaluate_benchmark_v3(training.model)
    return {"training": training, "report": report}


__all__ = [
    "ADAM_BETAS", "ADAM_EPS", "BATCH_SIZE", "FactorVocabulary", "GRADIENT_CLIP",
    "INTERVENTION_RULE", "LEARNING_RATE", "MAX_NEW_TOKENS", "PaddedBatch", "PreparedRow", "TRAIN_STEPS",
    "TrainingResult", "collate_rows", "evaluate_benchmark_v3", "factor_vocabulary",
    "factor_prediction_metadata", "decode_factor_logits", "alternate_intervention_one_hot",
    "greedy_generate",
    "greedy_generate_batch", "load_development_rows", "load_train_rows", "prepare_row",
    "run_benchmark_v3", "teacher_forced_gates", "train_benchmark_v3", "training_step",
]
