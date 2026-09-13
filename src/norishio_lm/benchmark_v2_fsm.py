"""Frozen, causal local-prefix FSM for benchmark-v2 arm B.

The FSM is deliberately independent of benchmark rows and model state.  It
compiles the two target templates from the authored specification into byte
strings whose UTF-8 bytes carry only one of four structural tags:

``literal``, ``participant``, ``time``, and ``predicate``.

For a consumed BOS-prefixed token history, all grammar candidates beginning
with the consumed bytes are retained.  A factor gate is true exactly when the
retained candidates are nonempty and every retained candidate has a next byte
with that factor's tag.  Event and operator intentionally share the predicate
tag.  The batch adapter computes each row position from the history ending at
that position, so it never reads a future token.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import string
from typing import Any, Iterable, Mapping, Sequence


PAD = 0
BOS = 1
EOS = 2
SEP = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260

FACTOR_ORDER = ("participant", "time", "event", "operator")


class ByteTag(str, Enum):
    """Structural tag attached to each byte of a compiled target."""

    LITERAL = "literal"
    PARTICIPANT = "participant"
    TIME = "time"
    PREDICATE = "predicate"


@dataclass(frozen=True)
class GrammarCandidate:
    """One allowed target string represented as raw UTF-8 bytes and tags.

    The candidate intentionally contains no row, target-frame, or template
    identifier.  ``byte_values`` are raw 0--255 UTF-8 bytes; ``token_ids`` is
    their benchmark-v2 byte-token representation.
    """

    text: str
    byte_values: tuple[int, ...]
    tags: tuple[ByteTag, ...]

    def __post_init__(self) -> None:
        if len(self.byte_values) != len(self.tags):
            raise ValueError("candidate bytes and tags must have equal length")
        if any(type(value) is not int or not 0 <= value <= 255 for value in self.byte_values):
            raise ValueError("candidate byte values must be raw UTF-8 bytes")
        if len(self.text.encode("utf-8")) != len(self.byte_values):
            raise ValueError("candidate text does not match its UTF-8 bytes")

    @property
    def bytes(self) -> tuple[int, ...]:
        """Alias for the raw UTF-8 byte sequence."""

        return self.byte_values

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Benchmark-v2 byte IDs for this candidate."""

        return tuple(BYTE_OFFSET + value for value in self.byte_values)


def _default_spec() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "data" / "benchmark_v2" / "spec.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _as_spec(spec: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _default_spec() if spec is None else spec


def _field_tag(field: str) -> ByteTag:
    if field == "participant":
        return ByteTag.PARTICIPANT
    if field == "time":
        return ByteTag.TIME
    if field == "predicate":
        return ByteTag.PREDICATE
    raise ValueError(f"unsupported target template field: {field}")


def _render_tagged_template(template: str, slots: Mapping[str, str]) -> tuple[str, bytes, tuple[ByteTag, ...]]:
    """Render one target template while preserving literal/slot byte tags."""

    formatter = string.Formatter()
    text_parts: list[str] = []
    raw_parts: list[bytes] = []
    tag_parts: list[tuple[ByteTag, ...]] = []
    for literal, field_name, format_spec, conversion in formatter.parse(template):
        if literal:
            literal_bytes = literal.encode("utf-8")
            text_parts.append(literal)
            raw_parts.append(literal_bytes)
            tag_parts.append((ByteTag.LITERAL,) * len(literal_bytes))
        if field_name is None:
            continue
        if format_spec or conversion:
            raise ValueError("target templates must use plain named fields")
        if field_name not in slots:
            raise ValueError(f"unsupported target template field: {field_name}")
        value = slots[field_name]
        value_bytes = value.encode("utf-8")
        text_parts.append(value)
        raw_parts.append(value_bytes)
        tag_parts.append((_field_tag(field_name),) * len(value_bytes))
    text = "".join(text_parts)
    raw = b"".join(raw_parts)
    tags = tuple(tag for part in tag_parts for tag in part)
    if raw != text.encode("utf-8"):
        raise ValueError("tagged template rendering lost UTF-8 bytes")
    return text, raw, tags


def compile_target_grammar(spec: Mapping[str, Any] | None = None) -> tuple[GrammarCandidate, ...]:
    """Compile every allowed benchmark-v2 target string with byte tags.

    The result contains two templates × six participants × six times × four
    events × four operators.  It is built from the authored specification
    only; no generated row or reference target is accepted by this API.
    """

    source = _as_spec(spec)
    participants = source.get("participants")
    times = source.get("times")
    events = source.get("events")
    operators = source.get("operators")
    templates = source.get("target_templates")
    if not isinstance(participants, list) or not isinstance(times, list):
        raise ValueError("spec participants and times must be lists")
    if not isinstance(events, list) or not isinstance(operators, list):
        raise ValueError("spec events and operators must be lists")
    if not isinstance(templates, list) or not templates:
        raise ValueError("spec target_templates must be a nonempty list")

    candidates: list[GrammarCandidate] = []
    for template in templates:
        if not isinstance(template, str):
            raise ValueError("target templates must be strings")
        for participant in participants:
            for time in times:
                for event in events:
                    if not isinstance(event, Mapping) or not isinstance(event.get("predicates"), Mapping):
                        raise ValueError("events must contain predicate mappings")
                    for operator in operators:
                        if not isinstance(participant, Mapping) or not isinstance(time, Mapping):
                            raise ValueError("participants and times must be mappings")
                        predicate = event["predicates"].get(operator)
                        if not all(isinstance(value, str) for value in (
                                participant.get("surface"), time.get("surface"), predicate)):
                            raise ValueError("factor surfaces and predicates must be strings")
                        rendered, raw, tags = _render_tagged_template(
                            template,
                            {"participant": participant["surface"],
                             "time": time["surface"],
                             "predicate": predicate},
                        )
                        candidates.append(GrammarCandidate(rendered, tuple(raw), tags))
    return tuple(candidates)


def _normalise_history(history: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    """Validate a BOS-prefixed history and return consumed raw bytes.

    ``ValueError`` is used for malformed token histories.  The FSM's public
    gate method catches this and returns all-zero gates, while this helper
    remains useful for callers that want strict validation.
    """

    values = tuple(history)
    if not values or values[0] != BOS:
        raise ValueError("token history must start with BOS")
    result: list[int] = []
    for token in values[1:]:
        if type(token) is not int or not BYTE_OFFSET <= token < VOCAB_SIZE:
            raise ValueError("history may contain only byte IDs after BOS")
        result.append(token - BYTE_OFFSET)
    return tuple(result)


class FrozenLocalPrefixFSM:
    """Causal local-injection FSM compiled from the frozen target grammar."""

    def __init__(self, spec: Mapping[str, Any] | None = None) -> None:
        self.candidates = compile_target_grammar(spec)
        # Compile the same candidate-retention rule into a prefix lookup once.
        # ``None`` marks a completed candidate, which forces all gates off.
        next_tags: dict[tuple[int, ...], set[ByteTag | None]] = {}
        for candidate in self.candidates:
            for length in range(len(candidate.byte_values) + 1):
                prefix = candidate.byte_values[:length]
                tag = candidate.tags[length] if length < len(candidate.tags) else None
                next_tags.setdefault(prefix, set()).add(tag)
        self._next_tags = {prefix: frozenset(tags) for prefix, tags in next_tags.items()}

    def compatible_candidates(self, history: Sequence[int] | Iterable[int]) -> tuple[GrammarCandidate, ...]:
        """Return all candidates compatible with consumed bytes in ``history``."""

        try:
            consumed = _normalise_history(history)
        except (TypeError, ValueError):
            return ()
        length = len(consumed)
        return tuple(candidate for candidate in self.candidates
                     if candidate.byte_values[:length] == consumed)

    def gates(self, history: Sequence[int] | Iterable[int]) -> tuple[bool, bool, bool, bool]:
        """Return participant/time/event/operator gates for one history.

        The history includes the current decoder input token.  Thus the gates
        describe the next byte predicted from that history, preserving causal
        generation alignment.  A completed candidate counts as having no next
        tag and therefore forces all gates off.
        """

        try:
            consumed = _normalise_history(history)
        except (TypeError, ValueError):
            return (False, False, False, False)
        next_tags = self._next_tags.get(consumed)
        if not next_tags or None in next_tags:
            return (False, False, False, False)
        return tuple(next_tags == {wanted}
                     for wanted in (ByteTag.PARTICIPANT, ByteTag.TIME,
                                    ByteTag.PREDICATE, ByteTag.PREDICATE))  # type: ignore[return-value]

    def next_tags(self, history: Sequence[int] | Iterable[int]) -> tuple[ByteTag, ...]:
        """Return retained candidates' next tags, or an empty tuple if invalid."""

        try:
            consumed = _normalise_history(history)
        except (TypeError, ValueError):
            return ()
        next_tags = self._next_tags.get(consumed)
        if not next_tags or None in next_tags:
            return ()
        return tuple(sorted(next_tags, key=lambda tag: tag.value))  # type: ignore[union-attr]

    def batch_gates(self, input_ids: Any, *, as_tensor: bool = True) -> Any:
        """Compute causal gates with shape ``[B, T, 4]``.

        Position ``t`` uses ``input_ids[:, :t + 1]`` only.  A torch tensor is
        returned when torch is available and ``as_tensor`` is true; otherwise a
        nested tuple of booleans is returned.  No row, target, or future token
        is read.
        """

        rows, device = _batch_rows(input_ids)
        output = [
            [self.gates(row[:position + 1]) for position in range(len(row))]
            for row in rows
        ]
        if not as_tensor:
            return tuple(tuple(step for step in row) for row in output)
        try:
            import torch
        except ImportError:
            return tuple(tuple(step for step in row) for row in output)
        # Explicit dimensions preserve the [B, T, 4] contract for empty
        # batches and zero-length histories as well.
        if not rows or not rows[0]:
            tensor = torch.zeros((len(rows), len(rows[0]) if rows else 0, 4), dtype=torch.bool)
        else:
            tensor = torch.tensor(output, dtype=torch.bool)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    __call__ = gates
    gates_for_history = gates


def _batch_rows(input_ids: Any) -> tuple[list[tuple[int, ...]], Any]:
    """Convert a [B,T] sequence or optional torch tensor into Python rows."""

    device = getattr(input_ids, "device", None)
    if hasattr(input_ids, "detach") and hasattr(input_ids, "cpu"):
        values = input_ids.detach().cpu().tolist()
    else:
        values = input_ids
    try:
        rows = [tuple(row) for row in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("input_ids must be a [B, T] batch") from exc
    if any(not isinstance(row, tuple) for row in rows):
        raise ValueError("input_ids must be a [B, T] batch")
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("input_ids rows must have equal length")
    return rows, device


def batch_fsm_gates(input_ids: Any, spec: Mapping[str, Any] | None = None, *,
                    as_tensor: bool = True) -> Any:
    """Functional wrapper for :meth:`FrozenLocalPrefixFSM.batch_gates`."""

    return FrozenLocalPrefixFSM(spec).batch_gates(input_ids, as_tensor=as_tensor)


def gates_for_history(history: Sequence[int] | Iterable[int],
                      spec: Mapping[str, Any] | None = None) -> tuple[bool, bool, bool, bool]:
    """Functional wrapper for one BOS-prefixed causal history."""

    return FrozenLocalPrefixFSM(spec).gates(history)


# Friendly aliases for integrations that prefer a shorter name.
LocalPrefixFSM = FrozenLocalPrefixFSM
FrozenPrefixFSM = FrozenLocalPrefixFSM
compile_grammar = compile_target_grammar
fsm_gates = batch_fsm_gates


__all__ = [
    "BOS", "BYTE_OFFSET", "EOS", "FACTOR_ORDER", "PAD", "SEP", "VOCAB_SIZE",
    "ByteTag", "GrammarCandidate", "FrozenLocalPrefixFSM", "FrozenPrefixFSM",
    "LocalPrefixFSM", "batch_fsm_gates", "compile_grammar", "compile_target_grammar",
    "fsm_gates", "gates_for_history",
]
