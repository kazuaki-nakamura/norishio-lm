"""Causal, prefix-only grammar gates for benchmark-v3.

The local conditioning path is compiled from the authored v3 target grammar,
never from a benchmark row.  Each candidate is represented by its UTF-8 byte
sequence and a tag for every byte.  A consumed BOS-prefixed history retains
all candidates with that exact byte prefix; a factor gate is true only when
every retained candidate assigns the next byte to that factor.  Event and
operator share the predicate span because the surface predicate is their
joint realization.
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
    """Structural tag attached to a candidate's next UTF-8 byte.

    ``EVENT`` and ``OPERATOR`` are aliases for ``PREDICATE``.  The v3 target
    grammar renders event and operator together as one predicate surface, so
    a predicate byte licenses both factor heads.
    """

    LITERAL = "literal"
    PARTICIPANT = "participant"
    TIME = "time"
    PREDICATE = "predicate"
    EVENT = "predicate"
    OPERATOR = "predicate"


@dataclass(frozen=True)
class GrammarCandidate:
    """One allowed v3 target string and its byte-level structural tags."""

    text: str
    byte_values: tuple[int, ...]
    tags: tuple[ByteTag, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("candidate text must be a string")
        if len(self.byte_values) != len(self.tags):
            raise ValueError("candidate bytes and tags must have equal length")
        if any(type(value) is not int or not 0 <= value <= 255
               for value in self.byte_values):
            raise ValueError("candidate byte values must be raw UTF-8 bytes")
        if len(self.text.encode("utf-8")) != len(self.byte_values):
            raise ValueError("candidate text does not match its UTF-8 bytes")
        if any(not isinstance(tag, ByteTag) for tag in self.tags):
            raise ValueError("candidate tags must be ByteTag values")

    @property
    def bytes(self) -> tuple[int, ...]:
        """Alias for the raw UTF-8 byte sequence."""

        return self.byte_values

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Benchmark byte-token IDs for this candidate."""

        return tuple(BYTE_OFFSET + value for value in self.byte_values)


def _default_spec() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "data" / "benchmark_v3_factor_path" / "spec.json"
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


def _render_tagged_template(
    template: str, slots: Mapping[str, str]
) -> tuple[str, bytes, tuple[ByteTag, ...]]:
    """Render one target template while retaining byte-level slot tags."""

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
        if not isinstance(value, str):
            raise ValueError("factor surfaces and predicates must be strings")
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


def compile_target_grammar(
    spec: Mapping[str, Any] | None = None,
) -> tuple[GrammarCandidate, ...]:
    """Compile every allowed v3 target string from the authored spec."""

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
    seen: set[str] = set()
    for template in templates:
        if not isinstance(template, str):
            raise ValueError("target templates must be strings")
        for participant in participants:
            for time in times:
                for event in events:
                    if (not isinstance(participant, Mapping)
                            or not isinstance(time, Mapping)
                            or not isinstance(event, Mapping)
                            or not isinstance(event.get("predicates"), Mapping)):
                        raise ValueError("factor entries must be mappings with predicate mappings")
                    for operator in operators:
                        predicate = event["predicates"].get(operator)
                        values = (participant.get("surface"), time.get("surface"), predicate)
                        if not all(isinstance(value, str) for value in values):
                            raise ValueError("factor surfaces and predicates must be strings")
                        rendered, raw, tags = _render_tagged_template(
                            template,
                            {"participant": participant["surface"],
                             "time": time["surface"],
                             "predicate": predicate},
                        )
                        if rendered in seen:
                            raise ValueError("target grammar contains duplicate candidate text")
                        seen.add(rendered)
                        candidates.append(GrammarCandidate(rendered, tuple(raw), tags))
    return tuple(candidates)


def _normalise_history(history: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    """Validate a BOS-prefixed history and return consumed raw bytes."""

    try:
        values = tuple(history)
    except (TypeError, ValueError) as exc:
        raise ValueError("token history must be an iterable") from exc
    if not values or type(values[0]) is not int or values[0] != BOS:
        raise ValueError("token history must start with BOS")
    result: list[int] = []
    for token in values[1:]:
        if type(token) is not int or not BYTE_OFFSET <= token < VOCAB_SIZE:
            raise ValueError("history may contain only byte IDs after BOS")
        result.append(token - BYTE_OFFSET)
    return tuple(result)


class FrozenLocalPrefixFSM:
    """Frozen causal local-injection FSM compiled from the v3 grammar."""

    def __init__(self, spec: Mapping[str, Any] | None = None) -> None:
        self.candidates = compile_target_grammar(spec)
        # ``None`` marks a completed candidate.  A completed short candidate
        # therefore makes the current next-byte gate zero even if another
        # candidate continues with the same bytes.
        next_tags: dict[tuple[int, ...], set[ByteTag | None]] = {}
        for candidate in self.candidates:
            for length in range(len(candidate.byte_values) + 1):
                prefix = candidate.byte_values[:length]
                tag = candidate.tags[length] if length < len(candidate.tags) else None
                next_tags.setdefault(prefix, set()).add(tag)
        self._next_tags = {
            prefix: frozenset(tags) for prefix, tags in next_tags.items()
        }

    def compatible_candidates(
        self, history: Sequence[int] | Iterable[int]
    ) -> tuple[GrammarCandidate, ...]:
        """Return candidates whose bytes begin with the consumed history."""

        try:
            consumed = _normalise_history(history)
        except (TypeError, ValueError):
            return ()
        length = len(consumed)
        return tuple(candidate for candidate in self.candidates
                     if candidate.byte_values[:length] == consumed)

    def gates(self, history: Sequence[int] | Iterable[int]) -> tuple[bool, bool, bool, bool]:
        """Return participant/time/event/operator gates for one history.

        The supplied history includes the current decoder input token.  Only
        bytes consumed so far are consulted; no row, target, template ID, or
        future byte is accepted by this API.
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
        """Return retained candidates' distinct next tags, or empty if invalid."""

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

        Position ``t`` uses only ``input_ids[:, :t + 1]``.  The input can be a
        Python batch or a rank-two torch tensor.  A torch bool tensor is
        returned by default when torch is available.
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
        width = len(rows[0]) if rows else 0
        if not rows or width == 0:
            tensor = torch.zeros((len(rows), width, 4), dtype=torch.bool)
        else:
            tensor = torch.tensor(output, dtype=torch.bool)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    __call__ = gates
    gates_for_history = gates


def _batch_rows(input_ids: Any) -> tuple[list[tuple[int, ...]], Any]:
    """Convert a rank-two sequence or optional torch tensor into Python rows."""

    device = getattr(input_ids, "device", None)
    if hasattr(input_ids, "detach") and hasattr(input_ids, "cpu"):
        if getattr(input_ids, "ndim", None) != 2:
            raise ValueError("input_ids must be a [B, T] batch")
        values = input_ids.detach().cpu().tolist()
    else:
        if isinstance(input_ids, (str, bytes)):
            raise ValueError("input_ids must be a [B, T] batch")
        try:
            values = list(input_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("input_ids must be a [B, T] batch") from exc
    try:
        rows = [tuple(row) for row in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("input_ids must be a [B, T] batch") from exc
    if any(isinstance(row, (str, bytes)) for row in values):
        raise ValueError("input_ids must be a [B, T] batch")
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("input_ids rows must have equal length")
    return rows, device


def batch_fsm_gates(
    input_ids: Any, spec: Mapping[str, Any] | None = None, *, as_tensor: bool = True
) -> Any:
    """Functional wrapper for :meth:`FrozenLocalPrefixFSM.batch_gates`."""

    return FrozenLocalPrefixFSM(spec).batch_gates(input_ids, as_tensor=as_tensor)


def gates_for_history(
    history: Sequence[int] | Iterable[int], spec: Mapping[str, Any] | None = None
) -> tuple[bool, bool, bool, bool]:
    """Functional wrapper for one BOS-prefixed history."""

    return FrozenLocalPrefixFSM(spec).gates(history)


# Explicit names make the same causal operation easy to use from diagnostics
# that distinguish teacher-forced prefixes from model-generated prefixes.
teacher_forced_gates = gates_for_history
generated_gates = gates_for_history

LocalPrefixFSM = FrozenLocalPrefixFSM
FrozenPrefixFSM = FrozenLocalPrefixFSM
compile_grammar = compile_target_grammar
fsm_gates = batch_fsm_gates


__all__ = [
    "BOS", "BYTE_OFFSET", "EOS", "FACTOR_ORDER", "PAD", "SEP", "VOCAB_SIZE",
    "ByteTag", "GrammarCandidate", "FrozenLocalPrefixFSM", "FrozenPrefixFSM",
    "LocalPrefixFSM", "batch_fsm_gates", "compile_grammar", "compile_target_grammar",
    "fsm_gates", "gates_for_history", "teacher_forced_gates", "generated_gates",
]
