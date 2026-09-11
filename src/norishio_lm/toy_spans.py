"""Exact byte spans for participant and time slots in authored toy targets.

The authored grammar is deliberately mechanical: a target is accepted only
when its text, concept, and one complete seed condition agree.  Slot offsets
are accumulated while rendering the parsed template, so a matching substring
elsewhere in the sentence can never be mistaken for a slot.
"""
from __future__ import annotations

import string
from collections.abc import Mapping, Sequence
from typing import Any

_REQUIRED_CONCEPT = (
    "event",
    "operators",
    "agent",
    "participant",
    "time",
    "location",
    "repeat_marked",
)


def _is_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _location(condition: Mapping[str, Any]) -> Any:
    if "location" in condition:
        return condition["location"]
    return "HERE" if condition.get("event") == "STAY" else "UNSPECIFIED"


def _concept_matches(condition: Mapping[str, Any], concept: Mapping[str, Any], person: str, time: str) -> bool:
    if any(field not in concept for field in _REQUIRED_CONCEPT):
        return False
    if concept["event"] != condition.get("event"):
        return False
    operators = condition.get("operators")
    if (not _is_string_sequence(operators) or
            not _is_string_sequence(concept["operators"]) or
            list(concept["operators"]) != list(operators)):
        return False
    return (
        concept["agent"] == condition.get("agent")
        and concept["participant"] == person
        and concept["time"] == time
        and concept["location"] == _location(condition)
        and concept["repeat_marked"] == bool(condition.get("repeat"))
    )


def authored_slot_spans(target: Mapping[str, Any], seed: Mapping[str, Any]) -> dict[str, dict[str, int]] | None:
    """Return exact UTF-8 byte spans for ``participant`` and ``time``.

    ``target`` must contain ``text`` and a complete seven-field ``concept``.
    The result uses zero-based, half-open byte offsets into ``target['text']``;
    BOS/EOS offsets are therefore not included.
    """
    if not isinstance(target, Mapping) or not isinstance(seed, Mapping):
        return None
    text = target.get("text")
    concept = target.get("concept")
    people = seed.get("people")
    times = seed.get("times")
    conditions = seed.get("conditions")
    if not isinstance(text, str) or not isinstance(concept, Mapping):
        return None
    if not _is_string_sequence(people) or not _is_string_sequence(times) or not _is_string_sequence(conditions):
        return None
    people = [value for value in people if isinstance(value, str) and value]
    times = [value for value in times if isinstance(value, str) and value]
    candidates: list[dict[str, dict[str, int]]] = []
    for condition in conditions:
        if not isinstance(condition, Mapping) or not isinstance(condition.get("target"), str):
            continue
        template = condition["target"]
        # Render independently for every permitted slot value.  This keeps
        # spans tied to formatter fields and avoids substring searching.
        for person in people:
            for time in times:
                try:
                    rendered_parts: list[str] = []
                    spans: dict[str, tuple[int, int]] = {}
                    position = 0
                    for literal, field_name, format_spec, conversion in string.Formatter().parse(template):
                        rendered_parts.append(literal)
                        position += len(literal.encode("utf-8"))
                        if field_name is None:
                            continue
                        if field_name not in ("person", "time") or format_spec or conversion is not None:
                            raise ValueError
                        value = person if field_name == "person" else time
                        if field_name in spans:
                            raise ValueError
                        encoded = value.encode("utf-8")
                        spans[field_name] = (position, position + len(encoded))
                        rendered_parts.append(value)
                        position += len(encoded)
                    if set(spans) != {"person", "time"}:
                        continue
                    rendered = "".join(rendered_parts)
                except (ValueError, TypeError):
                    continue
                if rendered != text or not _concept_matches(condition, concept, person, time):
                    continue
                candidates.append({
                    "participant": {"start": spans["person"][0], "end": spans["person"][1]},
                    "time": {"start": spans["time"][0], "end": spans["time"][1]},
                })
    if len(candidates) != 1:
        return None
    return candidates[0]


__all__ = ["authored_slot_spans"]
