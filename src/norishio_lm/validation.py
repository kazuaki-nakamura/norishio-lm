"""Strict JSON boundary validation, without network or coercion."""
from __future__ import annotations

import json
from typing import Any


class SchemaError(ValueError):
    """Invalid input with a JSON-style location."""

    def __init__(self, path: str, reason: str) -> None:
        self.path, self.reason = path, reason
        super().__init__(f"{path}: {reason}")


def object_value(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise SchemaError(path, "expected object with string keys")
    return value


def keys(value: dict[str, Any], allowed: set[str], required: set[str], path: str) -> None:
    for key in sorted(required - value.keys()):
        raise SchemaError(f"{path}.{key}", "required key missing")
    for key in sorted(value.keys() - allowed):
        raise SchemaError(f"{path}.{key}", "unknown key")


def string(value: Any, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise SchemaError(path, "expected string" if empty else "expected nonempty string")
    return value


def array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(path, "expected array")
    return value


def strings(value: Any, path: str, *, empty: bool = False) -> tuple[str, ...]:
    return tuple(string(item, f"{path}[{i}]", empty=empty)
                 for i, item in enumerate(array(value, path)))


class _Pairs(list):
    """Distinguish JSON objects from arrays until duplicate keys are checked."""


def loads(text: str) -> Any:
    def invalid_constant(value: str) -> None:
        raise SchemaError("$", f"non-finite JSON number: {value}")

    def unpack(value: Any, path: str) -> Any:
        if isinstance(value, _Pairs):
            result = {}
            for key, child in value:
                location = f"{path}[{json.dumps(key, ensure_ascii=False)}]"
                if key in result:
                    raise SchemaError(location, "duplicate JSON key")
                result[key] = unpack(child, location)
            return result
        if isinstance(value, list):
            return [unpack(child, f"{path}[{i}]") for i, child in enumerate(value)]
        return value

    try:
        return unpack(json.loads(text, object_pairs_hook=_Pairs,
                                 parse_constant=invalid_constant), "$")
    except json.JSONDecodeError as exc:
        raise SchemaError(f"$ (line {exc.lineno}, column {exc.colno})", exc.msg) from exc
