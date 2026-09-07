from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .schema import SCHEMA_VERSION, SemanticRecord
from .validation import SchemaError, keys, loads, object_value, string


class SemanticCompiler:
    """Validated exact lookup, not a learned tokenizer or contextual selector."""

    def __init__(self, lexicon: dict[str, Any] | None = None) -> None:
        raw = object_value({} if lexicon is None else lexicon, "$")
        prefix = "$"
        if "schema_version" in raw:
            keys(raw, {"schema_version", "entries"}, {"schema_version", "entries"}, "$")
            if raw["schema_version"] != SCHEMA_VERSION:
                raise SchemaError("$.schema_version", f"expected {SCHEMA_VERSION}")
            raw = object_value(raw["entries"], "$.entries")
            prefix = "$.entries"
        self._records: dict[str, SemanticRecord] = {}
        for surface, entry in raw.items():
            path = f"{prefix}[{surface!r}]"
            entry = object_value(entry, path)
            keys(entry, set(SemanticRecord.__dataclass_fields__) -
                 {"surface", "schema_version", "context", "span", "selected_sense_id", "excluded_layers"},
                 set(), path)
            data = {"schema_version": SCHEMA_VERSION, "surface": surface,
                    "tokens": [surface], "characters": list(surface), **entry}
            self._records[surface] = SemanticRecord.from_dict(data, path)

    @classmethod
    def from_json(cls, path: str | Path) -> SemanticCompiler:
        return cls(object_value(loads(Path(path).read_text(encoding="utf-8")), "$"))

    def compile(self, surface: str, *, context: str | None = None,
                span: tuple[int, int] | None = None,
                exclude_layers: Iterable[str] = ()) -> SemanticRecord:
        """Keep all candidates; context/span are validated metadata only."""
        string(surface, "$.surface", empty=True)
        if surface in self._records:
            raw = self._records[surface].to_dict()
        else:
            raw = {"schema_version": SCHEMA_VERSION, "surface": surface,
                   "tokens": [surface], "characters": list(surface)}
        if span is not None and not isinstance(span, (tuple, list)):
            raise SchemaError("$.span", "expected start/end pair")
        raw.update(context=context, span=list(span) if span is not None else None)
        return SemanticRecord.from_dict(raw).without_layers(exclude_layers)
