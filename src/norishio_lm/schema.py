from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Literal

from .validation import SchemaError, array, keys, loads, object_value, string, strings

SCHEMA_VERSION = "1.0"
LAYERS = ("surface", "tokens", "morphemes", "characters", "subcharacters",
          "etymology_notes", "senses", "sememes", "concepts", "relations")
Origin = Literal["authored_demo", "sourced", "inferred", "unknown"]


@dataclass(frozen=True)
class Provenance:
    """Declared origin, not independently verified authority or confidence."""

    kind: Origin = "unknown"
    source: str | None = None
    revision: str | None = None

    @classmethod
    def from_dict(cls, value: Any, path: str = "$") -> Provenance:
        raw = object_value(value, path)
        keys(raw, {"kind", "source", "revision"}, {"kind"}, path)
        kind = raw["kind"]
        if kind not in ("authored_demo", "sourced", "inferred", "unknown"):
            raise SchemaError(f"{path}.kind", "unsupported provenance kind")
        for key in ("source", "revision"):
            if raw.get(key) is not None:
                string(raw[key], f"{path}.{key}")
        if kind == "sourced" and raw.get("source") is None:
            raise SchemaError(f"{path}.source", "sourced information requires a source")
        if raw.get("revision") is not None and raw.get("source") is None:
            raise SchemaError(f"{path}.revision", "revision requires a source")
        return cls(kind=kind, source=raw.get("source"), revision=raw.get("revision"))


def _origins(value: Any, names: Iterable[str], path: str) -> dict[str, tuple[Provenance, ...]]:
    raw = object_value(value, path)
    names = tuple(names)
    keys(raw, set(names), set(), path)
    result = {}
    for name in names:
        entries = array(raw.get(name, [{"kind": "unknown"}]), f"{path}.{name}")
        if not entries:
            raise SchemaError(f"{path}.{name}", "use unknown instead of an empty provenance list")
        result[name] = tuple(Provenance.from_dict(item, f"{path}.{name}[{i}]")
                             for i, item in enumerate(entries))
    return result


@dataclass(frozen=True)
class LexicalSense:
    """A modern lexical sense, deliberately separate from etymology."""

    sense_id: str
    gloss: str
    sememes: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()
    usage: Literal["unspecified", "experimental_poetic"] = "unspecified"
    provenance: dict[str, tuple[Provenance, ...]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any, path: str = "$") -> LexicalSense:
        raw = object_value(value, path)
        keys(raw, {"sense_id", "gloss", "sememes", "concepts", "usage", "provenance"},
             {"sense_id", "gloss"}, path)
        usage = raw.get("usage", "unspecified")
        if usage not in ("unspecified", "experimental_poetic"):
            raise SchemaError(f"{path}.usage", "unsupported usage category")
        return cls(
            sense_id=string(raw["sense_id"], f"{path}.sense_id"),
            gloss=string(raw["gloss"], f"{path}.gloss"),
            sememes=strings(raw.get("sememes", []), f"{path}.sememes"),
            concepts=strings(raw.get("concepts", []), f"{path}.concepts"),
            usage=usage,
            provenance=_origins(raw.get("provenance", {}), ("sense", "sememes", "concepts"),
                                f"{path}.provenance"),
        )


@dataclass(frozen=True)
class SemanticRecord:
    """One expression represented across explicit linguistic layers."""

    surface: str
    tokens: tuple[str, ...] = ()
    morphemes: tuple[str, ...] = ()
    characters: tuple[str, ...] = ()
    subcharacters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    etymology_notes: dict[str, str] = field(default_factory=dict)
    senses: tuple[LexicalSense, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    schema_version: str = SCHEMA_VERSION
    provenance: dict[str, tuple[Provenance, ...]] = field(default_factory=dict)
    selected_sense_id: str | None = None
    context: str | None = None
    span: tuple[int, int] | None = None
    excluded_layers: tuple[str, ...] = ()

    def layer_names(self) -> tuple[str, ...]:
        return LAYERS

    @classmethod
    def from_dict(cls, value: Any, path: str = "$") -> SemanticRecord:
        raw = object_value(value, path)
        keys(raw, set(cls.__dataclass_fields__), {"schema_version", "surface"}, path)
        if raw["schema_version"] != SCHEMA_VERSION:
            raise SchemaError(f"{path}.schema_version", f"expected {SCHEMA_VERSION}")
        surface = string(raw["surface"], f"{path}.surface", empty=True)
        tuples = {name: strings(raw.get(name, []), f"{path}.{name}", empty=True)
                  for name in ("tokens", "morphemes", "characters")}
        subcharacters = {key: strings(val, f"{path}.subcharacters[{key!r}]") for key, val in
                         object_value(raw.get("subcharacters", {}), f"{path}.subcharacters").items()}
        etymology = {key: string(val, f"{path}.etymology_notes[{key!r}]") for key, val in
                    object_value(raw.get("etymology_notes", {}), f"{path}.etymology_notes").items()}
        senses = tuple(LexicalSense.from_dict(item, f"{path}.senses[{i}]") for i, item in
                       enumerate(array(raw.get("senses", []), f"{path}.senses")))
        seen = set()
        for i, sense in enumerate(senses):
            if sense.sense_id in seen:
                raise SchemaError(f"{path}.senses[{i}].sense_id", "duplicate sense ID in expression")
            seen.add(sense.sense_id)
        relations = []
        for i, item in enumerate(array(raw.get("relations", []), f"{path}.relations")):
            relation = strings(item, f"{path}.relations[{i}]")
            if len(relation) != 3:
                raise SchemaError(f"{path}.relations[{i}]", "expected subject, predicate, object")
            relations.append((relation[0], relation[1], relation[2]))
        selected = raw.get("selected_sense_id")
        if selected is not None:
            string(selected, f"{path}.selected_sense_id")
            if selected not in seen:
                raise SchemaError(f"{path}.selected_sense_id", "not a candidate sense ID")
        context = raw.get("context")
        if context is not None:
            string(context, f"{path}.context", empty=True)
        span = raw.get("span")
        if span is not None:
            array(span, f"{path}.span")
            if (len(span) != 2 or any(type(n) is not int for n in span) or context is None
                    or not 0 <= span[0] <= span[1] <= len(context)):
                raise SchemaError(f"{path}.span", "expected [start, end] within context")
            if context[span[0]:span[1]] != surface:
                raise SchemaError(f"{path}.span", "context slice must equal surface")
        excluded = strings(raw.get("excluded_layers", []), f"{path}.excluded_layers")
        if len(set(excluded)) != len(excluded) or set(excluded) - (set(LAYERS) - {"surface"}):
            raise SchemaError(f"{path}.excluded_layers", "expected unique auxiliary layer names")
        provenance = _origins(raw.get("provenance", {}), LAYERS, f"{path}.provenance")
        record = cls(surface=surface, **tuples, subcharacters=subcharacters,
                     etymology_notes=etymology, senses=senses, relations=tuple(relations),
                     provenance=provenance, selected_sense_id=selected, context=context,
                     span=tuple(span) if span is not None else None, excluded_layers=excluded)
        for layer in excluded:
            present = (any(getattr(sense, layer) for sense in senses)
                       if layer in ("sememes", "concepts") else bool(getattr(record, layer)))
            if present:
                raise SchemaError(f"{path}.{layer}", "excluded layer must be empty")
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON tree; validate even manually constructed records."""
        raw = json.loads(json.dumps(asdict(self), ensure_ascii=False))
        normalized = type(self).from_dict(raw)
        return json.loads(json.dumps(asdict(normalized), ensure_ascii=False))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SemanticRecord:
        return cls.from_dict(loads(text))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> SemanticRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def without_layers(self, layers: Iterable[str]) -> SemanticRecord:
        """Remove auxiliary channels on a detached copy, preserving surface."""
        if isinstance(layers, str):
            raise SchemaError("$.excluded_layers", "pass layer names as an iterable, not a string")
        names = strings(list(layers), "$.excluded_layers")
        removed = set(names) | set(self.excluded_layers)
        if removed - (set(LAYERS) - {"surface"}):
            raise SchemaError("$.excluded_layers", "cannot remove surface or an unknown layer")
        if "senses" in removed:
            removed.update(("sememes", "concepts"))
        raw = self.to_dict()
        for name in removed:
            raw["provenance"][name] = [{"kind": "unknown"}]
            if name in ("sememes", "concepts"):
                for sense in raw["senses"]:
                    sense[name] = []
                    sense["provenance"][name] = [{"kind": "unknown"}]
            else:
                raw[name] = {} if name in ("subcharacters", "etymology_notes") else []
        if "senses" in removed:
            raw["selected_sense_id"] = None
        raw["excluded_layers"] = [name for name in LAYERS if name in removed]
        return type(self).from_dict(raw)
