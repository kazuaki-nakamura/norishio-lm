"""Source-only adapter for the Issue 3 authored toy corpus.

The adapter deliberately has no access to corpus rows or their annotations.  Its
input is the allowlisted ``{"context": str, "text": str}`` source object.  The
canonical JSON source is retained as ``surface`` and its observable Unicode
characters are exposed as ``tokens``.  Token characters are surface features;
they are not morphology or character/glyph analysis.
"""
from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

from .schema import Provenance, SemanticRecord

_SOURCE_KEYS = frozenset(("context", "text"))

# This fixture is intentionally nonsensical and is available only through the
# explicit diagnostic API below.  It must never be inferred from input metadata.
MISLEADING_GLYPH_FIXTURE: dict[str, tuple[str, ...]] = {"性": ("金", "口")}

_SOURCE_PROVENANCE = Provenance(
    kind="authored_demo", source="data/issue3/seed.json", revision="1.0"
)
_FIXTURE_PROVENANCE = Provenance(
    kind="authored_demo",
    source="data/issue3/seed.json#diagnostic:glyph_adversarial:intentionally_false_fixture",
    revision="1.0",
)


def canonical_source(source: dict[str, str]) -> str:
    """Return the deterministic JSON representation used as the source surface."""
    _validate_source(source)
    return json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_source(source: Any) -> None:
    # A nested row, subclass, or extra key is a boundary violation.
    if type(source) is not dict or set(source) != _SOURCE_KEYS:
        raise ValueError("source must be exactly a dict containing context and text")
    if any(type(value) is not str for value in source.values()):
        raise ValueError("source context and text must be strings")


def source_record(source: dict[str, str]) -> SemanticRecord:
    """Adapt one source object without reading targets, IDs, split, or metadata."""
    surface = canonical_source(source)
    return SemanticRecord(
        surface=surface,
        tokens=tuple(surface),
        characters=tuple(surface),
        provenance={
            "surface": (_SOURCE_PROVENANCE,),
            "tokens": (_SOURCE_PROVENANCE,),
            "characters": (_SOURCE_PROVENANCE,),
        },
    )


class ToySourceAdapter:
    """Small stateless source adapter; vocabulary fitting belongs to the caller."""

    def adapt(self, source: dict[str, str]) -> SemanticRecord:
        return source_record(source)

    __call__ = adapt


def misleading_glyph_fixture_record(source: dict[str, str]) -> SemanticRecord:
    """Return a diagnostic-only record with the explicit misleading glyph cue.

    This function is intentionally separate from :func:`source_record`: normal
    adaptation cannot receive metadata or fixture overrides.  The fixture only
    changes only the subcharacter channel and never creates modern semantic labels.
    """
    record = source_record(source)
    return replace(
        record,
        subcharacters=dict(MISLEADING_GLYPH_FIXTURE),
        provenance={**record.provenance, "subcharacters": (_FIXTURE_PROVENANCE,)},
    )


__all__ = [
    "MISLEADING_GLYPH_FIXTURE",
    "ToySourceAdapter",
    "canonical_source",
    "misleading_glyph_fixture_record",
    "source_record",
]
