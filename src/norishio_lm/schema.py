from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LexicalSense:
    """A modern lexical sense, deliberately separate from etymology."""

    sense_id: str
    gloss: str
    sememes: tuple[str, ...] = ()
    concepts: tuple[str, ...] = ()


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

    def layer_names(self) -> tuple[str, ...]:
        return (
            "surface",
            "tokens",
            "morphemes",
            "characters",
            "subcharacters",
            "etymology_notes",
            "senses",
            "relations",
        )
