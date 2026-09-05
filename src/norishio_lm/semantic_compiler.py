from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import LexicalSense, SemanticRecord


class SemanticCompiler:
    """Compile surface expressions into explicit semantic layers.

    v0.1 intentionally uses a tiny declarative lexicon. Later adapters can
    replace each layer independently (morphological parser, radical DB,
    sense inventory, sememe KB, concept graph, etc.).
    """

    def __init__(self, lexicon: dict[str, Any] | None = None) -> None:
        self._lexicon = lexicon or {}

    @classmethod
    def from_json(cls, path: str | Path) -> "SemanticCompiler":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f))

    def compile(self, surface: str) -> SemanticRecord:
        raw = self._lexicon.get(surface, {})
        senses = tuple(
            LexicalSense(
                sense_id=s["sense_id"],
                gloss=s["gloss"],
                sememes=tuple(s.get("sememes", [])),
                concepts=tuple(s.get("concepts", [])),
            )
            for s in raw.get("senses", [])
        )

        return SemanticRecord(
            surface=surface,
            tokens=tuple(raw.get("tokens", [surface])),
            morphemes=tuple(raw.get("morphemes", [])),
            characters=tuple(raw.get("characters", list(surface))),
            subcharacters={
                key: tuple(value)
                for key, value in raw.get("subcharacters", {}).items()
            },
            etymology_notes=dict(raw.get("etymology_notes", {})),
            senses=senses,
            relations=tuple(tuple(r) for r in raw.get("relations", [])),
        )
