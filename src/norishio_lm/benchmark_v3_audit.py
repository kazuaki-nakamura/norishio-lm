"""Post-freeze audits that must not change the benchmark-v3 generator hash."""
from __future__ import annotations

import json
from pathlib import Path
import runpy
from typing import Any, Mapping, Sequence


_V2_GENERATOR = (
    Path(__file__).resolve().parents[2] / "data" / "benchmark_v2" / "benchmark.py"
)
_EVALUATION_SPLITS = ("diagnostic-validation", "final-confirmation")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_frozen_bundle(bundle: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Enforce surface isolation omitted by the hash-bound v3 generator audit.

    This lives outside ``benchmark_v3_data.py`` because that file's SHA-256 is
    part of the preregistered manifest and frozen checkpoint identity.
    """

    source_surfaces: set[str] = set()
    target_surfaces: set[str] = set()
    v3_evaluation: set[tuple[str, str, str]] = set()
    for split, rows in bundle.items():
        for row in rows:
            inputs = row["inputs"]
            source_surfaces.update(
                value for value in (inputs.get("context"), inputs.get("text"))
                if isinstance(value, str) and value
            )
            target = row["targets"]["text"]
            if target in target_surfaces:
                raise ValueError("duplicate target text")
            target_surfaces.add(target)
            if split in _EVALUATION_SPLITS:
                v3_evaluation.add((inputs["text"], target, _canonical(row["targets"]["frame"])))
    if source_surfaces & target_surfaces:
        raise ValueError("source and target text sets overlap")

    namespace = runpy.run_path(str(_V2_GENERATOR))
    v2_bundle = namespace["build"]()
    v2_evaluation = {
        (
            row["inputs"]["text"],
            row["targets"]["text"],
            _canonical(row["targets"]["frame"]),
        )
        for split in ("diagnostic-validation", "final-holdout")
        for row in v2_bundle[split]
    }
    if v3_evaluation & v2_evaluation:
        raise ValueError("v3 evaluation surface records reuse benchmark-v2 records")


__all__ = ["audit_frozen_bundle"]
