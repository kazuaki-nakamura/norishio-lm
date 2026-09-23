"""Read-only token-count collision audit for the Issue #46 head fixture.

The head encoder's masked mean is insensitive to token order.  This module
does not run a model: it computes a deterministic count signature from the
frozen fixture's :func:`source_ids`, then reports the label ambiguity and the
best possible factor-wise accuracy for each signature partition.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from . import benchmark_v3_head_fixture as fixture


FACTORS = fixture.FIELDS
SUPPORT_STRATA = fixture.CONFIRMATION_SUPPORTS
SCHEMA = "norishio.issue46.head-collision-audit.v1"


def token_count_signature(source_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return the sorted ``(token_id, count)`` signature of a source sequence.

    The caller should pass the complete sequence returned by
    ``benchmark_v3_head_fixture.source_ids``.  BOS and SEP therefore remain
    part of the signature, as do all bytes of the canonical JSON wrapper.
    """

    if isinstance(source_ids, (str, bytes, bytearray)):
        raise TypeError("source_ids must be a sequence of integer token ids")
    counts: Counter[int] = Counter()
    for token in source_ids:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError("source_ids must contain integer token ids")
        counts[token] += 1
    return tuple(sorted(counts.items()))


def _label(row: Mapping[str, Any], factor: str) -> Any:
    targets = row.get("targets")
    if not isinstance(targets, Mapping):
        raise ValueError("row targets must be a mapping")
    frame = targets.get("frame")
    if not isinstance(frame, Mapping) or factor not in frame:
        raise ValueError(f"row target frame is missing factor {factor!r}")
    return frame[factor]


def _signature(row: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return token_count_signature(fixture.source_ids(row))


def _audit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[tuple[int, int], ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each fixture row must be a mapping")
        groups[_signature(row)].append(row)

    collisions = {sig: grouped for sig, grouped in groups.items() if len(grouped) > 1}
    conflicting: dict[str, int] = {}
    conflicting_rows: dict[str, int] = {}
    ceilings: dict[str, float | None] = {}
    ceiling_detail: dict[str, dict[str, Any]] = {}
    gold_support: dict[str, dict[str, int]] = {}
    scheduled = len(rows)
    for factor in FACTORS:
        gold_support[factor] = dict(sorted(Counter(str(_label(row, factor)) for row in rows).items()))
        conflict_groups = 0
        conflict_row_count = 0
        majority_rows = 0
        for grouped in groups.values():
            labels = Counter(_label(row, factor) for row in grouped)
            majority_rows += max(labels.values(), default=0)
            if len(labels) > 1:
                conflict_groups += 1
                conflict_row_count += len(grouped)
        conflicting[factor] = conflict_groups
        conflicting_rows[factor] = conflict_row_count
        ceilings[factor] = majority_rows / scheduled if scheduled else None
        ceiling_detail[factor] = {
            "majority_label_rows": majority_rows,
            "scheduled_rows": scheduled,
        }

    return {
        "row_count": scheduled,
        "unique_signatures": len(groups),
        "collision_group_count": len(collisions),
        "collision_rows": sum(len(grouped) for grouped in collisions.values()),
        "per_factor_conflicting_gold_signature_counts": conflicting,
        "per_factor_conflicting_gold_rows": conflicting_rows,
        "per_factor_representation_ceiling": ceilings,
        "per_factor_representation_ceiling_detail": ceiling_detail,
        "per_factor_gold_support": gold_support,
    }


def audit_bundle(bundle: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Audit train, all confirmation, and each frozen confirmation stratum."""

    if not isinstance(bundle, Mapping) or set(bundle) != set(fixture.SPLITS):
        raise ValueError(f"bundle must contain exactly {fixture.SPLITS!r}")
    train = list(bundle["train"])
    confirmation = list(bundle["confirmation"])
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "signature": "sorted token-id counts from fixture.source_ids(row), including BOS/SEP and canonical JSON wrapper",
        "scopes": {
            "train": _audit_rows(train),
            "confirmation_all": _audit_rows(confirmation),
            "confirmation_by_support": {},
        },
    }
    by_support = report["scopes"]["confirmation_by_support"]
    for support in SUPPORT_STRATA:
        by_support[support] = _audit_rows([
            row for row in confirmation
            if isinstance(row, Mapping)
            and isinstance(row.get("metadata"), Mapping)
            and row["metadata"].get("support_class") == support
        ])
    return report


def audit_fixture(bundle: Mapping[str, Sequence[Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Build and audit the frozen fixture, or audit a supplied bundle."""

    frozen_bundle = fixture.build() if bundle is None else bundle
    fixture_digest = fixture.manifest_digest(frozen_bundle)
    report = audit_bundle(frozen_bundle)
    report["fixture_content_digest_sha256"] = fixture_digest
    return report


def write_json(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write a deterministic audit JSON file to an explicitly supplied path."""

    Path(path).write_bytes(
        (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="explicit path for the JSON audit")
    args = parser.parse_args(argv)
    write_json(args.output, audit_fixture())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FACTORS", "SCHEMA", "SUPPORT_STRATA", "audit_bundle", "audit_fixture", "token_count_signature", "write_json"]
