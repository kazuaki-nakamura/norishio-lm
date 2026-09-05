"""Small, deterministic primitives for publishing complete knowledge snapshots."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


POINTER_SCHEMA_VERSION = 1
RUN_MANIFEST_SCHEMA_VERSION = 1


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace a JSON file only after its complete temporary file is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_required_files(root: Path, required: Iterable[str]) -> dict[str, Any]:
    """Validate a staged run without reading document bodies."""
    required_paths = list(required)
    missing = [rel for rel in required_paths if not (root / rel).is_file()]
    file_count = sum(1 for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0
    return {
        "valid": not missing,
        "required": required_paths,
        "missing": missing,
        "file_count": file_count,
    }


def publish_directory(staging: Path, destination: Path) -> None:
    """Atomically expose a completed directory on the same volume."""
    if not staging.is_dir():
        raise FileNotFoundError(f"staging directory does not exist: {staging}")
    if destination.exists():
        raise FileExistsError(f"publish destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(destination)


def latest_complete_pointer(
    *,
    source_type: str,
    run_id: str,
    run_dir: Path,
    manifest_path: Path,
    completed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": POINTER_SCHEMA_VERSION,
        "source_type": source_type,
        "run_id": run_id,
        "completed_at": completed_at or datetime.now().astimezone().isoformat(),
        "run_dir": run_dir.as_posix(),
        "manifest": manifest_path.as_posix(),
        "snapshot_complete": True,
    }
