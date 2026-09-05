#!/usr/bin/env python3
"""Structured OKF update events with deterministic IDs and safe previews."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = PROJECT_ROOT / "okf" / "events.jsonl"
DEFAULT_PREVIEW = PROJECT_ROOT / "codex" / "work_output" / "okf-log-preview.md"
SCHEMA_VERSION = 1


def canonical_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "occurred_on": str(event.get("occurred_on", "")),
        "title": str(event.get("title", "")).strip(),
        "summary": str(event.get("summary", "")).strip(),
        "concept_paths": sorted(str(item) for item in event.get("concept_paths", [])),
        "source_paths": sorted(str(item) for item in event.get("source_paths", [])),
        "verification": str(event.get("verification", "original-source")),
    }


def make_event_id(event: dict[str, Any]) -> str:
    raw = json.dumps(canonical_payload(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "okfevt-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def validate_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    payload = canonical_payload(event)
    try:
        date.fromisoformat(payload["occurred_on"])
    except ValueError:
        errors.append("occurred_on must be YYYY-MM-DD")
    for key in ("title", "summary"):
        if not payload[key]:
            errors.append(f"{key} is required")
    for key in ("concept_paths", "source_paths"):
        for path in payload[key]:
            if Path(path).is_absolute() or ":\\" in path:
                errors.append(f"{key} must contain project-relative paths")
    expected = make_event_id(payload)
    if event.get("event_id") not in (None, expected):
        errors.append("event_id does not match canonical event fields")
    return errors


def load_events(path: Path = DEFAULT_EVENTS) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
            continue
        if not isinstance(event, dict):
            errors.append(f"line {line_number}: event must be an object")
            continue
        event_errors = validate_event(event)
        if event_errors:
            errors.extend(f"line {line_number}: {message}" for message in event_errors)
            continue
        event_id = str(event.get("event_id") or make_event_id(event))
        if event_id in seen:
            errors.append(f"line {line_number}: duplicate event_id {event_id}")
            continue
        event["event_id"] = event_id
        seen.add(event_id)
        events.append(event)
    return events, errors


def write_events_atomic(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    text = "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_event(path: Path, event: dict[str, Any]) -> bool:
    errors = validate_event(event)
    if errors:
        raise ValueError("; ".join(errors))
    events, load_errors = load_events(path)
    if load_errors:
        raise ValueError("existing event log is invalid: " + "; ".join(load_errors))
    normalized = canonical_payload(event)
    normalized["event_id"] = make_event_id(normalized)
    if any(item["event_id"] == normalized["event_id"] for item in events):
        return False
    events.append(normalized)
    events.sort(key=lambda item: (item["occurred_on"], item["event_id"]))
    write_events_atomic(path, events)
    return True


def render_markdown(events: list[dict[str, Any]]) -> str:
    rows = ["# OKF Update Log (structured preview)", ""]
    current_date = None
    for event in sorted(events, key=lambda item: (item["occurred_on"], item["event_id"]), reverse=True):
        if event["occurred_on"] != current_date:
            current_date = event["occurred_on"]
            rows.extend([f"## {current_date}", ""])
        paths = ", ".join(f"`{path}`" for path in event.get("concept_paths", []))
        suffix = f" 対象: {paths}" if paths else ""
        rows.append(f"- **{event['title']}**: {event['summary']}{suffix} <!-- {event['event_id']} -->")
        rows.append("")
    return "\n".join(rows).rstrip() + "\n"


def legacy_duplicate_report(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    bullets = [line.strip() for line in lines if line.startswith(("- ", "* "))]
    counts: dict[str, int] = {}
    for bullet in bullets:
        counts[bullet] = counts.get(bullet, 0) + 1
    duplicates = [{"count": count, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()} for text, count in counts.items() if count > 1]
    return {"bullet_count": len(bullets), "duplicate_groups": len(duplicates), "duplicates": duplicates}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "render", "scan-legacy"))
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--legacy-log", type=Path, default=PROJECT_ROOT / "okf" / "log.md")
    args = parser.parse_args()
    if args.command == "scan-legacy":
        print(json.dumps(legacy_duplicate_report(args.legacy_log), ensure_ascii=False, indent=2))
        return 0
    events, errors = load_events(args.events)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1
    if args.command == "render":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_markdown(events), encoding="utf-8")
    print(json.dumps({"valid": True, "event_count": len(events)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
