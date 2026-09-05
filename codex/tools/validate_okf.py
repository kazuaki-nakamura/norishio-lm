"""Lightweight offline validator for this project's OKF v0.2 bundle."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from okf_event_log import load_events


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OKF_ROOT = PROJECT_ROOT / "okf"
RESERVED = {"index.md", "log.md"}
STATUS_VALUES = {"draft", "stable", "deprecated"}
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?:^|[\s\"'`(])(?:[a-z]:\\)")
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def split_frontmatter(text: str) -> tuple[str, str] | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    return text[4:end], text[end + 5 :]


def resolve_link(source: Path, target: str) -> Path | None:
    clean = target.split("#", 1)[0].strip()
    if not clean or "://" in clean or clean.startswith("mailto:"):
        return None
    if clean.startswith("/"):
        return OKF_ROOT / clean.lstrip("/")
    return source.parent / clean


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not OKF_ROOT.is_dir():
        print(f"ERROR: OKF directory not found: {OKF_ROOT}")
        return 1

    files = sorted(OKF_ROOT.rglob("*.md"))
    concepts = [path for path in files if path.name not in RESERVED]

    event_path = OKF_ROOT / "events.jsonl"
    if event_path.exists():
        _, event_errors = load_events(event_path)
        errors.extend(f"events.jsonl: {message}" for message in event_errors)

    root_index = OKF_ROOT / "index.md"
    if not root_index.exists():
        errors.append("index.md: bundle-root index is missing")
    else:
        index_text = root_index.read_text(encoding="utf-8")
        if 'okf_version: "0.2"' not in index_text:
            errors.append('index.md: okf_version must be "0.2"')

    for path in concepts:
        rel = path.relative_to(OKF_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        frontmatter = split_frontmatter(text)
        if frontmatter is None:
            errors.append(f"{rel}: missing or unterminated YAML frontmatter")
            continue

        yaml_text, body = frontmatter
        if "\t" in yaml_text:
            errors.append(f"{rel}: tabs are not allowed in YAML frontmatter")

        type_match = re.search(r"(?m)^type:\s*(.+?)\s*$", yaml_text)
        if not type_match or not type_match.group(1).strip():
            errors.append(f"{rel}: non-empty type field is required")

        status_match = re.search(r"(?m)^status:\s*(\S+)\s*$", yaml_text)
        if status_match and status_match.group(1) not in STATUS_VALUES:
            errors.append(
                f"{rel}: status must be one of {sorted(STATUS_VALUES)}, "
                f"got {status_match.group(1)!r}"
            )

        if "generated:" in yaml_text and not re.search(
            r"generated:\s*\{[^}]*\bby:\s*[^,}]+", yaml_text
        ):
            errors.append(f"{rel}: generated.by is required when generated is present")

        for source_block in re.finditer(
            r"(?ms)^sources:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)", yaml_text
        ):
            entries = re.split(r"(?m)^\s*-\s+", source_block.group("body"))
            for entry in entries[1:]:
                if not re.search(r"(?m)^\s*resource:\s*\S+", entry):
                    errors.append(f"{rel}: every sources entry requires resource")

        if WINDOWS_ABSOLUTE.search(text):
            errors.append(f"{rel}: contains a Windows absolute path; use portable paths")

        for target in MARKDOWN_LINK.findall(body):
            resolved = resolve_link(path, target)
            if resolved is not None and not resolved.exists():
                warnings.append(f"{rel}: unresolved link {target}")

    print(
        f"OKF files={len(files)}, concepts={len(concepts)}, "
        f"errors={len(errors)}, warnings={len(warnings)}"
    )
    for message in errors:
        print(f"ERROR: {message}")
    for message in warnings:
        print(f"WARNING: {message}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
