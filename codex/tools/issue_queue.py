"""Deterministic ai-ready selection and explicit repository label setup."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPOSITORY = "kazuaki-nakamura/norishio-lm"
LABELS = {
    "ai-ready": ("0E8A16", "Ready for the AI development worker"),
    "ai-in-progress": ("FBCA04", "Claimed by the AI development worker"),
    "ai-review": ("1D76DB", "AI pull request awaits human review"),
    "ai-blocked": ("D93F0B", "Needs clarification or intervention before retry"),
}


def select_ready(issues: list[dict[str, Any]]) -> list[int]:
    """Return oldest issue numbers; ready is also an explicit retry request."""
    selected = set()
    for issue in issues:
        labels = {item["name"] if isinstance(item, dict) else item
                  for item in issue.get("labels", [])}
        if (issue.get("state") == "open" and "pull_request" not in issue
                and "ai-ready" in labels and "ai-in-progress" not in labels
                and "ai-review" not in labels):
            selected.add(int(issue["number"]))
    return sorted(selected)


def github_token() -> str:
    """Ask the configured Git credential helper without exposing its output."""
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n", text=True,
        capture_output=True, timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"},
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise RuntimeError("GitHub credential helper authentication unavailable")
    return fields["password"]


def ensure_labels() -> dict[str, list[str]]:
    """Create missing queue labels; preserve every existing label's properties."""
    token = github_token()
    base = f"https://api.github.com/repos/{REPOSITORY}/labels"
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json", "User-Agent": "norishio-lm"}
    existing = set()
    page = 1
    while True:
        with urlopen(Request(f"{base}?per_page=100&page={page}", headers=headers), timeout=30) as response:
            rows = json.load(response)
        existing.update(row["name"] for row in rows)
        if len(rows) < 100:
            break
        page += 1
    created = []
    for name, (color, description) in LABELS.items():
        if name in existing:
            continue
        body = json.dumps({"name": name, "color": color, "description": description}).encode()
        with urlopen(Request(base, data=body, headers=headers, method="POST"), timeout=30):
            pass
        created.append(name)
    return {"created": created, "existing": sorted(existing.intersection(LABELS))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure-labels", action="store_true", help="Create missing GitHub labels")
    parser.add_argument("--input", help="Offline JSON array of issues to select")
    args = parser.parse_args()
    if args.ensure_labels == bool(args.input):
        parser.error("Choose exactly one of --ensure-labels or --input")
    try:
        if args.ensure_labels:
            output = ensure_labels()
        else:
            with open(args.input, encoding="utf-8") as stream:
                output = {"ready": select_ready(json.load(stream))}
        print(json.dumps(output))
        return 0
    except HTTPError as exc:
        print(json.dumps({"error": "GitHub API request failed", "status": exc.code}))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        print(json.dumps({"error": "Queue operation failed; check authentication and input"}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
