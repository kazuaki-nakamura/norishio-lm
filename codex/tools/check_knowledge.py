"""Fail closed on missing or changed sources and broken local knowledge links."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote

from build_context import ROOT, OUTPUT, digest, inside, source_files


def check(root: Path, full: bool = False) -> dict:
    errors = []
    checked = 0
    try:
        root = root.resolve()
        for relative in ('AGENTS.md', 'okf/index.md', 'codex/okf_mcp/server.py'):
            if not (root / relative).is_file():
                errors.append('missing: ' + relative)
        manifest = json.loads((root / OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
        entries = manifest['entries']
        if manifest['schema_version'] != 1 or manifest['entry_count'] != len(entries):
            errors.append('invalid manifest metadata')
        recorded = [entry['rel_path'] for entry in entries]
        current = {p.relative_to(root).as_posix() for p in source_files(root)}
        if len(set(recorded)) != len(recorded) or set(recorded) != current:
            errors.append('source inventory differs; regenerate context')
        for entry in entries:
            path = inside(root, entry['rel_path'])
            checked += 1
            if not path.is_file():
                errors.append('missing source: ' + entry['rel_path'])
                continue
            stat = path.stat()
            if stat.st_size != entry['size_bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
                errors.append('changed source: ' + entry['rel_path'])
            if full and digest(path) != entry['sha256']:
                errors.append('hash mismatch: ' + entry['rel_path'])
        for concept in (root / 'okf').rglob('*.md'):
            content = concept.read_text(encoding='utf-8')
            for resource in re.findall(r'^\s+resource:\s*(.+)$', content, re.M):
                resource = resource.strip().strip('\"\'')
                if '://' not in resource and not inside(root, resource).is_file():
                    errors.append('missing source reference: ' + resource)
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
                target = unquote(target.split('#')[0])
                if not target or '://' in target or target.startswith('mailto:'):
                    continue
                path = root / 'okf' / target.lstrip('/') if target.startswith('/') else concept.parent / target
                if not path.resolve().is_relative_to((root / 'okf').resolve()) or not path.exists():
                    errors.append('invalid knowledge link: ' + target)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(type(exc).__name__ + ': invalid or unavailable knowledge input')
    return {'status': 'invalid' if errors else 'healthy', 'checked_sources': checked,
            'mode': 'full' if full else 'quick', 'errors': errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    args = parser.parse_args()
    result = check(ROOT, args.mode == 'full')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result['errors'] else 0)
