"""Build a portable metadata-only source inventory from explicit roots."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge_publish import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('codex/work_output/context')


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes project root')
    return path


def source_files(root: Path) -> list[Path]:
    config = json.loads((root / 'foundation.json').read_text(encoding='utf-8'))
    directories = config['source_roots']
    if not isinstance(directories, list) or not directories:
        raise ValueError('source_roots must be a nonempty list')
    files = set()
    excluded_parts = set(config.get('exclude_parts', []))
    excluded_suffixes = set(config.get('exclude_suffixes', []))
    for relative in directories:
        directory = inside(root, relative)
        if not directory.is_dir():
            raise ValueError('Configured source root is missing')
        for path in directory.rglob('*'):
            if excluded_parts.intersection(path.relative_to(root).parts) or path.suffix in excluded_suffixes:
                continue
            if path.is_file():
                resolved = inside(root, str(path.relative_to(root)))
                files.add(resolved)
    for relative in config.get('source_files', []):
        path = inside(root, relative)
        if not path.is_file():
            raise ValueError('Configured source file is missing')
        files.add(path)
    return sorted(files)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(root: Path) -> dict:
    root = root.resolve()
    entries = []
    for path in source_files(root):
        stat = path.stat()
        entries.append({'rel_path': path.relative_to(root).as_posix(),
                        'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                        'sha256': digest(path)})
    manifest = {'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
                'entry_count': len(entries), 'entries': entries}
    destination = root / OUTPUT
    atomic_write_json(destination / 'manifest.json', manifest)
    (destination / 'source_list.md').write_text(
        '# Source inventory\n\n' + '\n'.join('- ' + e['rel_path'] for e in entries) + '\n', encoding='utf-8')
    (destination / 'session_context.md').write_text(
        '# Context entry\n\nRead okf/index.md, retrieve selected concepts, then verify original sources.\n'
        + f'Indexed files: {len(entries)}. This inventory is a cache, not evidence.\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(json.dumps({'entry_count': build(ROOT)['entry_count']}))
