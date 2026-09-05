import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_context import build, source_files
from check_knowledge import check


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for path, text in {
            'foundation.json': json.dumps({'source_roots': ['documents']}),
            'AGENTS.md': '# Rules', 'okf/index.md': '# Index',
            'codex/okf_mcp/server.py': '# placeholder',
            'documents/source.md': 'first',
        }.items():
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding='utf-8')
        build(self.root)

    def test_healthy_and_added_source(self):
        self.assertEqual(check(self.root, True)['status'], 'healthy')
        (self.root / 'documents/added.md').write_text('new')
        self.assertEqual(check(self.root)['status'], 'invalid')

    def test_same_size_same_time_change_detected_by_hash(self):
        path = self.root / 'documents/source.md'
        stat = path.stat()
        path.write_text('other', encoding='utf-8')
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(check(self.root, True)['status'], 'invalid')

    def test_missing_source_and_broken_reference(self):
        (self.root / 'documents/source.md').unlink()
        self.assertEqual(check(self.root)['status'], 'invalid')
        build(self.root)
        (self.root / 'okf/bad.md').write_text('sources:\n  - id: x\n    resource: documents/missing.md\n')
        self.assertEqual(check(self.root)['status'], 'invalid')

    def test_path_escape_rejected(self):
        (self.root / 'foundation.json').write_text(json.dumps({'source_roots': ['../']}))
        with self.assertRaises(ValueError):
            source_files(self.root)

    def test_explicit_files_and_generated_cache_exclusion(self):
        (self.root / 'foundation.json').write_text(json.dumps({
            'source_roots': ['documents'], 'source_files': ['AGENTS.md'],
            'exclude_parts': ['__pycache__'], 'exclude_suffixes': ['.pyc'],
        }))
        cache = self.root / 'documents/__pycache__'
        cache.mkdir()
        (cache / 'module.pyc').write_bytes(b'cache')
        (self.root / 'documents/other.pyc').write_bytes(b'cache')
        paths = {p.relative_to(self.root).as_posix() for p in source_files(self.root)}
        self.assertEqual(paths, {'documents/source.md', 'AGENTS.md'})
        build(self.root)
        (cache / 'new.pyc').write_bytes(b'new')
        self.assertEqual(check(self.root, True)['status'], 'healthy')
        (self.root / 'AGENTS.md').write_text('changed rules')
        self.assertEqual(check(self.root, True)['status'], 'invalid')

    def test_explicit_file_missing_or_outside_root_rejected(self):
        for relative in ['missing.md', '../outside.md']:
            with self.subTest(relative=relative):
                (self.root / 'foundation.json').write_text(json.dumps({
                    'source_roots': ['documents'], 'source_files': [relative],
                }))
                with self.assertRaises(ValueError):
                    source_files(self.root)


if __name__ == '__main__':
    unittest.main()
