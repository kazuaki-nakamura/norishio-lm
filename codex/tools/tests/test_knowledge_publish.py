from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from knowledge_publish import (  # noqa: E402
    atomic_write_json,
    latest_complete_pointer,
    publish_directory,
    validate_required_files,
)


class KnowledgePublishTests(unittest.TestCase):
    def test_atomic_write_json_replaces_complete_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "state.json"
            atomic_write_json(target, {"value": 1})
            atomic_write_json(target, {"value": 2})
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_required_file_validation_reports_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "present.json").write_text("{}", encoding="utf-8")
            result = validate_required_files(root, ["present.json", "missing.json"])
            self.assertFalse(result["valid"])
            self.assertEqual(result["missing"], ["missing.json"])

    def test_publish_directory_never_overwrites_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            destination = root / "published"
            staging.mkdir()
            destination.mkdir()
            with self.assertRaises(FileExistsError):
                publish_directory(staging, destination)
            self.assertTrue(staging.exists())

    def test_latest_pointer_marks_only_complete_snapshots(self) -> None:
        pointer = latest_complete_pointer(
            source_type="example",
            run_id="run-1",
            run_dir=Path("imports/example/run-1"),
            manifest_path=Path("imports/example/run-1/manifest.json"),
            completed_at="2026-08-18T10:00:00+09:00",
        )
        self.assertTrue(pointer["snapshot_complete"])
        self.assertEqual(pointer["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
