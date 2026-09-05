from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from okf_event_log import append_event, load_events, make_event_id, render_markdown  # noqa: E402


EVENT = {
    "occurred_on": "2026-08-18",
    "title": "Deterministic update",
    "summary": "Verified one concept against an original source.",
    "concept_paths": ["okf/project/overview.md"],
    "source_paths": ["documents/source.md"],
    "verification": "original-source",
}


class OkfEventLogTests(unittest.TestCase):
    def test_event_id_is_stable_and_duplicate_append_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.jsonl"
            self.assertTrue(append_event(path, EVENT))
            self.assertFalse(append_event(path, EVENT))
            events, errors = load_events(path)
            self.assertEqual(errors, [])
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_id"], make_event_id(EVENT))

    def test_markdown_order_is_deterministic(self) -> None:
        first = dict(EVENT, occurred_on="2026-08-17")
        first["event_id"] = make_event_id(first)
        second = dict(EVENT, occurred_on="2026-08-18")
        second["event_id"] = make_event_id(second)
        self.assertEqual(render_markdown([first, second]), render_markdown([second, first]))


if __name__ == "__main__":
    unittest.main()
