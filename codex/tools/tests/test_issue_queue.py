import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from issue_queue import select_ready


class IssueQueueTests(unittest.TestCase):
    def test_bot_and_human_are_equally_eligible_and_sorted(self):
        self.assertEqual(select_ready([
            {"number": 8, "state": "open", "labels": [{"name": "ai-ready"}], "user": {"type": "Bot"}},
            {"number": 2, "state": "open", "labels": ["ai-ready"]},
        ]), [2, 8])

    def test_closed_pull_requests_and_claimed_issues_are_excluded(self):
        rows = [
            {"number": 1, "state": "closed", "labels": ["ai-ready"]},
            {"number": 2, "state": "open", "labels": ["ai-ready"], "pull_request": {}},
            {"number": 3, "state": "open", "labels": ["ai-ready", "ai-in-progress"]},
            {"number": 4, "state": "open", "labels": ["ai-ready", "ai-review"]},
            {"number": 5, "state": "open", "labels": []},
        ]
        self.assertEqual(select_ready(rows), [])

    def test_blocked_requires_explicit_ready_label_and_duplicates_are_removed(self):
        issue = {"number": 9, "state": "open", "labels": ["ai-blocked", "ai-ready"]}
        self.assertEqual(select_ready([issue, issue]), [9])
        self.assertEqual(select_ready([{**issue, "labels": ["ai-blocked"]}]), [])
