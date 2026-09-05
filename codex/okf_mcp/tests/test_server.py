import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import OkfIndex, Server

class OkfMcpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "design").mkdir()
        (root / "findings").mkdir()
        (root / "design" / "buzzer.md").write_text(
            "---\nstatus: current\ngenerated: { by: codex, at: 2026-08-06T00:00:00+09:00 }\n"
            "sources:\n  - id: req\n    resource: 要件定義書.md\nstale_after: 2099-01-01\n---\n"
            "# 資料検索\nREQ-DEMO-001。登録した資料を検索する。"
            "[課題](/findings/open.md)\n", encoding="utf-8")
        (root / "design" / "old.md").write_text(
            "---\nstatus: obsolete\n---\n# 旧キャッシュ方式\nLocMemCacheを使う。\n", encoding="utf-8")
        (root / "findings" / "open.md").write_text(
            "# 未決定事項\n検索性能試験は要確認。\n", encoding="utf-8")
        (root / "design" / "frontmatter-title.md").write_text(
            "---\ntitle: 資料検索 スナップショット\nstatus: stable\n---\n"
            "# 目的\n過去資料の改訂を記録する。\n", encoding="utf-8")
        self.root = root
    def tearDown(self):
        self.tmp.cleanup()
    def test_search_excludes_obsolete(self):
        index = OkfIndex(self.root)
        self.assertEqual(index.search("資料検索", None, False, 5, 500)[0]["concept"], "design/buzzer.md")
        self.assertEqual(index.search("LocMemCache", None, False, 5, 500), [])
    def test_frontmatter_title_is_preferred_over_first_heading(self):
        index = OkfIndex(self.root)
        row = index.search("資料検索 スナップショット", None, False, 5, 500)[0]
        self.assertEqual(row["concept"], "design/frontmatter-title.md")
        self.assertEqual(row["title"], "資料検索 スナップショット")
    def test_trace_and_related(self):
        server = Server(self.root)
        result = server.call("okf_trace", {"identifier": "REQ-DEMO-001"})
        self.assertIn("design/buzzer.md", result["content"][0]["text"])
        self.assertEqual(server.index.related(server.index.resolve("design/buzzer.md"))[0]["concept"], "findings/open.md")
        self.assertEqual(server.index.resolve("design/buzzer.md").meta["generated.at"], "2026-08-06T00:00:00+09:00")
        self.assertEqual(server.index.resolve("design/buzzer.md").meta["sources"][0]["resource"], "要件定義書.md")
    def test_protocol_tools_and_resources(self):
        server = Server(self.root)
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        self.assertEqual(len(listed["result"]["tools"]), 6)
        resources = server.handle({"jsonrpc": "2.0", "id": 2, "method": "resources/list", "params": {}})
        self.assertEqual(len(resources["result"]["resources"]), 4)
        issues = json.loads(server.call("okf_list_open_issues", {})["content"][0]["text"])
        self.assertTrue(any(x["concept"] == "findings/open.md" for x in issues["issues"]))

if __name__ == "__main__":
    unittest.main()
