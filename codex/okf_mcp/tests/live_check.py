"""End-to-end JSON-RPC check against the real project OKF."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    process = subprocess.Popen(
        [sys.executable, str(args.server), "--root", str(args.root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    def request(request_id, method, params):
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, ensure_ascii=True) + "\n")
        process.stdin.flush()
        response = json.loads(process.stdout.readline())
        if "error" in response:
            raise AssertionError(response["error"])
        return response["result"]
    try:
        initialized = request(1, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "live-check", "version": "1"}})
        tools = request(2, "tools/list", {})
        resources = request(3, "resources/list", {})
        search = json.loads(request(4, "tools/call", {"name": "okf_search", "arguments": {"query": "SemanticCompiler", "limit": 5, "max_chars": 500}})["content"][0]["text"])
        trace = json.loads(request(5, "tools/call", {"name": "okf_trace", "arguments": {"identifier": "NLM-SEM-001", "limit": 10, "max_chars": 400}})["content"][0]["text"])
        issues = json.loads(request(6, "tools/call", {"name": "okf_list_open_issues", "arguments": {"limit": 30, "max_chars": 300}})["content"][0]["text"])
        freshness = json.loads(request(7, "tools/call", {"name": "okf_check_freshness", "arguments": {}})["content"][0]["text"])
        assert initialized["serverInfo"]["name"] == "project-okf"
        assert len(tools["tools"]) == 6
        assert len(resources["resources"]) >= 1
        assert len(search["results"]) >= 1
        assert len(trace["matches"]) >= 1
        assert len(issues["issues"]) >= 1
        assert "as_of" in freshness
        concept = json.loads(request(8, "tools/call", {"name": "okf_get_concept", "arguments": {"concept": "design/semantic-compiler.md"}})["content"][0]["text"])
        assert "手書き" in concept["content"]
        evidence = request(9, "resources/read", {"uri": "okf://research/semantic-separation.md"})
        assert "NLM-SEM-001" in evidence["contents"][0]["text"]
        print(json.dumps({
            "ok": True, "concepts": len(resources["resources"]), "tools": len(tools["tools"]),
            "search_hits": len(search["results"]), "search_top": search["results"][0]["concept"],
            "trace_hits": len(trace["matches"]), "open_issue_concepts": len(issues["issues"]),
            "stale_or_obsolete": len(freshness["concepts"]),
        }, ensure_ascii=False))
    finally:
        process.stdin.close()
        process.terminate()
        process.wait(timeout=5)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
