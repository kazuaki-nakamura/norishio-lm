#!/usr/bin/env python3
"""Read-only MCP server for a project OKF Markdown bundle."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

SERVER_NAME = "project-okf"
SERVER_VERSION = "0.1.0"
TOOLS = [
    {"name": "okf_search", "description": "OKFを検索し、関連度順に現在情報と根拠候補を返す。",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string"}, "category": {"type": "string"},
         "include_obsolete": {"type": "boolean", "default": False},
         "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
         "max_chars": {"type": "integer", "minimum": 200, "maximum": 8000, "default": 1200}},
         "required": ["query"]}},
    {"name": "okf_get_concept", "description": "相対パスまたはokf:// URIで概念本文を取得する。",
     "inputSchema": {"type": "object", "properties": {
         "concept": {"type": "string"}, "include_related": {"type": "boolean", "default": True},
         "max_chars": {"type": "integer", "minimum": 200, "maximum": 30000, "default": 6000}},
         "required": ["concept"]}},
    {"name": "okf_trace", "description": "要件ID、画面名、API、テーブル名などをOKF全体で横断追跡する。",
     "inputSchema": {"type": "object", "properties": {
         "identifier": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
         "max_chars": {"type": "integer", "minimum": 200, "maximum": 4000, "default": 1000}},
         "required": ["identifier"]}},
    {"name": "okf_list_open_issues", "description": "未決定、未解決、要確認、リスクを含む現行概念を一覧する。",
     "inputSchema": {"type": "object", "properties": {
         "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 30},
         "max_chars": {"type": "integer", "minimum": 200, "maximum": 4000, "default": 800}}}},
    {"name": "okf_check_freshness", "description": "OKF概念のstatusとstale_afterを確認する。",
     "inputSchema": {"type": "object", "properties": {
         "concept": {"type": "string"}, "include_current": {"type": "boolean", "default": False}}}},
    {"name": "okf_list_categories", "description": "OKFのカテゴリ別ファイル数と状態を返す。",
     "inputSchema": {"type": "object", "properties": {}}},
]
FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)")
HEADER_RE = re.compile(r"^#\s+(.+)$", re.M)
OPEN_RE = re.compile(r"(未決定|未解決|要確認|未確認|残課題|今後設計|リスク|TBD)", re.I)
OBSOLETE_RE = re.compile(r"^(obsolete|deprecated|retired|廃番|廃止)$", re.I)

@dataclass
class Concept:
    path: str
    title: str
    category: str
    text: str
    body: str
    meta: dict[str, Any]
    links: list[str]
    @property
    def status(self) -> str:
        return str(self.meta.get("status", "current"))
    @property
    def obsolete(self) -> bool:
        return bool(OBSOLETE_RE.match(self.status.strip()))
    @property
    def uri(self) -> str:
        return "okf://" + quote(self.path)

def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = FRONT_RE.search(text)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    current_list = None
    for raw in match.group(1).splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if re.match(r"^\s*-\s+", raw) and current_list:
            item_text = re.sub(r"^\s*-\s+", "", raw).strip()
            pair = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", item_text)
            if pair:
                meta.setdefault(current_list, []).append({pair.group(1): pair.group(2).strip().strip("'\"")})
            else:
                meta.setdefault(current_list, []).append(item_text.strip("'\""))
            continue
        nested = re.match(r"^\s+([A-Za-z0-9_.-]+):\s*(.*)$", raw)
        if nested and current_list and meta.get(current_list) and isinstance(meta[current_list][-1], dict):
            meta[current_list][-1][nested.group(1)] = nested.group(2).strip().strip("'\"")
            continue
        item = re.match(r"^([A-Za-z0-9_.-]+):\s*(.*)$", raw)
        if item:
            key, value = item.groups()
            if value:
                value = value.strip()
                if value.startswith("{") and value.endswith("}"):
                    fields = {}
                    for part in value[1:-1].split(","):
                        if ":" in part:
                            subkey, subvalue = part.split(":", 1)
                            fields[subkey.strip()] = subvalue.strip().strip("'\"")
                    meta[key] = fields
                    for subkey, subvalue in fields.items():
                        meta[f"{key}.{subkey}"] = subvalue
                else:
                    meta[key] = value.strip("'\"")
                current_list = None
            else:
                meta[key] = []
                current_list = key
    return meta, text[match.end():]

def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()

def query_terms(query: str) -> list[str]:
    full = normalize(query)
    parts = [p for p in re.split(r"[\s　,、/／:：()（）\[\]「」]+", full) if len(p) >= 2]
    return list(dict.fromkeys(([full] if full else []) + parts))

def snippet(text: str, query: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    pos = normalize(clean).find(normalize(query))
    start = max(0, pos - max_chars // 4) if pos >= 0 else 0
    result = clean[start:start + max_chars]
    return ("…" if start else "") + result + ("…" if start + max_chars < len(clean) else "")

class OkfIndex:
    def __init__(self, root: Path):
        self.root = Path(os.path.abspath(root))
        self.concepts: dict[str, Concept] = {}
        self.signature = ()
        self.refresh()
    def refresh(self) -> None:
        try:
            with os.scandir(self.root) as entries:
                next(entries, None)
        except OSError:
            raise RuntimeError(f"OKF root not found: {self.root}")
        files = sorted(
            Path(directory) / name
            for directory, _subdirs, names in os.walk(self.root)
            for name in names
            if name.casefold().endswith(".md")
        )
        signature = tuple((p.relative_to(self.root).as_posix(), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if signature == self.signature:
            return
        concepts = {}
        for path in files:
            rel = path.relative_to(self.root).as_posix()
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            meta, body = parse_frontmatter(text)
            heading = HEADER_RE.search(body)
            links = []
            for link in LINK_RE.findall(body):
                target = unquote(link.split("#", 1)[0])
                candidate = self.root / target.lstrip("/") if target.startswith("/") else path.parent / target
                resolved = Path(os.path.abspath(os.path.normpath(candidate)))
                try:
                    links.append(resolved.relative_to(self.root).as_posix())
                except ValueError:
                    pass
            title = str(meta.get("title") or (heading.group(1).strip() if heading else path.stem))
            concepts[rel] = Concept(rel, title,
                                    rel.split("/", 1)[0] if "/" in rel else "(root)", text, body, meta, links)
        self.concepts, self.signature = concepts, signature
    def summary(self, c: Concept, query: str, max_chars: int, score=None) -> dict[str, Any]:
        row = {"concept": c.path, "uri": c.uri, "title": c.title, "category": c.category,
               "status": c.status, "source": c.meta.get("sources", c.meta.get("source")),
               "updated_at": c.meta.get("generated.at", c.meta.get("updated_at")),
               "stale_after": c.meta.get("stale_after"), "snippet": snippet(c.body, query, max_chars)}
        if score is not None:
            row["score"] = score
        return row
    def search(self, query: str, category, include_obsolete: bool, limit: int, max_chars: int):
        self.refresh()
        terms, results = query_terms(query), []
        for c in self.concepts.values():
            if category and c.category.casefold() != str(category).casefold():
                continue
            if c.obsolete and not include_obsolete:
                continue
            path, title, body = normalize(c.path), normalize(c.title), normalize(c.body)
            score = sum(title.count(t) * 12 + path.count(t) * 8 + min(body.count(t), 10) * 2 for t in terms)
            if normalize(query) in body:
                score += 10
            if score <= 0:
                continue
            if c.category in {"design", "decisions", "requirements"}:
                score += 4
            if c.category == "(root)":
                score -= 8
            if c.path == "log.md":
                score -= 8
            results.append((score, c))
        results.sort(key=lambda x: (-x[0], x[1].path))
        return [self.summary(c, query, max_chars, score) for score, c in results[:limit]]
    def resolve(self, name: str) -> Concept:
        self.refresh()
        if name.startswith("okf://"):
            name = unquote(name[6:])
        name = name.replace("\\", "/").lstrip("/")
        if name not in self.concepts and not name.endswith(".md"):
            found = [c for p, c in self.concepts.items() if p == name + ".md" or Path(p).stem == name]
            if len(found) == 1:
                return found[0]
        if name not in self.concepts:
            raise KeyError(f"Concept not found: {name}")
        return self.concepts[name]
    def related(self, c: Concept):
        incoming = [other.path for other in self.concepts.values() if c.path in other.links]
        paths = list(dict.fromkeys(c.links + incoming))
        return [{"concept": p, "title": self.concepts[p].title,
                 "relation": "outgoing" if p in c.links else "incoming"} for p in paths if p in self.concepts]

def json_text(value):
    return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}]}

class Server:
    def __init__(self, root: Path):
        self.index = OkfIndex(root)
    def call(self, name: str, args: dict[str, Any]):
        if name == "okf_search":
            return json_text({"results": self.index.search(str(args["query"]), args.get("category"),
                bool(args.get("include_obsolete", False)), int(args.get("limit", 5)), int(args.get("max_chars", 1200)))})
        if name == "okf_get_concept":
            c, max_chars = self.index.resolve(str(args["concept"])), int(args.get("max_chars", 6000))
            result = self.index.summary(c, c.title, max_chars)
            result["content"] = c.body[:max_chars] + ("…" if len(c.body) > max_chars else "")
            if args.get("include_related", True):
                result["related"] = self.index.related(c)
            return json_text(result)
        if name == "okf_trace":
            identifier = str(args["identifier"])
            return json_text({"identifier": identifier, "matches": self.index.search(identifier, None, True,
                int(args.get("limit", 15)), int(args.get("max_chars", 1000)))})
        if name == "okf_list_open_issues":
            self.index.refresh()
            rows = []
            for c in self.index.concepts.values():
                match = OPEN_RE.search(c.body + " " + c.status)
                if match and not c.obsolete:
                    rows.append(self.index.summary(c, match.group(0), int(args.get("max_chars", 800))))
            return json_text({"issues": rows[:int(args.get("limit", 30))]})
        if name == "okf_check_freshness":
            self.index.refresh()
            concepts = [self.index.resolve(str(args["concept"]))] if args.get("concept") else list(self.index.concepts.values())
            rows, today = [], date.today()
            for c in concepts:
                stale_after, stale = c.meta.get("stale_after"), False
                if stale_after:
                    try:
                        stale = date.fromisoformat(str(stale_after)[:10]) < today
                    except ValueError:
                        pass
                if args.get("include_current", False) or stale or c.obsolete:
                    rows.append({"concept": c.path, "title": c.title, "status": c.status,
                                 "stale_after": stale_after, "stale": stale})
            return json_text({"as_of": today.isoformat(), "concepts": rows})
        if name == "okf_list_categories":
            self.index.refresh()
            categories = {}
            for c in self.index.concepts.values():
                row = categories.setdefault(c.category, {"total": 0, "current": 0, "obsolete": 0})
                row["total"] += 1
                row["obsolete" if c.obsolete else "current"] += 1
            return json_text({"root": str(self.index.root), "categories": categories})
        raise KeyError(f"Unknown tool: {name}")
    def handle(self, request):
        method, request_id = request.get("method"), request.get("id")
        if method and method.startswith("notifications/"):
            return None
        try:
            if method == "initialize":
                version = request.get("params", {}).get("protocolVersion", "2024-11-05")
                result = {"protocolVersion": version, "capabilities": {"tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "instructions": "最初にokf_searchで対象概念を絞り、必要な概念だけokf_get_conceptで展開する。現行判断ではstable/currentを優先し、deprecated/obsoleteは比較時だけ取得する。重要な結論は返却されたsourcesから原本へ戻って確認する。OKFは読取専用であり、更新はプロジェクトのOKF保守手順に従う。"}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = request.get("params", {})
                result = self.call(params.get("name", ""), params.get("arguments") or {})
            elif method == "resources/list":
                self.index.refresh()
                result = {"resources": [{"uri": c.uri, "name": c.title, "description": c.path,
                    "mimeType": "text/markdown"} for c in self.index.concepts.values()]}
            elif method == "resources/read":
                c = self.index.resolve(request.get("params", {}).get("uri", ""))
                result = {"contents": [{"uri": c.uri, "mimeType": "text/markdown", "text": c.text}]}
            else:
                return {"jsonrpc": "2.0", "id": request_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, ValueError, RuntimeError) as exc:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(exc)}}
        except Exception as exc:
            return {"jsonrpc": "2.0", "id": request_id,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}

def main():
    # MCP JSON-RPC uses UTF-8 even when Windows defaults to a legacy code page.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "okf")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    try:
        server = Server(args.root)
    except Exception as exc:
        print(f"{SERVER_NAME}: {exc}", file=sys.stderr)
        return 2
    if args.self_check:
        print(json.dumps({"ok": True, "root": str(server.index.root),
                          "concepts": len(server.index.concepts)}, ensure_ascii=False))
        return 0
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = server.handle(json.loads(line))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
        except json.JSONDecodeError as exc:
            print(json.dumps({"jsonrpc": "2.0", "id": None,
                              "error": {"code": -32700, "message": str(exc)}}), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
