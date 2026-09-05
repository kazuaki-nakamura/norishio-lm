from __future__ import annotations

from pathlib import Path
from pprint import pprint
import sys

from .semantic_compiler import SemanticCompiler


def main() -> None:
    # Windows terminals may default to CP932, which cannot represent every
    # sub-character used by the demo lexicon.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = Path(__file__).resolve().parents[2]
    compiler = SemanticCompiler.from_json(root / "data" / "demo_lexicon.json")
    for text in ["性", "性器", "心生", "ハナレナイ"]:
        print(f"\n=== {text} ===")
        pprint(compiler.compile(text))


if __name__ == "__main__":
    main()
