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
    print("Hand-authored demo dictionary; tokens are demo splits, not a learned tokenizer.")
    print("All senses are unselected candidates; no trained model or contextual selector is used.")
    for text in ["性", "性器", "心生", "ハナレナイ", "離れない", "離れたくない"]:
        print(f"\n=== {text} ===")
        pprint(compiler.compile(text))


if __name__ == "__main__":
    main()
