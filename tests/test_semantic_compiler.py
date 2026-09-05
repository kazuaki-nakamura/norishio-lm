from pathlib import Path

from norishio_lm import SemanticCompiler


ROOT = Path(__file__).resolve().parents[1]


def compiler() -> SemanticCompiler:
    return SemanticCompiler.from_json(ROOT / "data" / "demo_lexicon.json")


def test_sei_keeps_glyph_and_sense_separate() -> None:
    record = compiler().compile("性")
    assert record.subcharacters["性"] == ("忄", "生")
    assert len(record.senses) >= 2
    assert "literal" not in record.etymology_notes["性"].lower()


def test_seiki_has_anatomy_concept() -> None:
    record = compiler().compile("性器")
    assert any("ANATOMY" in sense.concepts for sense in record.senses)


def test_unknown_expression_remains_compilable() -> None:
    record = compiler().compile("未知語")
    assert record.surface == "未知語"
    assert record.tokens == ("未知語",)
