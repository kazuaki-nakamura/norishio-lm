import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("strict_decoder_data", ROOT / "data/issue3/strict_decoder.py")
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_strict_decoder_input_does_not_read_source_or_gold_concepts():
    a = {"inputs": {"text": "SOURCE_A"}, "targets": {"text": "答え。", "concept": "GOLD_A"}}
    b = {"inputs": {"text": "SOURCE_B"}, "targets": {"text": "答え。", "concept": "GOLD_B"}}
    assert module.target_history(a) == module.target_history(b)


def test_strict_decoder_labels_are_next_byte_with_eos():
    row = {"targets": {"text": "答え。"}}
    example = module.target_history(row)
    answer = [b + 4 for b in row["targets"]["text"].encode("utf-8")]
    assert example["input_ids"] == [1, *answer]
    assert example["labels"] == [*answer, 2]
    assert len(example["input_ids"]) == len(example["labels"])
