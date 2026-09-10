from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_diagnosis import teacher_diagnostics, utf8_failure
from norishio_lm.toy_experiment import fixture_module


@pytest.mark.parametrize("tokens,kind,position", [
    ([235, 171, 133, 231, 133, 231], "illegal_transition", 5),
    ([231, 133], "incomplete_tail", 2),
    ([235, 171, 133, 2], "valid", None),
    ([2, 259], "valid", None),
    ([0], "special_token", 0),
])
def test_distinguishes_invalid_transition_from_cap_truncation(tokens, kind, position):
    result = utf8_failure(tokens)
    assert result["kind"] == kind
    assert result["position"] == position


def test_eos_measurement_uses_true_end_not_padding():
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    rows = [{"inputs": {"context": "", "text": "source"}, "targets": {"text": text}}
            for text in ("a", "bc")]
    tensorizer = SemanticTensorizer.fit([source_record(r["inputs"]) for r in rows])

    class Decoder:
        def decode_with_concept_intervention(self, ids, probabilities):
            assert ids.tolist() == [[1, 101, 0], [1, 102, 103]]
            logits = torch.full((2, 3, 260), -10.0)
            # EOS at PAD is deliberately high but must not affect counts.
            winners = torch.tensor([[101, 2, 2], [102, 103, 2]])
            logits.scatter_(2, winners.unsqueeze(-1), 10)
            return {"logits": logits}

    model = SimpleNamespace(eval=lambda: None, decoder=Decoder())
    result = teacher_diagnostics(model, rows, torch.zeros(2, 1), tensorizer, corpus, strict)
    assert result["byte_targets"] == 3
    assert result["byte_correct"] == 3
    assert result["eos_targets"] == 2
    assert result["eos_correct"] == 2
    assert result["mean_eos_probability_at_gold_end"] > .999
