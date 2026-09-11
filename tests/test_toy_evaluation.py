import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm import toy_evaluation as evaluation
from norishio_lm.toy_generation import GenerationResult


def test_parameter_updates_count_final_element_differences():
    model = torch.nn.Linear(2, 1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.weight[0, 1] += 1
    result = evaluation.parameter_changes(before, model)
    assert result["registered"] == 3
    assert result["changed_elements"] == 1
    assert result["elements_in_changed_tensors"] == 2


@pytest.mark.parametrize("shape", [(3,), (1, 3), (1, 7, 1)])
def test_concept_group_dimensions_fail_before_argmax(shape):
    vocab = ConceptVocabulary.fit([])
    with pytest.raises(ValueError, match="dimensions"):
        evaluation.concept_ids(torch.zeros(shape), vocab)


def test_generation_receives_only_conditioning_and_reports_reference_separately(monkeypatch):
    vocab = ConceptVocabulary.fit([])
    probabilities = torch.ones(2, len(CONCEPT_FIELDS))
    class Model:
        decoder = object()
        def eval(self):
            return self
    class Corpus:
        @staticmethod
        def model_inputs(row):
            return row["source"]
    def generate(decoder, received, cap):
        assert decoder is model.decoder
        assert received is probabilities and cap == 128
        return [GenerationResult([2], "", True, True, []),
                GenerationResult([69], "A", False, True, [])]
    model = Model()
    monkeypatch.setattr(evaluation, "greedy_generate", generate)
    monkeypatch.setattr(evaluation, "conditional_loss", lambda *args: {"lm": 0})
    rows = [{"source": {"context": "", "text": "source"}, "targets": {"text": "gold"}}] * 2
    result = evaluation.score_condition(model, rows, probabilities, None, vocab, Corpus, None)
    assert [r["generation"]["stop_reason"] for r in result["examples"]] == ["eos", "max_new_tokens"]
    assert result["examples"][0]["source"]["text"] == "source"
    assert result["examples"][0]["authored_reference"] == "gold"
    assert result["generation"]["exact_match"] == 0
    json.dumps(result)
