import pytest

torch = pytest.importorskip("torch")

from norishio_lm.toy_step_experiment import conditioning_sensitivity, decision_history


def test_sensitivity_prefix_covers_only_emitted_decisions_for_eos_and_cap():
    assert decision_history([9, 10, 2]) == [1, 9, 10]
    cap_tokens = [9] * 128
    history = decision_history(cap_tokens)
    assert history == [1] + [9] * 127
    assert len(history) == len(cap_tokens)


class ControlledDecoder(torch.nn.Module):
    def decode_with_concept_intervention(self, ids, concepts):
        # Fixture with exactly known two-class differences, including padding.
        first = concepts[:, :1].expand_as(ids) * (ids + 1)
        return {"logits": torch.stack([first, torch.zeros_like(first)], dim=-1)}


def test_sensitivity_excludes_padding_and_uses_same_fixed_history():
    model = ControlledDecoder()
    histories = [[1, 4, 5], [1]]
    baseline, changed = torch.zeros(2, 1), torch.tensor([[1.], [3.]])
    result = conditioning_sensitivity(model, histories, baseline, changed)
    positions = result["positions"]
    assert [p["rows"] for p in positions] == [2, 1, 1]
    assert [p["mean_logit_l1"] for p in positions] == [2., 2.5, 3.]
    assert all(p["mean_kl"] > 0 for p in positions)
    assert all(p["argmax_change_rate"] == 0 for p in positions)
    assert histories == [[1, 4, 5], [1]]


def test_identical_conditioning_has_zero_sensitivity_every_position():
    p = torch.tensor([[.3], [.8]])
    result = conditioning_sensitivity(ControlledDecoder(), [[1, 4], [1]], p, p)
    assert all(row["mean_logit_l1"] == row["mean_kl"] == row["argmax_change_rate"] == 0
               for row in result["positions"])


def test_history_must_start_at_bos_and_be_aligned():
    p = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        conditioning_sensitivity(ControlledDecoder(), [[4]], p, p)
    with pytest.raises(ValueError):
        conditioning_sensitivity(ControlledDecoder(), [[1], [1]], p, p)


def test_per_step_normal_generation_cannot_see_changed_gold():
    from copy import deepcopy
    from norishio_lm.concept_model import ConceptVocabulary
    from norishio_lm.tensorizer import SemanticTensorizer
    from norishio_lm.toy_adapter import source_record
    from norishio_lm.toy_collapse import source_encodings
    from norishio_lm.toy_evaluation import score_condition
    from norishio_lm.toy_experiment import ToyModel, fixture_module
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    rows = corpus.build()["train"][:2]
    sources = [corpus.model_inputs(r) for r in rows]
    tensorizer = SemanticTensorizer.fit([source_record(s) for s in sources])
    vocab = ConceptVocabulary.fit([r["targets"] for r in rows])
    model = ToyModel("C", tensorizer, vocab, conditioning_mode="per_step_additive")
    _, probabilities = source_encodings(model, sources, tensorizer)
    original = score_condition(model, rows, probabilities, tensorizer, vocab, corpus, strict)
    changed = deepcopy(rows)
    for row in changed:
        row["targets"]["text"] = "different reference"
        row["targets"]["concept"] = None
    altered = score_condition(model, changed, probabilities, tensorizer, vocab, corpus, strict)
    assert [r["generation"] for r in original["examples"]] == [r["generation"] for r in altered["examples"]]
    assert original["lm"] != altered["lm"]
    with pytest.raises(ValueError):
        source_encodings(model, [{**sources[0], "targets": {}}], tensorizer)
