import pytest

torch = pytest.importorskip("torch")
from norishio_lm.head_following_diagnostics import evaluate_condition
from norishio_lm.local_slot_model import DuplicateRemovedModel
from norishio_lm.explicit_slot_model import source_distributions
from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import fixture_module, ToyModel
from norishio_lm.toy_controls import state_digest


def test_byte_v2_is_reachable_without_changing_legacy_generation_or_weights():
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    train = corpus.build()["train"]
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in train])
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in train])
    model = DuplicateRemovedModel(ToyModel("C", tensorizer, vocabulary,
        conditioning_mode="per_step_additive"), vocabulary, local=True)
    rows = train[:2]
    old, heads = source_distributions(model, [corpus.model_inputs(r) for r in rows], tensorizer)
    probs = torch.cat((old, heads), -1)
    before = state_digest(model)
    legacy = evaluate_condition(model, rows, probs, tensorizer, vocabulary, corpus, strict)
    new = evaluate_condition(model, rows, probs, tensorizer, vocabulary, corpus, strict, include_teacher_forced=True)
    assert legacy["teacher_forced"] is None
    diagnostic = new["teacher_forced"]
    assert diagnostic["version"] == "teacher-forced-byte-v2"
    assert diagnostic["reference_history_oracle"] is True
    assert diagnostic["metrics"] is not None
    for field in ("participant", "time"):
        value = diagnostic["metrics"][field]
        assert value["byte_count"] == 12
        assert value["row_count"] == 2
        assert value["mean_nll"] > 0
        assert value["mean_logit_l1"] == value["mean_kl"] == 0
    new["teacher_forced"] = None
    assert legacy == new
    assert before == state_digest(model)
