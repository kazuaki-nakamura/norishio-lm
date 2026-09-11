import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.toy_experiment import fixture_module
from norishio_lm.toy_explicit_slots import conditioning_cases, gold_heads, head_statistics, sensitivity, run


def test_head_only_interventions_preserve_old_concepts_and_oracles_only_named_groups():
    corpus = fixture_module("toy_corpus")
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in corpus.build()["train"]])
    widths = [len(vocabulary.fields[f]) + 1 for f in CONCEPT_FIELDS]
    old = torch.cat([torch.full((2, w), 1/w) for w in widths], -1)
    old_gold = torch.cat([torch.nn.functional.one_hot(torch.ones(2, dtype=torch.long), w).float() for w in widths], -1)
    heads = torch.tensor([[.1,.2,.3,.2,.2, .2,.1,.3,.1,.3], [.3,.1,.1,.4,.1, .1,.1,.1,.2,.5]])
    gold = torch.tensor([[1.,0,0,0,0, 0,1.,0,0,0]]).expand(2,-1)
    cases = conditioning_cases(old, heads, old_gold, gold, heads.mean(0,keepdim=True), [1,0], vocabulary)
    for name in ("predicted", "head_train_mean", "head_permuted", "head_gold"):
        assert torch.equal(cases[name][:,:33], old)
    assert torch.equal(cases["head_permuted"][:,33:], heads[[1,0]])
    assert torch.equal(cases["participant_oracle"][:,38:], heads[:,5:])
    assert torch.equal(cases["time_oracle"][:,33:38], heads[:,:5])
    offset = 0
    for field, width in zip(CONCEPT_FIELDS, widths):
        if field != "participant":
            assert torch.equal(cases["participant_oracle"][:,offset:offset+width],old[:,offset:offset+width])
        offset += width


def test_five_class_head_stats_match_gold_and_reject_unknown():
    corpus = fixture_module("toy_corpus")
    bundle = corpus.build()
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    targets = [r["targets"] for r in bundle["validation"]]
    stats = head_statistics(gold_heads(targets,vocabulary),targets,vocabulary)
    for field in ("participant","time"):
        assert stats[field]["support"] == [30]*5
        assert stats[field]["correct"] == 150
        assert stats[field]["balanced_accuracy"] == 1
        assert stats[field]["entropy_mean"] == 0
    with pytest.raises(ValueError):
        gold_heads([{"concept": {"participant": "unseen", "time": None}}], vocabulary)


def test_sensitivity_tracks_before_and_inside_byte_positions_without_id_offset():
    base = torch.zeros(1, 6, 260)
    changed = base.clone()
    changed[0,0,4] = 2
    changed[0,1,4] = 2
    spans = [{"participant": {"start":1,"end":3}, "time":{"start":4,"end":6}}]
    result = sensitivity(base,changed,spans)["fields"]
    assert result["participant"]["before"]["argmax_change_rate"] == 1
    assert result["participant"]["inside"]["argmax_change_rate"] == .5
    assert result["participant"]["before"]["mean_logit_l1"] == pytest.approx(2/260)
    assert result["time"]["inside"]["mean_kl"] == 0


def test_wrong_historical_report_fails_before_outputs(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="hash mismatch"):
        run(report,tmp_path / "out")
    assert not (tmp_path / "out").exists()
