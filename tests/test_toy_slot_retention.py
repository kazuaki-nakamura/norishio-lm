import hashlib
import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.toy_slot_retention import single_field_conditions, verify_baseline


def test_single_slot_oracles_leave_every_other_group_and_input_unchanged():
    vocab = ConceptVocabulary.fit([{"concept": {f: (["WANT"] if f == "operators" else "x") for f in CONCEPT_FIELDS}}])
    p = torch.tensor([[.3, .7] * 7])
    before = p.clone()
    oracle = torch.tensor([[0., 1.] * 7])
    result = single_field_conditions(p, oracle, vocab)
    replacements = {"predicted": (), "participant_oracle": ("participant",), "time_oracle": ("time",),
                    "both_oracle": ("participant", "time"), "full_oracle": CONCEPT_FIELDS}
    for name, changed in replacements.items():
        for i, field in enumerate(CONCEPT_FIELDS):
            wanted = oracle if field in changed else before
            assert torch.equal(result[name][:, i*2:i*2+2], wanted[:, i*2:i*2+2])
    assert torch.equal(p, before)


def test_future_reference_bytes_do_not_influence_current_or_earlier_logits():
    from norishio_lm.tensorizer import SemanticTensorizer
    from norishio_lm.toy_adapter import source_record
    from norishio_lm.toy_experiment import ToyModel, fixture_module
    from norishio_lm.toy_collapse import source_encodings
    from norishio_lm.toy_spans import authored_slot_spans
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    row = corpus.build()["train"][0]
    source = corpus.model_inputs(row)
    tensorizer = SemanticTensorizer.fit([source_record(source)])
    vocab = ConceptVocabulary.fit([row["targets"]])
    model = ToyModel("C", tensorizer, vocab, conditioning_mode="per_step_additive").eval()
    _, probs = source_encodings(model, [source], tensorizer)
    inputs = torch.tensor([strict.target_history(row)["input_ids"]])
    position = authored_slot_spans(row["targets"], corpus.seed_data())["participant"]["start"]
    changed = inputs.clone()
    changed[:, position+1:] = 4
    with torch.no_grad():
        a = model.decoder.decode_with_concept_intervention(inputs, probs)["logits"]
        b = model.decoder.decode_with_concept_intervention(changed, probs)["logits"]
    assert torch.equal(a[:, :position+1], b[:, :position+1])
    assert not torch.equal(a[:, position+1:], b[:, position+1:])
    with pytest.raises(ValueError):
        source_encodings(model, [{**source, "concept": row["targets"]["concept"]}], tensorizer)


def test_baseline_rejects_different_budget_before_replay(tmp_path):
    with pytest.raises(ValueError, match="fixed Issue12"):
        verify_baseline(tmp_path / "missing.pt", {"version": "other"})


def test_baseline_checkpoint_hash_detects_substituted_weights(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"original checkpoint")
    budget = {"seed": 7, "updates": 600, "batch_size": 16, "optimizer": "Adam",
              "learning_rate": .003, "clip": 1., "all_four_loss_weights": 1.,
              "threads": 1, "device": "cpu", "max_new_tokens": 128}
    arm = {"checkpoint": {"sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    report = {"version": "issue12-step-conditioning-1", "budget": budget, "arms": {"per_step_additive": arm}}
    assert verify_baseline(checkpoint, report) == arm
    checkpoint.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="hash"):
        verify_baseline(checkpoint, report)


def test_scored_target_ids_decode_to_exact_slot_without_positional_offset():
    from norishio_lm.toy_experiment import fixture_module
    from norishio_lm.toy_spans import authored_slot_spans
    from norishio_lm.span_metrics import span_metrics
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    row = corpus.build()["train"][0]
    span = authored_slot_spans(row["targets"], corpus.seed_data())
    labels = torch.tensor([strict.target_history(row)["labels"]])
    logits = torch.zeros(1, labels.shape[1], 260)
    metrics = span_metrics(logits, logits, labels, [span])
    for field in ("participant", "time"):
        scored = metrics["examples"][field][0]
        actual = bytes(item["target_id"] - 4 for item in scored).decode("utf-8")
        assert actual == row["targets"]["concept"][field]
        assert scored[0]["position"] == span[field]["start"]
        assert scored[-1]["position"] + 1 == span[field]["end"]
