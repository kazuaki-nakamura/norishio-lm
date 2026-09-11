import pytest

torch = pytest.importorskip("torch")

from norishio_lm.local_slot_model import LocalSlotDecoder
from norishio_lm.toy_experiment import fixture_module, pad
from norishio_lm.toy_spans import authored_slot_spans
from norishio_lm.local_slot_model import DuplicateRemovedModel, PREFIXES
from test_local_slot_model import _setup


def test_fixed_prefix_rule_matches_training_grammar_slots():
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    # References are test expectations only; the gate API receives past-token IDs.
    rows = corpus.build()["train"]
    ids = pad([strict.target_history(row)["input_ids"] for row in rows])
    participant, time = LocalSlotDecoder._gate_masks(ids)
    for i, row in enumerate(rows):
        spans = authored_slot_spans(row["targets"], corpus.seed_data())
        for field, gates in (("participant", participant), ("time", time)):
            span = spans[field]
            assert gates[i, :, 0].nonzero().flatten().tolist() == list(range(span["start"], span["end"]))


def test_future_special_and_labels_cannot_change_earlier_outputs():
    initial, vocab, _ = _setup()
    model = DuplicateRemovedModel(initial, vocab, local=True).eval()
    ids = torch.tensor([[1, *[b + 4 for b in PREFIXES[0]], *([40] * 14)]])
    changed = ids.clone()
    changed[0, 13] = 3
    probs = torch.randn(1, 43)
    with torch.no_grad():
        first = model.decoder.decode_with_concept_intervention(ids, probs, labels=ids)
        second = model.decoder.decode_with_concept_intervention(changed, probs)
        relabeled = model.decoder.decode_with_concept_intervention(ids, probs, labels=ids + 1)
    assert torch.equal(first["logits"][:, :13], second["logits"][:, :13])
    assert torch.equal(first["logits"], relabeled["logits"])
    pg, tg = model.decoder._gate_masks(changed)
    assert tg[0, 9:13].all()
    assert not pg[0, 13:].any() and not tg[0, 13:].any()
    original_gates = model.decoder._gate_masks(ids)
    assert original_gates[0][0, 15:21].all()


def test_local_branch_direct_contribution_is_zero_outside_its_window():
    initial, vocab, _ = _setup()
    decoder = DuplicateRemovedModel(initial, vocab, local=True).decoder
    ids = torch.tensor([[1, *[b + 4 for b in PREFIXES[0]], *([40] * 14)]])
    probs = torch.randn(1, 43)
    pg, tg = decoder._gate_masks(ids)
    for section, gate in ((slice(33, 38), pg), (slice(38, 43), tg)):
        changed = probs.clone()
        changed[:, section] += 10
        with torch.no_grad():
            first = decoder.conditioning_for_prefix(ids, probs)
            second = decoder.conditioning_for_prefix(ids, changed)
        assert torch.equal(first[~gate.squeeze(-1)], second[~gate.squeeze(-1)])
        assert not torch.equal(first[gate.squeeze(-1)], second[gate.squeeze(-1)])
