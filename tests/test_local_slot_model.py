from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.local_slot_generation import greedy_generate_local
from norishio_lm.local_slot_model import DuplicateRemovedModel, PREFIXES
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import ToyModel


def _setup():
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1,
                                             '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 3, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 2}
    fields = {name: {f"{name}-{i}": i + 1 for i in range(count)}
              for name, count in counts.items()}
    vocab = ConceptVocabulary(fields, {"sense": 1}, {"sememe": 0})
    initial = ToyModel("C", tensorizer, vocab, hidden_dim=32,
                       conditioning_mode="per_step_additive")
    return initial, vocab, tensorizer


def test_capacity_and_rng_are_bounded_and_reproducible():
    initial, vocab, _ = _setup()
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    first = DuplicateRemovedModel(initial, vocab, new_seed=20)
    observed = torch.rand(4)
    second = DuplicateRemovedModel(initial, vocab, new_seed=20)
    assert torch.equal(observed, expected)
    assert sum(p.numel() for p in first.parameters()) - sum(p.numel() for p in initial.parameters()) == 266
    assert first.decoder.concept_projection.base_proj.in_features == 21
    assert first.decoder.concept_projection.in_features == 43
    assert first.state_dict().keys() == second.state_dict().keys()
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])


def test_kept_indices_use_fixed_concept_field_order():
    initial, vocab, _ = _setup()
    reversed_vocab = ConceptVocabulary(dict(reversed(list(vocab.fields.items()))), vocab.senses, vocab.sememes)
    model = DuplicateRemovedModel(initial, reversed_vocab)
    assert model.kept_old_indices == tuple(i for i in range(33) if i not in range(14, 26))


def test_duplicate_old_groups_do_not_change_decoder():
    initial, vocab, _ = _setup()
    model = DuplicateRemovedModel(initial, vocab)
    probs = torch.randn(2, 43)
    changed = probs.clone()
    changed[:, 14:26] += 100.0
    ids = torch.tensor([[1, 4, 5], [1, 4, 5]])
    with torch.no_grad():
        a = model.decoder(ids, pathway="C", concept_probs=probs)["logits"]
        b = model.decoder(ids, pathway="C", concept_probs=changed)["logits"]
    assert torch.equal(a, b)


def test_local_gates_require_exact_prefix_and_have_no_future_dependency():
    initial, vocab, _ = _setup()
    model = DuplicateRemovedModel(initial, vocab, local=True).eval()
    probs = torch.randn(2, 43)
    prefix = [1] + [4 + x for x in PREFIXES[0]]
    ids = torch.tensor([prefix + [4, 5, 6], prefix + [99, 99, 99]])
    with torch.no_grad():
        cond = model.decoder.conditioning_for_prefix(ids, probs)
        altered = ids.clone()
        altered[:, -1] = 250
        cond2 = model.decoder.conditioning_for_prefix(altered, probs)
    # Prefix remains valid; slot contents are deliberately unconstrained.
    assert torch.equal(cond, cond2)
    bad = ids.clone()
    bad[:, 2] = 3
    assert torch.equal(model.decoder.conditioning_for_prefix(bad, probs),
                       model.decoder.initial_conditioning(probs)[:, None, :].expand_as(cond))


def test_both_prefixes_activate_time_then_participant_windows():
    initial, vocab, _ = _setup()
    model = DuplicateRemovedModel(initial, vocab, local=True).eval()
    probs = torch.randn(2, 43)
    rows = []
    for prefix in PREFIXES:
        rows.append([1] + [4 + x for x in prefix] + [4] * 12)
    width = max(map(len, rows))
    ids = torch.tensor([row + [4] * (width - len(row)) for row in rows])
    with torch.no_grad():
        cond = model.decoder.conditioning_for_prefix(ids, probs)
        base = model.decoder._projection_parts(probs)[0]
    pg, tg = model.decoder._gate_masks(ids)
    for row, prefix in enumerate(PREFIXES):
        n = len(prefix)
        assert not bool(tg[row, n:n + 6].sum() == 0)
        assert not bool(pg[row, n + 6:n + 12].sum() == 0)
        assert not bool(tg[row, :n].any() or tg[row, n + 6:].any())
        assert not bool(pg[row, :n + 6].any() or pg[row, n + 12:].any())
        assert torch.equal(cond[row, 0], torch.tanh(base[row]))


def test_lm_gradients_reach_heads_and_projection():
    initial, vocab, tensorizer = _setup()
    model = DuplicateRemovedModel(initial, vocab, local=True)
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})] * 2)
    prefix = [1] + [4 + x for x in PREFIXES[0]] + [4] * 12
    ids = torch.tensor([prefix, prefix])
    labels = torch.tensor([prefix, prefix])
    loss = model(ids, semantic, labels=labels)[0]["lm_loss"]
    loss.backward()
    assert model.slot_heads.participant.weight.grad.abs().sum() > 0
    assert model.slot_heads.time.weight.grad.abs().sum() > 0
    assert model.decoder.concept_projection.base_proj.weight.grad.abs().sum() > 0
    assert model.decoder.concept_projection.participant_proj.weight.grad.abs().sum() > 0
    assert model.decoder.concept_projection.time_proj.weight.grad.abs().sum() > 0


def test_local_generation_matches_full_prefix_logits():
    initial, vocab, _ = _setup()
    model = DuplicateRemovedModel(initial, vocab, local=True).eval()
    probs = torch.randn(1, 43)
    generated = greedy_generate_local(model.decoder, probs, max_new_tokens=5)
    seq = generated[0].token_ids
    for position, token in enumerate(seq):
        ids = torch.tensor([[1] + seq[:position]])
        with torch.no_grad():
            full = model.decoder(ids, pathway="C", concept_probs=probs)["logits"][:, -1]
        assert int(full.argmax(-1)) == token
