import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import TinyConceptDecoder
from norishio_lm.prefix_generation import BOS, EOS, prefix_generate
from norishio_lm.toy_generation import greedy_generate


def _decoder(dim=5, concept_dim=3, conditioning_mode="initial_only"):
    decoder = TinyConceptDecoder(vocab_size=260, hidden_dim=dim, conditioning_mode=conditioning_mode)
    decoder.concept_projection = torch.nn.Linear(concept_dim, dim)
    return decoder.eval()


def test_empty_prefix_matches_greedy_and_reports_generated_logits():
    decoder = _decoder()
    concepts = torch.randn(2, 3)
    with torch.no_grad():
        decoder.lm_head.bias[EOS] = -1e9
    rows = prefix_generate(decoder, concepts, [[], []], max_total_tokens=4)
    expected = greedy_generate(decoder, concepts, max_new_tokens=4)
    assert [row["token_ids"] for row in rows] == [row.token_ids for row in expected]
    assert all(row["decision_logits"].shape == (4, 260) for row in rows)
    assert all(row["generated_token_ids"] == row["token_ids"] for row in rows)


def test_first_decision_is_full_prefix_reference():
    decoder = _decoder()
    concepts = torch.randn(1, 3)
    prefix = [4, 17, 255]
    result = prefix_generate(decoder, concepts, [prefix], max_total_tokens=4)[0]
    input_ids = torch.tensor([[BOS, *prefix]])
    with torch.no_grad():
        reference = decoder.decode_with_concept_intervention(input_ids, concepts)["logits"][0, -1]
    assert torch.equal(result["decision_logits"][0], reference)


@pytest.mark.parametrize("conditioning_mode", ["initial_only", "per_step_additive"])
def test_every_decision_matches_full_history_public_decoder(conditioning_mode):
    decoder = _decoder(conditioning_mode=conditioning_mode)
    concepts = torch.randn(1, 3)
    prefix = [4, 17]
    result = prefix_generate(decoder, concepts, [prefix], max_total_tokens=6)[0]
    history = torch.tensor([[BOS, *prefix]])
    expected = []
    for _ in result["generated_token_ids"]:
        with torch.no_grad():
            logits = decoder.decode_with_concept_intervention(history, concepts)["logits"][0, -1]
        expected.append(logits)
        token = int(logits.argmax())
        history = torch.cat((history, torch.tensor([[token]])), dim=1)
    assert len(expected) == result["decision_logits"].shape[0]
    for actual, reference in zip(result["decision_logits"], expected):
        assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)


def test_generation_does_not_accept_or_use_future_gold_history():
    decoder = _decoder()
    concepts = torch.randn(1, 3)
    with pytest.raises(TypeError):
        prefix_generate(decoder, concepts, [[]], 4, reference=[4])  # type: ignore[call-arg]
    a = prefix_generate(decoder, concepts, [[4]], max_total_tokens=5)[0]
    b = prefix_generate(decoder, concepts, [[4]], max_total_tokens=5)[0]
    assert a["token_ids"] == b["token_ids"]


def test_prefix_uses_self_history_and_eos_counts_toward_cap():
    decoder = _decoder()
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.fill_(-100)
        decoder.lm_head.bias[EOS] = 10
    row = prefix_generate(decoder, torch.zeros(1, 3), [[4, 5]], max_total_tokens=3)[0]
    assert row["token_ids"] == [4, 5, EOS]
    assert row["generated_token_ids"] == [EOS]
    assert row["decision_logits"].shape == (1, 260)
    assert row["ended_eos"]


def test_total_cap_can_leave_no_generation_and_validates_prefixes():
    decoder = _decoder()
    row = prefix_generate(decoder, torch.zeros(1, 3), [[4, 5]], max_total_tokens=2)[0]
    assert row["token_ids"] == [4, 5]
    assert row["decision_logits"].shape == (0, 260)
    with pytest.raises(ValueError):
        prefix_generate(decoder, torch.zeros(1, 3), [[3]], 4)
    with pytest.raises(ValueError):
        prefix_generate(decoder, torch.zeros(1, 3), [[260]], 4)
    with pytest.raises(ValueError):
        prefix_generate(decoder, torch.zeros(1, 3), [[4, 5]], 1)
