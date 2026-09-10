import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import TinyConceptDecoder
from norishio_lm.toy_generation import EOS, GenerationResult, greedy_generate


def _decoder(vocab_size=8, dim=4, concept_dim=3):
    decoder = TinyConceptDecoder(vocab_size=vocab_size, hidden_dim=dim)
    decoder.concept_projection = torch.nn.Linear(concept_dim, dim)
    return decoder.eval()


def test_greedy_generation_reports_utf8_eos_and_specials():
    decoder = _decoder()
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.fill_(-100)
        decoder.lm_head.bias[5] = 10  # byte value 1 is valid UTF-8
        decoder.lm_head.bias[0] = 11  # PAD wins and is reported, then repeats
    result = greedy_generate(decoder, torch.zeros(2, 3), max_new_tokens=2)
    assert all(isinstance(row, GenerationResult) for row in result)
    assert result[0].token_ids == [0, 0]
    assert result[0].invalid_special_tokens == [0, 0]
    assert result[0].text == ""
    assert not result[0].ended_eos and result[0].valid_utf8


def test_eos_is_emitted_and_finished_rows_stop():
    decoder = _decoder(vocab_size=8)
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.fill_(-10)
        decoder.lm_head.bias[4] = 5
        decoder.lm_head.bias[EOS] = 6
    # EOS wins immediately and is included in the emitted sequence.
    rows = greedy_generate(decoder, torch.zeros(2, 3), max_new_tokens=10)
    assert [row.token_ids for row in rows] == [[EOS], [EOS]]
    assert all(row.ended_eos for row in rows)


def test_cap_and_incremental_logits_match_full_history():
    decoder = _decoder(vocab_size=12, dim=5)
    concepts = torch.randn(2, 3)
    cap = 4
    with torch.no_grad():
        decoder.lm_head.bias[EOS] = -1e9
    rows = greedy_generate(decoder, concepts, max_new_tokens=cap)
    assert all(len(row.token_ids) == cap for row in rows)
    # Recompute the same argmax decisions with the decoder's full-history C
    # forward pass, proving the incremental hidden state is equivalent.
    for row_index, row in enumerate(rows):
        history = torch.tensor([[1]], dtype=torch.long)
        expected = []
        for _ in range(cap):
            logits = decoder(history, pathway="C", concept_probs=concepts[row_index:row_index + 1])["logits"]
            token = int(logits[0, -1].argmax())
            expected.append(token)
            history = torch.cat((history, torch.tensor([[token]])), dim=1)
            if token == EOS:
                break
        assert row.token_ids == expected


def test_mixed_batch_rows_stop_at_different_times():
    decoder = _decoder(vocab_size=8)
    # Make the first row emit EOS immediately and the second row emit a byte
    # twice before EOS. The scripted head supplies the row distinction.
    class ScriptedHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden):
            self.calls += 1
            logits = torch.full((hidden.shape[0], 8), -100.0)
            logits[:, 4] = 1.0
            logits[0, EOS] = 10.0
            if hidden.shape[0] > 1:
                logits[1, EOS] = 10.0 if self.calls >= 3 else -100.0
            return logits
    decoder.lm_head = ScriptedHead()
    rows = greedy_generate(decoder, torch.zeros(2, 3), max_new_tokens=4)
    assert rows[0].token_ids == [EOS]
    assert rows[1].token_ids == [4, 4, EOS]


def test_fixed_concepts_produce_identical_sequences():
    decoder = _decoder()
    concepts = torch.randn(1, 3).expand(3, -1).clone()
    rows = greedy_generate(decoder, concepts, max_new_tokens=5)
    assert [row.token_ids for row in rows] == [rows[0].token_ids] * 3


def test_input_validation():
    decoder = _decoder()
    with pytest.raises(ValueError):
        greedy_generate(decoder, torch.zeros(3), 1)
    with pytest.raises(ValueError):
        greedy_generate(decoder, torch.zeros(1, 3), -1)
    with pytest.raises(ValueError):
        greedy_generate(decoder, torch.zeros(1, 3), 1.5)


def test_invalid_utf8_is_reported_without_repairing_emitted_ids():
    decoder = _decoder(vocab_size=260)
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.fill_(-100)
        decoder.lm_head.bias[259] = 10  # Byte 255 is invalid UTF-8.
    row = greedy_generate(decoder, torch.zeros(1, 3), 2)[0]
    assert row.token_ids == [259, 259]
    assert not row.valid_utf8
    assert row.text == "\ufffd\ufffd"
    assert row.invalid_special_tokens == []
    assert not row.ended_eos


def test_zero_cap_emits_nothing():
    row = greedy_generate(_decoder(), torch.zeros(1, 3), 0)[0]
    assert row.token_ids == []
    assert row.text == ""
    assert not row.ended_eos
