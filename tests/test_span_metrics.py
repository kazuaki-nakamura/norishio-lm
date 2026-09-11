import json

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.span_metrics import span_metrics


def test_uniform_logits_have_tie_rank_one_and_logv_metrics():
    base = torch.zeros((1, 8, 260))
    got = span_metrics(base, base, torch.full((1, 8), 4),
                       [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}])
    assert got["participant"]["mean_correct_rank"] == 1
    assert got["participant"]["mean_correct_probability"] == pytest.approx(1 / 260)
    assert got["participant"]["mean_nll"] == pytest.approx(torch.log(torch.tensor(260.)).item())
    assert got["examples"]["participant"][0][0]["position"] == 0


def test_controlled_logits_and_unequal_spans_are_byte_weighted():
    base = torch.zeros((2, 9, 260))
    changed = base.clone()
    changed[0, 0, 10] = 10
    changed[1, 0, 11] = 10
    changed[1, 1, 12] = 10
    labels = torch.full((2, 9), 4)
    labels[0, 0] = 10
    labels[1, 0] = 11
    labels[1, 1] = 12
    got = span_metrics(base, changed, labels,
                       [{"participant": {"start": 0, "end": 1}, "time": {"start": 0, "end": 1}},
                        {"participant": {"start": 0, "end": 2}, "time": {"start": 0, "end": 1}}])
    assert got["participant"]["byte_count"] == 3
    assert got["participant"]["correct_argmax"] == 3
    assert got["participant"]["accuracy"] == 1
    assert got["participant"]["mean_logit_l1"] == pytest.approx(10 / 260)
    assert got["participant"]["row_count"] == 2
    json.dumps(got)


@pytest.mark.parametrize("bad", [0, 3, 260, -100])
def test_eos_pad_and_unknown_labels_are_rejected(bad):
    labels = torch.full((1, 8), 4)
    labels[0, 0] = bad
    with pytest.raises(ValueError):
        span_metrics(torch.zeros((1, 8, 260)), torch.zeros((1, 8, 260)), labels,
                     [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}])


def test_shape_span_and_nonfinite_validation():
    valid = [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}]
    with pytest.raises(ValueError):
        span_metrics(torch.zeros((1, 8, 260)), torch.zeros((1, 8, 260)), torch.zeros((1, 7), dtype=torch.long), valid)
    with pytest.raises(ValueError):
        span_metrics(torch.zeros((1, 8, 260)), torch.zeros((1, 8, 260)), torch.full((1, 8), 4), [{"participant": {"start": 0, "end": 1}}])
    invalid = torch.zeros((1, 8, 260)); invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        span_metrics(invalid, invalid, torch.full((1, 8), 4), valid)


def test_label_validation_uses_same_decoder_slot_without_position_offset():
    labels = torch.full((1, 8), 4)
    labels[0, 0] = 2  # EOS in the scored slot; shifted slot remains valid.
    valid = [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}}]
    with pytest.raises(ValueError):
        span_metrics(torch.zeros((1, 8, 260)), torch.zeros((1, 8, 260)), labels, valid)
