import math

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.slot_objective import slot_ce
from norishio_lm.toy_slot_objective import run


def test_slot_loss_is_byte_weighted_not_field_or_row_averaged():
    probabilities = torch.empty(2, 5, 260)
    for row, correct in enumerate((.5, .25)):
        probabilities[row].fill_((1 - correct) / 259)
        probabilities[row, :, 4] = correct
    labels = torch.full((2, 5), 4, dtype=torch.long)
    spans = [{"participant": {"start": 0, "end": 1}, "time": {"start": 1, "end": 2}},
             {"participant": {"start": 0, "end": 3}, "time": {"start": 3, "end": 4}}]
    actual = slot_ce(probabilities.log(), labels, spans)
    assert float(actual) == pytest.approx((2 * math.log(2) + 4 * math.log(4)) / 6)


def test_changed_historical_report_rejected_before_training_or_output(tmp_path):
    report = tmp_path / "changed.json"
    report.write_text("{}", encoding="utf-8")
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="report hash mismatch"):
        run(report, tmp_path / "missing.pt", out)
    assert not out.exists()
