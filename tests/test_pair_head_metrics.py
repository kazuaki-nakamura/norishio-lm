import math

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.pair_head_metrics import pair_head_metrics


def row(participant, time):
    return {"concept": {"participant": participant, "time": time}}


def vocab():
    return ConceptVocabulary.fit([row(f"p{i}", f"t{i}") for i in range(5)])


def test_pair_order_metrics_calibration_rank_and_seen_groups():
    v = vocab()
    targets = [row("p0", "t0"), row("p1", "t1"), row("p2", "t2")]
    probs = torch.full((3, 25), 0.02, dtype=torch.float64)
    probs[0, 0], probs[1, 6], probs[2, 12] = 0.50, 0.20, 0.20
    probs[1, 0] = 0.20  # tie for row 1: lowest pair index must rank first
    probs[2, 12], probs[2, 0] = 0.20, 0.20
    # Renormalize while preserving the intended ties and gold columns.
    probs /= probs.sum(-1, keepdim=True)
    result = pair_head_metrics(probs.tolist(), targets, v, [row("p0", "t0")])
    assert result["pair_order"] == "participant_class_id_then_time_class_id"
    assert result["tie_policy"] == "lowest_pair_index"
    assert result["all"]["count"] == 3
    assert result["train_seen"]["count"] == 1
    assert result["train_unseen"]["count"] == 2
    assert result["all"]["rows"][0]["gold_pair_index"] == 0
    assert result["all"]["rows"][1]["gold_rank"] == 2
    assert result["all"]["rows"][2]["gold_rank"] == 2
    assert result["all"]["ece"]["bins"]
    assert sum(result["all"]["support"]) == 3


def test_pair_metrics_hand_computed_nll_brier_and_empty_group_nulls():
    v = vocab()
    target = [row("p0", "t0")]
    probs = [[0.4] + [0.6 / 24] * 24]
    result = pair_head_metrics(probs, target, v, [row("p1", "t1")])
    assert result["all"]["nll"] == pytest.approx(-math.log(0.4))
    expected_brier = (0.6**2 + 24 * (0.6 / 24) ** 2)
    assert result["all"]["brier"] == pytest.approx(expected_brier)
    assert result["train_seen"]["accuracy"] is None
    assert result["train_seen"]["nll"] is None
    assert all(item["count"] == 0 for item in result["train_seen"]["ece"]["bins"])


def test_pair_metrics_reject_invalid_shape_values_and_unknown_labels():
    v = vocab()
    target = [row("p0", "t0")]
    with pytest.raises(ValueError, match="shape"):
        pair_head_metrics(torch.ones((1, 24)) / 24, target, v)
    with pytest.raises(ValueError, match="nonnegative"):
        pair_head_metrics([[-1.0] + [2 / 24] * 24], target, v)
    with pytest.raises(ValueError, match="sum"):
        pair_head_metrics([[0.0] * 25], target, v)
    with pytest.raises(ValueError, match="unknown participant"):
        pair_head_metrics([[1.0] + [0.0] * 24], [row("nope", "t0")], v)
    with pytest.raises(ValueError, match="finite"):
        pair_head_metrics([[float("nan")] + [0.0] * 24], target, v)
