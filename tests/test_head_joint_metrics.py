import json

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.head_joint_metrics import head_joint_metrics


def _targets():
    return [{"concept": {"participant": f"p{i}", "time": f"t{i}"}} for i in range(5)]


def _vocab():
    return ConceptVocabulary.fit(_targets())


def test_head_and_joint_metrics_are_json_safe_and_count_conjunctions():
    vocab = _vocab()
    heads = torch.full((5, 10), 0.05)
    for i in range(5):
        heads[i, i] = 0.8
        heads[i, 5 + i] = 0.8
    heads[1, 0], heads[1, 1], heads[1, 2] = 0.1, 0.1, 0.7  # participant wrong
    heads[2, 5], heads[2, 7], heads[2, 8] = 0.1, 0.1, 0.7  # time wrong
    result = head_joint_metrics(heads, _targets(), vocab)
    assert result["joint_exact"]["correct_both"] == result["joint_exact"]["correct"] == 3
    assert result["joint_exact"]["participant_only"] == 1
    assert result["joint_exact"]["time_only"] == 1
    assert result["joint_exact"]["both_wrong"] == 0
    assert result["joint_exact"]["count"] == 5
    assert result["joint_exact"]["accuracy"] == pytest.approx(0.6)
    assert result["per_head"]["participant"]["accuracy"] == pytest.approx(0.8)
    assert sum(result["per_head"]["participant"]["support"]) == 5
    assert len(result["joint"]["gold_pair_cells"]) == 5
    assert json.loads(json.dumps(result))["joint"]["raw_rows"][0]["gold_joint_logprob"] == pytest.approx(2 * __import__("math").log(.8))


def test_brier_nll_ece_boundary_and_gold_pair_counts():
    vocab = _vocab()
    heads = torch.zeros((5, 10))
    for i in range(5):
        heads[i, i] = 1.0
        heads[i, 5 + i] = 1.0
    result = head_joint_metrics(heads, _targets(), vocab)
    metrics = result["participant"]
    assert metrics["brier"] == pytest.approx(0.0)
    assert metrics["nll"] == pytest.approx(-__import__("math").log(1.0))
    assert metrics["ece"]["bins"][-1]["count"] == 5
    cells = result["joint"]["gold_pair_cells"]
    assert all(cells[i][i]["count"] == 1 for i in range(5))
    assert all(cells[i][j]["count"] == 0 for i in range(5) for j in range(5) if i != j)
    correlated = heads.clone()
    correlated[1].zero_(); correlated[1, 0] = 1.0; correlated[1, 6] = 1.0
    correlated[2].zero_(); correlated[2, 2] = 1.0; correlated[2, 5] = 1.0
    corr_result = head_joint_metrics(correlated, _targets(), vocab)
    assert corr_result["joint"]["error_binary_covariance"] == pytest.approx(-0.04)
    assert corr_result["joint"]["error_binary_correlation"] == pytest.approx(-0.25)


def test_empty_input_has_counts_and_nullable_metrics():
    vocab = _vocab()
    result = head_joint_metrics(torch.empty((0, 10)), [], vocab)
    assert result["joint_exact"]["correct"] == 0
    assert result["joint_exact"]["count"] == 0
    assert result["joint_exact"]["accuracy"] is None
    assert result["participant"]["accuracy"] is None
    assert result["participant"]["ece"]["bins"][0]["count"] == 0


def test_unknown_class_and_invalid_probabilities_rejected():
    vocab = _vocab()
    with pytest.raises(ValueError, match="unknown participant"):
        head_joint_metrics(torch.full((1, 10), 0.1), [{"concept": {"participant": "nope", "time": "t0"}}], vocab)
    target = [_targets()[0]]
    with pytest.raises(ValueError, match="sum"):
        head_joint_metrics(torch.full((1, 10), 0.1), target, vocab)
    bad = torch.full((1, 10), 0.1)
    bad[0, 0] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        head_joint_metrics(bad, target, vocab)


def test_scalar_ece_and_separate_clamping_entropy_and_logprob():
    vocab = _vocab()
    targets = [_targets()[0], _targets()[1]]
    heads = torch.full((2, 10), 0.15, dtype=torch.float64)
    heads[0, 0] = 0.4
    heads[0, 5] = 0.4
    heads[1].zero_()
    heads[1, 1] = 1.0
    heads[1, 6] = 1.0
    result = head_joint_metrics(heads, targets, vocab)
    # Row 0 is correct with confidence .4; row 1 is correct with confidence 1.
    assert result["participant"]["ece"]["value"] == pytest.approx(0.3)
    one_row = head_joint_metrics(heads[:1], targets[:1], vocab)
    assert one_row["participant"]["brier"] == pytest.approx(0.45)

    tiny = torch.zeros((1, 10), dtype=torch.float64)
    tiny[0, 0] = 1e-15
    tiny[0, 5] = 1e-15
    tiny[0, 1:5] = 0.25
    tiny[0, 6:10] = 0.25
    tiny_result = head_joint_metrics(tiny, [_targets()[0]], vocab)
    import math
    assert tiny_result["joint"]["raw_rows"][0]["gold_joint_logprob"] == pytest.approx(2 * math.log(1e-12))
    expected_entropy = 2 * (-(1e-15 * math.log(1e-12) + 4 * 0.25 * math.log(0.25)))
    assert tiny_result["joint"]["raw_rows"][0]["joint_entropy"] == pytest.approx(expected_entropy)
