from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_contract import INTERVENTION_RULE
from norishio_lm.benchmark_v3_future_intervention import (
    alternate_intervention_one_hot,
    score_future_intermediate_probability_interventions,
)


def _frame(participant="p1"):
    return {"participant": participant, "time": "t1", "event": "meet", "operator": "want"}


def _output(text):
    return {"text": text, "ended_eos": True, "valid_utf8": True, "unique_output": True}


def _parse(text):
    return {"ok": _frame(), "changed": _frame("p2")}.get(text)


def test_alternate_helper_selects_a_different_class_and_wraps() -> None:
    probabilities = {
        "participant": torch.tensor([[0.1, 0.2, 0.7, 0.0, 0.0, 0.0]]),
        "time": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "event": torch.tensor([[0.2, 0.8, 0.0, 0.0]]),
        "operator": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    }

    one_hot, baseline, selected = alternate_intervention_one_hot(probabilities, "event")
    assert (baseline, selected) == (1, 2)
    assert torch.equal(one_hot, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))
    wrapped, baseline, selected = alternate_intervention_one_hot(probabilities, "time")
    assert (baseline, selected) == (5, 0)
    assert torch.equal(wrapped, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))


def test_future_scorer_validates_and_retains_alternate_metadata() -> None:
    baseline = {
        "participant": [0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
        "time": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "event": [1.0, 0.0, 0.0, 0.0],
        "operator": [1.0, 0.0, 0.0, 0.0],
    }
    intervened = {**baseline, "participant": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
    item = {
        "factor": "participant",
        "intervention_rule": INTERVENTION_RULE,
        "baseline_argmax_class": 1,
        "intervention_class": 2,
        "baseline_probabilities": baseline,
        "intervened_probabilities": intervened,
        "baseline_source_identity": "s",
        "intervened_source_identity": "s",
        "baseline_decoder_prefix": [1],
        "intervened_decoder_prefix": [1],
        "baseline": _output("ok"),
        "intervened": _output("changed"),
    }

    report = score_future_intermediate_probability_interventions([item], parser=_parse)

    assert report["intervention_rule"] == INTERVENTION_RULE
    assert report["examples"][0]["baseline_argmax_class"] == 1
    assert report["examples"][0]["intervention_class"] == 2
    with pytest.raises(ValueError, match="alternate intervention class"):
        score_future_intermediate_probability_interventions(
            [{**item, "intervention_class": 0}], parser=_parse,
        )
