from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.benchmark_v3_contract import INTERVENTION_RULE
from norishio_lm.benchmark_v3_future_intervention import (
    FACTOR_WIDTHS,
    alternate_intervention_one_hot,
    score_future_intermediate_probability_interventions,
)


def _frame(participant="p1"):
    return {"participant": participant, "time": "t1", "event": "meet", "operator": "want"}


def _output(text):
    return {"text": text, "ended_eos": True, "valid_utf8": True, "unique_output": True}


def _parse(text):
    return {"ok": _frame(), "changed": _frame("p2")}.get(text)


def _one_hot(width, selected):
    return [1.0 if index == selected else 0.0 for index in range(width)]


def _item(factor="participant", baseline_class=1, intervention_class=2):
    baseline = {
        name: _one_hot(width, 0)
        for name, width in FACTOR_WIDTHS.items()
    }
    baseline[factor] = _one_hot(FACTOR_WIDTHS[factor], baseline_class)
    intervened = {name: list(values) for name, values in baseline.items()}
    intervened[factor] = _one_hot(FACTOR_WIDTHS[factor], intervention_class)
    return {
        "factor": factor,
        "intervention_rule": INTERVENTION_RULE,
        "baseline_argmax_class": baseline_class,
        "intervention_class": intervention_class,
        "baseline_probabilities": baseline,
        "intervened_probabilities": intervened,
        "baseline_source_identity": "s",
        "intervened_source_identity": "s",
        "baseline_decoder_prefix": [1],
        "intervened_decoder_prefix": [1],
        "baseline": _output("ok"),
        "intervened": _output("changed"),
    }


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


def test_future_scorer_rejects_actual_vector_at_baseline_class() -> None:
    item = _item()
    same_class = {**item, "intervened_probabilities": {
        **item["intervened_probabilities"],
        "participant": _one_hot(FACTOR_WIDTHS["participant"], 1),
    }}

    with pytest.raises(ValueError, match="must differ from baseline argmax"):
        score_future_intermediate_probability_interventions([same_class], parser=_parse)


def test_future_scorer_rejects_actual_nonbaseline_class_mismatch() -> None:
    item = _item()
    wrong_class = {**item, "intervened_probabilities": {
        **item["intervened_probabilities"],
        "participant": _one_hot(FACTOR_WIDTHS["participant"], 3),
    }}

    with pytest.raises(ValueError, match="actual intervention class mismatch"):
        score_future_intermediate_probability_interventions([wrong_class], parser=_parse)


@pytest.mark.parametrize(
    ("factor", "baseline_class"),
    [("participant", 5), ("time", 2), ("event", 3), ("operator", 0)],
)
def test_future_scorer_accepts_actual_alternate_class_for_every_factor(
    factor, baseline_class,
) -> None:
    width = FACTOR_WIDTHS[factor]
    intervention_class = (baseline_class + 1) % width
    report = score_future_intermediate_probability_interventions(
        [_item(factor, baseline_class, intervention_class)], parser=_parse,
    )

    example = report["examples"][0]
    assert example["factor"] == factor
    assert example["baseline_argmax_class"] == baseline_class
    assert example["intervention_class"] == intervention_class
    assert example["one_hot"] == _one_hot(width, intervention_class)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_argmax_class", True),
        ("baseline_argmax_class", 1.0),
        ("intervention_class", True),
        ("intervention_class", "2"),
    ],
)
def test_future_scorer_rejects_non_int_class_metadata(field, value) -> None:
    item = _item()

    with pytest.raises(ValueError, match="metadata must be an int"):
        score_future_intermediate_probability_interventions(
            [{**item, field: value}], parser=_parse,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("intervened_source_identity", "other", "identical nonempty source identity"),
        ("intervened_decoder_prefix", [2], "identical decoder prefix"),
    ],
)
def test_future_scorer_retains_base_evidence_validation(field, value, message) -> None:
    item = _item()

    with pytest.raises(ValueError, match=message):
        score_future_intermediate_probability_interventions(
            [{**item, field: value}], parser=_parse,
        )


def test_future_scorer_retains_base_non_target_validation() -> None:
    item = _item()
    intervened = {
        **item["intervened_probabilities"],
        "time": _one_hot(FACTOR_WIDTHS["time"], 1),
    }

    with pytest.raises(ValueError, match="changed non-target probability vector 'time'"):
        score_future_intermediate_probability_interventions(
            [{**item, "intervened_probabilities": intervened}], parser=_parse,
        )
