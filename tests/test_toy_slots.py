import json
from pathlib import Path
from copy import deepcopy
import pytest

from norishio_lm.toy_slots import parse_authored_output, score_slots


def seed():
    return json.loads((Path(__file__).parents[1] / "data/issue3/seed.json").read_text(encoding="utf-8"))


def concept(event="MEET", operators=None, participant="友人", time="今日", agent="SELF", repeat=False):
    return {"event": event, "operators": operators or ["WANT"], "agent": agent,
            "participant": participant, "time": time,
            "location": "HERE" if event == "STAY" else "UNSPECIFIED", "repeat_marked": repeat}


def test_full_template_and_scope_order_are_parsed():
    data = seed()
    condition = next(c for c in data["conditions"] if c["id"] == "meet_want_not")
    text = condition["target"].format(time="今日", person="友人")
    assert parse_authored_output(text, data) == concept(operators=["WANT", "NOT"])


def test_substrings_and_malformed_outputs_are_rejected():
    data = seed()
    assert parse_authored_output("前置き私は、今日友人と会うことを望んでいる。", data) is None
    assert parse_authored_output("私は、昨日友人と会うことを望んでいる。", data) is None


def test_gold_is_only_used_after_parsing_and_failures_count_wrong():
    data = seed()
    text = data["conditions"][0]["target"].format(time="今日", person="友人")
    result = score_slots(
        [{"text": text, "ended_eos": True, "valid_utf8": True}, {"text": text, "valid_utf8": False, "ended_eos": True}],
        [{"concept": concept()}, {"concept": concept(operators=["PLAN"])}],
        data,
    )
    assert result["parseable_count"] == 1
    assert result["fields"]["event"] == {"correct": 1, "count": 2, "accuracy": 0.5, "evaluable_count": 1, "conditional_accuracy": 1.0}
    assert result["fields"]["operators"]["correct"] == 1
    assert result["evaluable_count"] == 1
    assert result["fullframe_correct"] == 1


def test_all_target_rules_parse_and_source_only_paraphrase_is_not_supported():
    data = seed()
    for condition in data["conditions"]:
        text = condition["target"].format(time="来月", person="先輩")
        expected = concept(condition["event"], condition["operators"], "先輩", "来月", condition["agent"], condition["repeat"])
        assert parse_authored_output(text, data) == expected
    assert parse_authored_output(data["conditions"][0]["source"][0].format(time="今日", person="友人"), data) is None


@pytest.mark.parametrize("change", [{"valid_utf8": False}, {"ended_eos": False}, {"invalid_special_tokens": [0]}, {"valid_utf8": None}, {"ended_eos": None}])
def test_invalid_or_unverified_output_is_not_evaluable(change):
    data = seed()
    g = {"text": data["conditions"][0]["target"].format(time="今日", person="友人"), "valid_utf8": True, "ended_eos": True}
    r = score_slots([{**g, **change}], [{"concept": concept()}], data)
    assert r["parseable_count"] == 0
    assert r["fields"]["event"]["count"] == 1
    assert r["fields"]["event"]["evaluable_count"] == 0
    assert r["fields"]["event"]["conditional_accuracy"] is None


def test_ambiguity_target_changes_and_missing_gold_denominators():
    data = seed()
    text = data["conditions"][0]["target"].format(time="今日", person="友人")
    g = {"text": text, "valid_utf8": True, "ended_eos": True}
    one = score_slots([g], [{"concept": concept()}], data)
    two = score_slots([g], [{"concept": {"event": "STAY"}}], data)
    assert one["examples"] == two["examples"]
    assert two["fields"]["event"]["correct"] == 0
    assert two["fields"]["time"]["count"] == 0
    assert two["fullframe_count"] == 0
    ambiguous = deepcopy(data)
    ambiguous["conditions"].append({**ambiguous["conditions"][0], "operators": ["NOT", "WANT"]})
    assert parse_authored_output(text, ambiguous) is None
    assert score_slots([], [], data)["parse_coverage"] is None
    with pytest.raises(ValueError):
        score_slots([g], [], data)
