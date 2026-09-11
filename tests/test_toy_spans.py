import json
from pathlib import Path

from norishio_lm.toy_spans import authored_slot_spans


def _seed():
    return json.loads((Path(__file__).parents[1] / "data/issue3/seed.json").read_text(encoding="utf-8"))


def _concept(event="MEET", operators=None, participant="友人", time="今日", agent="SELF", repeat=False):
    return {"event": event, "operators": operators or ["WANT"], "agent": agent,
            "participant": participant, "time": time,
            "location": "HERE" if event == "STAY" else "UNSPECIFIED",
            "repeat_marked": repeat}


def test_authored_spans_are_utf8_bytes_and_exclude_special_token_offsets():
    seed = _seed()
    condition = seed["conditions"][0]
    text = condition["target"].format(time="今日", person="友人")
    spans = authored_slot_spans({"text": text, "concept": _concept()}, seed)
    assert spans == {
        "participant": {"start": len("私は、今日".encode()), "end": len("私は、今日友人".encode())},
        "time": {"start": len("私は、".encode()), "end": len("私は、今日".encode())},
    }
    assert text.encode()[spans["participant"]["start"]:spans["participant"]["end"]].decode() == "友人"


def test_literal_occurrence_does_not_select_same_text_before_the_slot():
    seed = {
        "people": ["友人", "先輩"],
        "times": ["今日"],
        "conditions": [{"event": "MEET", "operators": ["WANT"], "agent": "SELF",
                        "repeat": False, "target": "{time}友人{person}"}],
    }
    target = {"text": "今日友人先輩", "concept": _concept(participant="先輩")}
    assert authored_slot_spans(target, seed) == {
        "participant": {"start": len("今日友人".encode()), "end": len("今日友人先輩".encode())},
        "time": {"start": 0, "end": len("今日".encode())},
    }


def test_absent_unknown_mismatched_and_ambiguous_targets_are_rejected():
    seed = _seed()
    condition = seed["conditions"][0]
    text = condition["target"].format(time="今日", person="友人")
    target = {"text": text, "concept": _concept()}
    assert authored_slot_spans({"text": text}, seed) is None
    assert authored_slot_spans({"text": text + "余計", "concept": _concept()}, seed) is None
    assert authored_slot_spans({"text": text, "concept": _concept(time="明日")}, seed) is None
    malformed = {**seed, "conditions": [dict(condition), dict(condition)]}
    assert authored_slot_spans(target, malformed) is None


def test_each_template_must_have_exactly_one_participant_and_time_slot():
    seed = _seed()
    for bad_template in ("{time}友人と会う", "{time}{person}{person}", "{time:{x}}{person}"):
        bad = {"people": ["友人"], "times": ["今日"],
               "conditions": [{"event": "MEET", "operators": ["WANT"], "agent": "SELF",
                               "repeat": False, "target": bad_template}]}
        assert authored_slot_spans({"text": "今日友人", "concept": _concept()}, bad) is None
