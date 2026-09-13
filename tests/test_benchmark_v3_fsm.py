from __future__ import annotations

import pytest

from norishio_lm.benchmark_v3_fsm import (
    BOS,
    BYTE_OFFSET,
    ByteTag,
    FACTOR_ORDER,
    FrozenLocalPrefixFSM,
    generated_gates,
    teacher_forced_gates,
)


def _tokens(text: str) -> list[int]:
    return [BOS, *(BYTE_OFFSET + value for value in text.encode("utf-8"))]


@pytest.fixture()
def fsm() -> FrozenLocalPrefixFSM:
    return FrozenLocalPrefixFSM()


def test_compiles_both_v3_templates_and_utf8_factor_tags(fsm: FrozenLocalPrefixFSM) -> None:
    assert len(fsm.candidates) == 2 * 6 * 6 * 4 * 4
    assert len({candidate.text for candidate in fsm.candidates}) == len(fsm.candidates)

    first = next(candidate for candidate in fsm.candidates
                 if candidate.text.startswith("今日には友人と会う"))
    time_end = len("今日".encode("utf-8"))
    participant_start = len("今日には".encode("utf-8"))
    event_start = len("今日には友人と".encode("utf-8"))
    assert first.tags[:time_end] == (ByteTag.TIME,) * time_end
    assert first.tags[participant_start:participant_start + len("友人".encode("utf-8"))] == (
        ByteTag.PARTICIPANT,
    ) * len("友人".encode("utf-8"))
    assert first.tags[event_start:event_start + len("会う".encode("utf-8"))] == (
        ByteTag.EVENT,
    ) * len("会う".encode("utf-8"))
    assert all(tag is ByteTag.LITERAL for tag in first.tags[time_end:participant_start])

    second = next(candidate for candidate in fsm.candidates
                  if candidate.text.startswith("友人と会う。時は今日"))
    time_start = len("友人と会う。時は".encode("utf-8"))
    assert second.tags[:len("友人".encode("utf-8"))] == (
        ByteTag.PARTICIPANT,
    ) * len("友人".encode("utf-8"))
    assert second.tags[time_start:time_start + len("今日".encode("utf-8"))] == (
        ByteTag.TIME,
    ) * len("今日".encode("utf-8"))
    assert FACTOR_ORDER == ("participant", "time", "event", "operator")


def test_bos_literals_and_ambiguous_prefixes_are_zero(fsm: FrozenLocalPrefixFSM) -> None:
    assert fsm.gates([BOS]) == (False, False, False, False)
    assert fsm.gates(_tokens("今日には")) == (True, False, False, False)
    assert fsm.gates(_tokens("今日")) == (False, False, False, False)
    assert fsm.gates(_tokens("x")) == (False, False, False, False)

    # All predicate bytes license event and operator, but a short predicate
    # and longer predicates diverge immediately after their shared prefix.
    prefix = "今日には友人と"
    assert fsm.gates(_tokens(prefix)) == (False, False, True, True)
    assert fsm.gates(_tokens(prefix + "会")) == (False, False, True, True)
    assert fsm.gates(_tokens(prefix + "会う")) == (False, False, False, False)
    assert fsm.gates(_tokens(prefix + "会う。")) == (False, False, False, False)


def test_v3_slot_order_and_utf8_byte_lengths(fsm: FrozenLocalPrefixFSM) -> None:
    assert fsm.gates(_tokens("今日には友人")) == (False, False, False, False)
    assert fsm.gates(_tokens("今日には友人と")) == (False, False, True, True)

    assert fsm.gates(_tokens("友人")) == (False, False, False, False)
    assert fsm.gates(_tokens("友人と")) == (False, False, True, True)
    assert fsm.gates(_tokens("友人と会う。時は")) == (False, True, False, False)
    assert fsm.gates(_tokens("友人と会う。時は夜")) == (False, False, False, False)


def test_malformed_histories_are_rejected_as_zero(fsm: FrozenLocalPrefixFSM) -> None:
    good = _tokens("今日には友人と")
    assert fsm.gates([]) == (False, False, False, False)
    assert fsm.gates([BYTE_OFFSET]) == (False, False, False, False)
    assert fsm.gates(good + [2]) == (False, False, False, False)
    assert fsm.gates(good + [3]) == (False, False, False, False)
    assert fsm.gates(good + [BOS]) == (False, False, False, False)
    assert fsm.gates(good + [BYTE_OFFSET + 255]) == (False, False, False, False)
    assert fsm.gates(good + [True]) == (False, False, False, False)
    assert fsm.compatible_candidates(good + [2]) == ()


def test_batch_gates_are_causal_and_support_teacher_forced_or_generated_histories(
    fsm: FrozenLocalPrefixFSM,
) -> None:
    row = _tokens("今日には友人と会う。")
    gates = fsm.batch_gates([row])
    assert tuple(gates.shape) == (1, len(row), 4)
    assert str(gates.dtype) == "torch.bool"
    for position in range(len(row)):
        assert tuple(bool(value) for value in gates[0, position].tolist()) == fsm.gates(
            row[:position + 1]
        )

    changed = list(row)
    changed[-1] = BYTE_OFFSET + 1
    changed_gates = fsm.batch_gates([changed])
    for position in range(len(row) - 1):
        assert gates[0, position].tolist() == changed_gates[0, position].tolist()

    # The public helpers intentionally accept both reference (teacher-forced)
    # and self-generated histories without any row or target object.
    history = row[: len(_tokens("今日には友人と"))]
    assert teacher_forced_gates(history) == generated_gates(history) == fsm.gates(history)


def test_python_batch_contract_and_compiled_lookup(fsm: FrozenLocalPrefixFSM) -> None:
    assert fsm.batch_gates([[BOS], [BOS]], as_tensor=False) == (
        ((False, False, False, False),),
        ((False, False, False, False),),
    )
    for candidate in fsm.candidates[::37]:
        for length in range(len(candidate.byte_values) + 2):
            history = [BOS, *(BYTE_OFFSET + value
                              for value in candidate.byte_values[:length])]
            retained = [item for item in fsm.candidates
                        if item.byte_values[:length] == candidate.byte_values[:length]]
            if not retained or any(length >= len(item.tags) for item in retained):
                expected = (False, False, False, False)
            else:
                tags = {item.tags[length] for item in retained}
                expected = tuple(tags == {wanted} for wanted in (
                    ByteTag.PARTICIPANT, ByteTag.TIME,
                    ByteTag.EVENT, ByteTag.OPERATOR,
                ))
            assert fsm.gates(history) == expected
