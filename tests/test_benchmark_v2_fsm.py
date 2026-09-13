from __future__ import annotations

import pytest

from norishio_lm.benchmark_v2_fsm import (
    BOS,
    BYTE_OFFSET,
    ByteTag,
    FrozenLocalPrefixFSM,
)


def _tokens(text: str) -> list[int]:
    return [BOS, *(BYTE_OFFSET + value for value in text.encode("utf-8"))]


@pytest.fixture()
def fsm() -> FrozenLocalPrefixFSM:
    return FrozenLocalPrefixFSM()


def test_compiles_both_templates_and_byte_tags(fsm: FrozenLocalPrefixFSM) -> None:
    assert len(fsm.candidates) == 2 * 6 * 6 * 4 * 4
    assert {candidate.text for candidate in fsm.candidates}.__len__() == len(fsm.candidates)

    first = next(candidate for candidate in fsm.candidates if candidate.text.startswith("今日、私は友人と会う"))
    assert first.tags[: len("今日".encode())] == (ByteTag.TIME,) * len("今日".encode())
    participant_start = len("今日、私は".encode())
    assert first.tags[participant_start:participant_start + len("友人".encode())] == (
        ByteTag.PARTICIPANT,
    ) * len("友人".encode())
    predicate_start = len("今日、私は友人と".encode())
    assert first.tags[predicate_start:predicate_start + len("会う".encode())] == (
        ByteTag.PREDICATE,
    ) * len("会う".encode())
    assert all(tag is ByteTag.LITERAL for tag in first.tags[len("今日".encode()):participant_start])

    second = next(candidate for candidate in fsm.candidates if candidate.text.startswith("私は友人と朝に"))
    assert second.tags[: len("私は".encode())] == (ByteTag.LITERAL,) * len("私は".encode())
    assert second.tags[len("私は友人と".encode()):len("私は友人と朝".encode())] == (
        ByteTag.TIME,
    ) * len("朝".encode())


def test_bos_and_literal_prefixes_are_zero(fsm: FrozenLocalPrefixFSM) -> None:
    assert fsm.gates([BOS]) == (False, False, False, False)
    assert fsm.gates(_tokens("今日、")) == (False, False, False, False)
    assert fsm.gates(_tokens("今日")) == (False, False, False, False)
    # A literal byte shared by no candidate is an unrecognised prefix.
    assert fsm.gates(_tokens("x")) == (False, False, False, False)


def test_participant_and_time_gates_follow_slot_byte_spans(fsm: FrozenLocalPrefixFSM) -> None:
    # In template 0, the complete time is followed by a literal, then the
    # participant begins.  Template 1 exercises the opposite slot order.
    assert fsm.gates(_tokens("今日、私は")) == (True, False, False, False)
    assert fsm.gates(_tokens("今日、私は友人")) == (False, False, False, False)
    assert fsm.gates(_tokens("今日、私は友人と")) == (False, False, True, True)
    assert fsm.gates(_tokens("私は")) == (True, False, False, False)
    assert fsm.gates(_tokens("私は友人と")) == (False, True, False, False)
    assert fsm.gates(_tokens("私は友人と朝")) == (False, False, False, False)
    assert fsm.gates(_tokens("私は友人と朝に")) == (False, False, True, True)

    # Morning/night are three bytes while other times are six; the gate uses
    # actual UTF-8 byte spans rather than a fixed character count.
    assert fsm.gates(_tokens("私は友人と夜")) == (False, False, False, False)
    assert fsm.gates(_tokens("私は友人と夜に")) == (False, False, True, True)


def test_predicate_ambiguity_and_completion_force_zero(fsm: FrozenLocalPrefixFSM) -> None:
    # 会う, 会いたい, 会う予定だ, and 会わない share a prefix but diverge at
    # different points.  The first byte is predicate-tagged for every retained
    # candidate, while a completed short candidate makes the exact prefix zero.
    prefix = "今日、私は友人と"
    assert fsm.gates(_tokens(prefix)) == (False, False, True, True)
    assert fsm.gates(_tokens(prefix + "会")) == (False, False, True, True)
    assert fsm.gates(_tokens(prefix + "会う")) == (False, False, False, False)
    assert fsm.gates(_tokens(prefix + "会う。")) == (False, False, False, False)


def test_malformed_special_and_unrecognised_histories_are_zero(fsm: FrozenLocalPrefixFSM) -> None:
    good = _tokens("今日、私は友人と")
    assert fsm.gates(good + [2]) == (False, False, False, False)  # EOS is not a byte.
    assert fsm.gates(good + [3]) == (False, False, False, False)  # SEP is not a byte.
    assert fsm.gates(good + [BOS]) == (False, False, False, False)
    assert fsm.gates(good + [BYTE_OFFSET + 255]) == (False, False, False, False)
    assert fsm.gates([BYTE_OFFSET]) == (False, False, False, False)


def test_batch_output_is_causal_and_has_expected_shape(fsm: FrozenLocalPrefixFSM) -> None:
    row = _tokens("今日、私は友人と会う。")
    gates = fsm.batch_gates([row])
    assert tuple(gates.shape) == (1, len(row), 4)
    assert str(gates.dtype) == "torch.bool"
    for position in range(len(row)):
        expected = fsm.gates(row[: position + 1])
        assert tuple(bool(value) for value in gates[0, position].tolist()) == expected

    # Changing a future token cannot alter a prior position's gate.
    changed = list(row)
    changed[-1] = BYTE_OFFSET + 1
    changed_gates = fsm.batch_gates([changed])
    for position in range(len(row) - 1):
        assert gates[0, position].tolist() == changed_gates[0, position].tolist()


def test_batch_adapter_can_return_python_values_without_torch_contract(fsm: FrozenLocalPrefixFSM) -> None:
    values = fsm.batch_gates([[BOS], [BOS]], as_tensor=False)
    assert values == (((False, False, False, False),), ((False, False, False, False),))
