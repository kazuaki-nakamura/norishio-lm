import pytest
torch = pytest.importorskip("torch")

from norishio_lm.prefix_metrics import score_prefix_rows


def row(prefix, generated, reference, logits=None, ended=True):
    logits = torch.zeros((len(generated), 260)) if logits is None else logits
    return {"token_ids": list(range(prefix)) + generated, "generated_token_ids": generated,
            "prefix_length": prefix, "decision_logits": logits, "ended_eos": ended}


def spans(start=1, end=3, time_start=3, time_end=4):
    return {"participant": {"start": start, "end": end}, "time": {"start": time_start, "end": time_end}}


def test_prefix_end_excludes_supplied_slot_and_reports_fullslot_ineligible():
    got = score_prefix_rows([row(2, [7, 2], [5, 6, 7, 8])], [[5, 6, 7, 8]], [spans(1, 3)])
    participant = got["fields"]["participant"]
    assert participant["fullslot"]["eligible_count"] == 0
    assert participant["fullslot"]["ineligible_count"] == 1
    assert participant["suffix"]["expected_bytes"] == 1
    assert participant["suffix"]["matched_bytes"] == 1


def test_supplied_prefix_may_differ_but_correct_slot_gets_full_credit():
    reference = [4, 6, 8, 9]
    got = score_prefix_rows([row(1, [6, 8, 2], reference)], [reference], [spans(1, 3)])
    full = got["fields"]["participant"]["fullslot"]
    assert full["eligible_count"] == 1
    assert full["exact_rows"] == 1
    assert full["exact_all_row_rate"] == 1


def test_premature_eos_keeps_expected_denominator_but_lacks_logit_coverage():
    got = score_prefix_rows([row(0, [2], [7, 8])], [[7, 8]], [spans(0, 2, 0, 1)])
    byte = got["fields"]["participant"]["byte"]
    assert byte["expected_bytes"] == 2
    assert byte["evaluated_bytes"] == 1
    assert byte["coverage"] == pytest.approx(.5)
    assert byte["matched_bytes"] == 0
    assert got["aggregate"]["continuation"]["success_rows"] == 0


def test_premature_eos_before_intersected_span_does_not_index_generated_ids():
    reference = [4, 5, 6]
    got = score_prefix_rows([row(1, [], reference)], [reference], [spans(0, 3, 0, 1)])
    assert got["fields"]["participant"]["suffix"]["evaluated_bytes"] == 0


def test_partial_slot_does_not_receive_full_credit_and_rank_uses_decision_offset():
    logits = torch.zeros((2, 260))
    logits[0, 7] = 5
    logits[1, 7] = 5
    got = score_prefix_rows([row(1, [7, 2], [6, 7, 8], logits)], [[6, 7, 8]], [spans(0, 2, 2, 3)])
    slot = got["fields"]["participant"]
    assert slot["fullslot"]["eligible_count"] == 0
    assert slot["suffix"]["evaluated_bytes"] == 1
    assert got["examples"]["participant"][0]["bytes"][0]["decision_step"] == 0
    assert got["examples"]["participant"][0]["bytes"][0]["target_id"] == 7
    assert got["examples"]["participant"][0]["bytes"][0]["rank"] == 1


def test_continuation_excludes_prefix_and_requires_eos():
    reference = [4, 5, 6]
    generation = row(1, [5, 6, 2], reference)
    got = score_prefix_rows([generation], [reference], [spans(1, 3, 1, 2)])
    assert got["aggregate"]["continuation"]["success_rows"] == 1
    assert got["aggregate"]["slot_start_continuation"]["participant"]["success_rows"] == 1
    assert got["aggregate"]["slot_start_continuation"]["time"]["success_rows"] == 1


def test_validation_rejects_padded_logits_and_out_of_range_spans():
    reference = [4, 5]
    with pytest.raises(ValueError):
        score_prefix_rows([row(0, [4], reference, torch.zeros((2, 260)))], [reference], [spans(0, 2)])
    with pytest.raises(ValueError):
        score_prefix_rows([row(0, [4], reference)], [reference], [{"participant": {"start": 0, "end": 3}, "time": {"start": 0, "end": 1}}])


def test_early_eos_before_later_slot_has_zero_coverage_without_index_error():
    reference = [4, 5, 6, 7]
    got = score_prefix_rows([row(0, [2], reference)], [reference], [spans(2, 4)])
    assert got["fields"]["participant"]["byte"]["evaluated_bytes"] == 0
    assert got["fields"]["participant"]["byte"]["expected_bytes"] == 2
    assert got["fields"]["participant"]["fullslot"]["exact_rows"] == 0


def test_generated_preslot_error_does_not_cancel_exact_slot_credit():
    reference = [4, 5, 6, 7]
    got = score_prefix_rows([row(0, [259, 5, 6, 7, 2], reference)], [reference], [spans(1, 3)])
    assert got["fields"]["participant"]["fullslot"]["exact_rows"] == 1
    assert got["aggregate"]["continuation"]["success_rows"] == 0
    assert got["aggregate"]["slot_start_continuation"]["participant"]["success_rows"] == 1


def test_fully_supplied_slot_has_no_success_credit_or_byte_denominator():
    reference = [4, 5, 6, 7]
    got = score_prefix_rows([row(3, [7, 2], reference)], [reference], [spans(1, 3)])
    slot = got["fields"]["participant"]
    assert slot["fullslot"]["eligible_count"] == 0
    assert slot["fullslot"]["exact_rows"] == 0
    assert slot["byte"]["expected_bytes"] == 0
    assert slot["byte"]["mean_nll"] is None
    assert slot["byte"]["exact_rows"] == 0
