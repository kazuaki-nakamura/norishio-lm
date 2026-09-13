from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from norishio_lm.benchmark_v2_protocol import (
    batch_schedule,
    collect_terminal_records,
    failed_record,
    schedule_sha256,
    write_terminal_record,
)
from norishio_lm.benchmark_v2_checkpoint import expected_schedule_sha256
from norishio_lm.benchmark_v2_tournament import ARM_COUNTS, SEEDS


def test_schedule_is_fixed_valid_and_seed_specific():
    first = batch_schedule(seed=7)
    assert first == batch_schedule(seed=7)
    assert first != batch_schedule(seed=17)
    assert len(first) == 600 and all(len(batch) == 16 for batch in first)
    assert all(0 <= index < 384 for batch in first for index in batch)
    assert len(schedule_sha256(first)) == 64
    assert schedule_sha256(first) == expected_schedule_sha256(7)
    with pytest.raises(ValueError):
        batch_schedule(seed=8)
    with pytest.raises(ValueError):
        schedule_sha256(first[:-1])


def test_all_failed_records_are_terminal_and_immutable(tmp_path):
    for arm in ARM_COUNTS:
        for seed in SEEDS:
            record = failed_record(
                arm=arm, seed=seed, phase="training", error_type="ExampleError",
                redacted_message="fixture failure",
            )
            path = write_terminal_record(tmp_path, record)
            assert json.loads(path.read_text(encoding="utf-8")) == record
    records = collect_terminal_records(tmp_path)
    assert len(records) == 18 and all(record["status"] == "failed" for record in records)
    with pytest.raises(FileExistsError):
        write_terminal_record(tmp_path, records[0])


def test_failure_messages_must_be_short_redacted_lines():
    with pytest.raises(ValueError):
        failed_record(
            arm="A_G0", seed=7, phase="training", error_type="Error",
            redacted_message="line one\nD:/private/path",
        )


def test_terminal_writer_rejects_extra_fields_and_bypassed_redaction(tmp_path):
    record = failed_record(
        arm="A_G0", seed=7, phase="training", error_type="Error",
        redacted_message="safe summary",
    )
    with pytest.raises(ValueError, match="unexpected"):
        write_terminal_record(tmp_path, {**record, "private_path": "D:/secret"})
    with pytest.raises(ValueError, match="short line"):
        write_terminal_record(tmp_path, {**record, "redacted_message": "unsafe\npath"})
