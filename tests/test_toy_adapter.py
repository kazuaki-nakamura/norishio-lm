from __future__ import annotations

import pytest

from norishio_lm.tensorizer import SemanticTensorizer
from norishio_lm.toy_adapter import (
    MISLEADING_GLYPH_FIXTURE,
    ToySourceAdapter,
    canonical_source,
    misleading_glyph_fixture_record,
    source_record,
)


def test_adapter_rejects_rows_and_target_metadata() -> None:
    adapter = ToySourceAdapter()
    with pytest.raises(ValueError):
        adapter({"inputs": {"context": "", "text": "また会いたい"}})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        adapter({"context": "", "text": "x", "targets": {}})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        adapter({"context": "", "text": "x", "id": "secret"})  # type: ignore[arg-type]


def test_same_source_is_deterministic_and_only_observable_channels_are_filled() -> None:
    source = {"context": "", "text": "また会いたい"}
    first = source_record(source)
    second = source_record(dict(source))
    assert first == second
    assert first.surface == canonical_source(source)
    assert first.tokens == tuple(first.surface)
    assert first.characters == tuple(first.surface)
    assert first.morphemes == ()
    assert first.subcharacters == {}
    assert first.etymology_notes == {}
    assert first.senses == ()
    assert first.relations == ()


def test_context_changes_source_features() -> None:
    no_context = source_record({"context": "", "text": "また会いたい"})
    context = source_record({"context": "明日は雨", "text": "また会いたい"})
    assert no_context.surface != context.surface
    assert no_context.tokens != context.tokens


def test_misleading_fixture_is_explicit_and_has_no_modern_labels() -> None:
    record = misleading_glyph_fixture_record({"context": "", "text": "性質"})
    normal = source_record({"context": "", "text": "性質"})
    assert record.characters == normal.characters
    assert record.subcharacters == MISLEADING_GLYPH_FIXTURE
    assert record.senses == ()
    assert record.morphemes == ()
    assert record.etymology_notes == {}


def test_tensorizer_keeps_heldout_source_tokens_unknown() -> None:
    pytest.importorskip("torch")
    fit = [source_record({"context": "", "text": "既知"})]
    tensorizer = SemanticTensorizer.fit(fit)
    heldout = source_record({"context": "", "text": "未知"})
    encoded = tensorizer.encode([heldout])
    surface = encoded.channels["surface"]
    tokens = encoded.channels["tokens"]
    assert surface.ids[0, 0].item() == 1
    assert any(value == 1 for value in tokens.ids[0].tolist())
