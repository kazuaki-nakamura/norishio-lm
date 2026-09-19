from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from norishio_lm.plastic_memory_checkpoint import (
    load_plastic_memory_checkpoint,
    save_plastic_memory_checkpoint,
)


def _adapter() -> torch.nn.Module:
    torch.manual_seed(37)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)).float()


def _kwargs() -> dict[str, object]:
    return {
        "created_at": "2026-09-19T12:00:00Z",
        "base_fingerprint": "base-sha256:abc123",
        "adapter_id": "adapter-37",
        "source_episode_ids": ["episode-1", "episode-2"],
        "update_count": 7,
        "tags": ["validated", "cpu"],
        "importance": 0.75,
        "config": {"dimension": 4, "input_dim": 3, "output_dim": 2},
        "metadata": {"owner": "test", "schema": "plastic-memory.v1"},
    }


def test_roundtrip_preserves_tensors_and_metadata(tmp_path: Path):
    model = _adapter()
    path = tmp_path / "adapter.pt"
    expected = _kwargs()

    saved = save_plastic_memory_checkpoint(path, model, **expected)
    loaded = load_plastic_memory_checkpoint(
        path,
        expected_base_fingerprint=expected["base_fingerprint"],
        expected_adapter_id=expected["adapter_id"],
        expected_config=expected["config"],
        expected_metadata=expected["metadata"],
        expected_dimension=4,
    )

    assert path.is_file()
    for key in ("adapter_id", "base_fingerprint", "source_episode_ids", "created_at",
                "update_count", "tags", "importance", "config", "metadata", "operation_counts"):
        if key == "operation_counts":
            assert loaded[key] == {"reads": 0, "writes": expected["update_count"]}
            continue
        assert loaded[key] == expected[key]
    assert loaded["state_schema_sha256"] == saved["state_schema_sha256"]
    assert loaded["state_content_sha256"] == saved["state_content_sha256"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded["state_dict"][name])


def test_publish_is_exclusive_and_payload_has_strict_keys(tmp_path: Path):
    model = _adapter()
    path = tmp_path / "adapter.pt"
    save_plastic_memory_checkpoint(path, model, **_kwargs())
    with pytest.raises(FileExistsError):
        save_plastic_memory_checkpoint(path, model, **_kwargs())

    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["unexpected"] = True
    torch.save(raw, path)
    with pytest.raises(ValueError, match="keys"):
        load_plastic_memory_checkpoint(path)


def test_tampered_state_and_metadata_are_rejected(tmp_path: Path):
    model = _adapter()
    path = tmp_path / "adapter.pt"
    save_plastic_memory_checkpoint(path, model, **_kwargs())
    raw = torch.load(path, map_location="cpu", weights_only=True)

    altered = copy.deepcopy(raw)
    altered["state_dict"]["0.weight"].view(-1)[0] += 1
    torch.save(altered, tmp_path / "state-tampered.pt")
    with pytest.raises(ValueError, match="content digest"):
        load_plastic_memory_checkpoint(tmp_path / "state-tampered.pt")

    altered = copy.deepcopy(raw)
    altered["base_fingerprint"] = "wrong-base"
    torch.save(altered, tmp_path / "metadata-tampered.pt")
    with pytest.raises(ValueError, match="integrity digest"):
        load_plastic_memory_checkpoint(tmp_path / "metadata-tampered.pt")


@pytest.mark.parametrize(
    "mutator, pattern",
    [
        (lambda model: model.double(), "float32"),
        (lambda model: model, "nonfinite"),
    ],
)
def test_save_rejects_wrong_dtype_or_nonfinite(mutator, pattern: str, tmp_path: Path):
    model = mutator(_adapter())
    if pattern == "nonfinite":
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match=pattern):
        save_plastic_memory_checkpoint(tmp_path / f"{pattern}.pt", model, **_kwargs())


def test_load_binds_base_adapter_and_dimensions(tmp_path: Path):
    model = _adapter()
    path = tmp_path / "adapter.pt"
    save_plastic_memory_checkpoint(path, model, **_kwargs())
    with pytest.raises(ValueError, match="base fingerprint"):
        load_plastic_memory_checkpoint(path, expected_base_fingerprint="other")
    with pytest.raises(ValueError, match="adapter id"):
        load_plastic_memory_checkpoint(path, expected_adapter_id="other")
    with pytest.raises(ValueError, match="dimension"):
        load_plastic_memory_checkpoint(path, expected_dimension=8)


def test_plastic_adapter_roundtrip_restores_operation_counts(tmp_path: Path):
    from norishio_lm.plastic_memory import PlasticAdapter, PlasticAdapterConfig

    adapter = PlasticAdapter(PlasticAdapterConfig(2, 1), base_fingerprint="base")
    adapter.write(torch.tensor([1.0, 0.0]), torch.tensor([1.0]))
    adapter.read(torch.tensor([1.0, 0.0]))
    path = tmp_path / "plastic.pt"
    save_plastic_memory_checkpoint(
        path,
        adapter,
        created_at="deterministic",
        adapter_id="plastic",
        update_count=1,
    )
    restored = PlasticAdapter(PlasticAdapterConfig(2, 1), base_fingerprint="base")
    loaded = load_plastic_memory_checkpoint(path, adapter=restored)
    assert loaded["operation_counts"] == {"reads": 1, "writes": 1}
    assert restored.counts() == {"reads": 1, "writes": 1}
