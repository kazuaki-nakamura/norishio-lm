from __future__ import annotations

import pytest
import torch

from norishio_lm.plastic_memory import PlasticAdapter, PlasticAdapterConfig, consolidate_snapshots


def config(**overrides: object) -> PlasticAdapterConfig:
    values: dict[str, object] = {"key_dim": 3, "value_dim": 2, "learning_rate": 0.25, "read_scale": 2.0}
    values.update(overrides)
    return PlasticAdapterConfig(**values)


def test_zero_state_read_reset_and_deterministic_write() -> None:
    adapter = PlasticAdapter(config(), base_fingerprint="base-a")
    query = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    value = torch.tensor([2.0, -3.0], dtype=torch.float32)
    assert torch.equal(adapter.read(query), torch.zeros(2))
    adapter.write(query, value)
    expected = torch.tensor([1.0, 0.0, 0.0,], dtype=torch.float32)
    assert torch.allclose(adapter.read(query), expected[:2] * 0 + torch.tensor([1.0, -1.5]))
    assert adapter.counts() == {"reads": 2, "writes": 1}
    digest = adapter.digest()
    adapter.reset()
    assert torch.equal(adapter.read(query), torch.zeros(2))
    assert adapter.counts() == {"reads": 1, "writes": 0}
    adapter.write(query, value)
    assert adapter.digest() != digest  # the reset/read count is part of the audit state
    adapter.reset()
    clean = PlasticAdapter(config(), base_fingerprint="base-a")
    assert adapter.digest() == clean.digest()


def test_detached_and_through_time_write_gradient_boundaries() -> None:
    key = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    value = torch.tensor([2.0, 1.0], requires_grad=True)
    detached = PlasticAdapter(config(detach_writes=True))
    detached.write(key, value)
    detached.read(key).sum().backward()
    assert key.grad is not None  # the read query path remains trainable
    assert value.grad is None

    key2 = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)
    value2 = torch.tensor([2.0, 1.0], requires_grad=True)
    through_time = PlasticAdapter(config(detach_writes=False))
    through_time.write(key2, value2)
    through_time.read(key2).sum().backward()
    assert key2.grad is not None
    assert value2.grad is not None
    assert through_time.key_projection.weight.grad is not None
    assert through_time.value_projection.weight.grad is not None


def test_snapshot_and_consolidation_are_deterministic_and_audited() -> None:
    first = PlasticAdapter(config(), base_fingerprint="base-a")
    first.write(torch.tensor([1.0, 0.0, 0.0]), torch.tensor([2.0, 0.0]))
    second = PlasticAdapter(config(), base_fingerprint="base-a")
    second.write(torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 4.0]))
    a = consolidate_snapshots([first.snapshot(), second.snapshot()])
    b = consolidate_snapshots([second.snapshot(), first.snapshot()])
    assert torch.equal(a["state"], b["state"])
    restored = PlasticAdapter.from_snapshot(a)
    assert torch.allclose(restored.read(torch.tensor([1.0, 1.0, 0.0])), torch.tensor([0.5, 1.0]))
    assert a["counts"] == {"reads": 0, "writes": 2}

    with torch.no_grad():
        first.key_projection.weight[0, 0] = 0.5
    projected = PlasticAdapter.from_snapshot(first.snapshot())
    assert torch.equal(projected.key_projection.weight, first.key_projection.weight)

    with pytest.raises(ValueError, match="config"):
        consolidate_snapshots([first.snapshot(), PlasticAdapter(config(value_dim=3), base_fingerprint="base-a").snapshot()])
    with pytest.raises(ValueError, match="base fingerprint"):
        consolidate_snapshots([first.snapshot(), PlasticAdapter(config(), base_fingerprint="base-b").snapshot()])


def test_strict_config_shape_finite_and_snapshot_validation() -> None:
    with pytest.raises(ValueError):
        PlasticAdapterConfig(key_dim=0, value_dim=2)
    with pytest.raises(ValueError):
        PlasticAdapterConfig(key_dim=True, value_dim=2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PlasticAdapterConfig(key_dim=2, value_dim=2, learning_rate=float("inf"))
    adapter = PlasticAdapter(config())
    with pytest.raises(ValueError, match="shape"):
        adapter.read(torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        adapter.write(torch.tensor([float("nan"), 0.0, 0.0]), torch.zeros(2))
    snapshot = adapter.snapshot()
    bad = dict(snapshot)
    bad["state"] = torch.full((3, 2), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        PlasticAdapter.from_snapshot(bad)
    with pytest.raises(ValueError, match="at least two"):
        consolidate_snapshots([snapshot])
