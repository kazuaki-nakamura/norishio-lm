"""Deterministic CPU toy experiment for the plastic-memory prototype.

The authored vectors in this module are a mechanism test.  They are not
language-model outputs and do not measure human memory or continual learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .plastic_memory import PlasticAdapter, PlasticAdapterConfig, consolidate_snapshots
from .plastic_memory_checkpoint import (
    load_plastic_memory_checkpoint,
    save_plastic_memory_checkpoint,
)
from .plastic_memory_episodes import MemoryEpisode, build_memory_episodes, episode_digest


SEED = 37
BASE_FINGERPRINT_PREFIX = "sha256:"


@dataclass(frozen=True)
class Evaluation:
    examples: int
    accuracy: float
    mean_squared_error: float
    predictions: tuple[int, ...]


class FrozenToyCore(nn.Module):
    """A fixed identity feature map used only to expose the gradient boundary."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dimension, dimension, bias=False, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(dimension, dtype=torch.float32))
        self.requires_grad_(False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


def _state_bytes(module: nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return buffer.getvalue()


def module_fingerprint(module: nn.Module) -> str:
    return BASE_FINGERPRINT_PREFIX + hashlib.sha256(_state_bytes(module)).hexdigest()


def _tensor(values: Iterable[float]) -> torch.Tensor:
    return torch.tensor(tuple(values), dtype=torch.float32, device="cpu")


def write_episode(core: FrozenToyCore, adapter: PlasticAdapter, episode: MemoryEpisode, *, detach_boundary: bool) -> None:
    for example in episode.support:
        feature = core(_tensor(example.key))
        if detach_boundary:
            feature = feature.detach()
        adapter.write(feature, _tensor(example.value))


def evaluate_episode(
    core: FrozenToyCore,
    adapter: PlasticAdapter | None,
    episode: MemoryEpisode,
    *,
    detach_boundary: bool,
) -> Evaluation:
    predictions: list[int] = []
    squared_error = 0.0
    for example in episode.queries:
        feature = core(_tensor(example.key))
        if detach_boundary:
            feature = feature.detach()
        logits = torch.zeros(episode.value_dim, dtype=torch.float32) if adapter is None else adapter.read(feature)
        target = _tensor(example.value)
        predictions.append(int(torch.argmax(logits).item()))
        squared_error += float(torch.mean((logits.detach() - target) ** 2).item())
    gold = [int(max(range(len(item.value)), key=lambda index: item.value[index])) for item in episode.queries]
    correct = sum(prediction == target for prediction, target in zip(predictions, gold))
    return Evaluation(
        examples=len(gold),
        accuracy=correct / len(gold),
        mean_squared_error=squared_error / len(gold),
        predictions=tuple(predictions),
    )


def _evaluate_adapter_sum(core: FrozenToyCore, adapters: Iterable[PlasticAdapter], episode: MemoryEpisode) -> Evaluation:
    adapter_list = tuple(adapters)
    predictions: list[int] = []
    squared_error = 0.0
    gold: list[int] = []
    for example in episode.queries:
        feature = core(_tensor(example.key))
        logits = sum((adapter.read(feature) for adapter in adapter_list), torch.zeros(episode.value_dim, dtype=torch.float32))
        target = _tensor(example.value)
        predictions.append(int(torch.argmax(logits).item()))
        gold.append(int(torch.argmax(target).item()))
        squared_error += float(torch.mean((logits.detach() - target) ** 2).item())
    return Evaluation(
        examples=len(gold),
        accuracy=sum(prediction == target for prediction, target in zip(predictions, gold)) / len(gold),
        mean_squared_error=squared_error / len(gold),
        predictions=tuple(predictions),
    )


def _gradient_boundary_probe(core: FrozenToyCore, adapter: PlasticAdapter, episode: MemoryEpisode, *, detach_boundary: bool) -> dict[str, int]:
    source = _tensor(episode.queries[0].key).requires_grad_(True)
    feature = core(source)
    if detach_boundary:
        feature = feature.detach()
    loss = torch.mean((adapter.read(feature) - _tensor(episode.queries[0].value)) ** 2)
    loss.backward()
    source_gradient_elements = 0 if source.grad is None else int(torch.count_nonzero(source.grad).item())
    adapter_gradient_elements = sum(
        int(torch.count_nonzero(parameter.grad).item())
        for parameter in adapter.parameters()
        if parameter.grad is not None
    )
    adapter.zero_grad(set_to_none=True)
    return {
        "source_gradient_elements": source_gradient_elements,
        "adapter_gradient_elements": adapter_gradient_elements,
    }


def _evaluation_dict(value: Evaluation) -> dict[str, Any]:
    result = asdict(value)
    result["predictions"] = list(value.predictions)
    return result


def run_plastic_memory_experiment(output_dir: str | Path, *, seed: int = SEED) -> dict[str, Any]:
    """Run the complete deterministic mechanism test and return JSON-safe evidence."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    torch.manual_seed(seed)
    episodes = build_memory_episodes(seed)
    core = FrozenToyCore(episodes[0].key_dim)
    fingerprint = module_fingerprint(core)
    core_before = {name: value.detach().clone() for name, value in core.state_dict().items()}
    config = PlasticAdapterConfig(
        key_dim=episodes[0].key_dim,
        value_dim=episodes[0].value_dim,
        detach_writes=False,
    )
    adapters: list[PlasticAdapter] = []
    episode_results: list[dict[str, Any]] = []
    for episode in episodes:
        adapter = PlasticAdapter(config, base_fingerprint=fingerprint)
        disabled = evaluate_episode(core, None, episode, detach_boundary=False)
        write_episode(core, adapter, episode, detach_boundary=False)
        enabled = evaluate_episode(core, adapter, episode, detach_boundary=False)
        adapters.append(adapter)
        episode_results.append(
            {
                "episode_id": episode.episode_id,
                "disabled": _evaluation_dict(disabled),
                "enabled": _evaluation_dict(enabled),
                "adapter_digest": adapter.digest(),
                "counts": adapter.counts(),
            }
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "episode-a.pt"
    saved = save_plastic_memory_checkpoint(
        checkpoint_path,
        adapters[0],
        created_at="deterministic-toy-run",
        base_fingerprint=fingerprint,
        adapter_id="episode-a-adapter",
        source_episode_ids=(episodes[0].episode_id,),
        update_count=adapters[0].write_count,
        tags=("authored-fixture", "toy"),
        metadata={"episode_digest": episode_digest(episodes)},
    )
    reloaded = PlasticAdapter(config, base_fingerprint=fingerprint)
    loaded = load_plastic_memory_checkpoint(
        checkpoint_path,
        adapter=reloaded,
        expected_base_fingerprint=fingerprint,
        expected_adapter_id="episode-a-adapter",
        expected_source_episode_ids=(episodes[0].episode_id,),
    )
    original_replay = evaluate_episode(core, adapters[0], episodes[0], detach_boundary=False)
    loaded_replay = evaluate_episode(core, reloaded, episodes[0], detach_boundary=False)

    consolidated_snapshot = consolidate_snapshots(adapter.snapshot() for adapter in adapters)
    consolidated = PlasticAdapter.from_snapshot(consolidated_snapshot)
    consolidated_results = [evaluate_episode(core, consolidated, episode, detach_boundary=False) for episode in episodes]
    fast_ensemble_results = [_evaluate_adapter_sum(core, adapters, episode) for episode in episodes]
    isolated_cross_results = [
        evaluate_episode(core, adapter, episode, detach_boundary=False)
        for adapter in adapters
        for episode in episodes
    ]

    detach_adapter = PlasticAdapter(config, base_fingerprint=fingerprint)
    write_episode(core, detach_adapter, episodes[0], detach_boundary=True)
    freeze_probe = _gradient_boundary_probe(core, adapters[0], episodes[0], detach_boundary=False)
    detach_probe = _gradient_boundary_probe(core, detach_adapter, episodes[0], detach_boundary=True)
    trainable_parameters = sum(parameter.numel() for parameter in adapters[0].parameters() if parameter.requires_grad)
    frozen_parameters = sum(parameter.numel() for parameter in core.parameters())
    optimizer = torch.optim.Adam(adapters[0].parameters())
    optimizer_targets_adapter_only = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    } == {id(parameter) for parameter in adapters[0].parameters()}
    core_after = core.state_dict()
    core_drifts = [
        float(torch.max(torch.abs(core_after[name] - before)).item())
        for name, before in core_before.items()
    ]
    actual_optimizer_state_elements = sum(
        int(value.numel())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )

    return {
        "notice": "Authored CPU toy mechanism test; not a language model, learned lexical knowledge, human-memory model, or practical continual-learning result.",
        "seed": seed,
        "device": "cpu",
        "episode_digest": episode_digest(episodes),
        "base_fingerprint": fingerprint,
        "config": config.as_dict(),
        "parameters": {
            "frozen_core": frozen_parameters,
            "trainable_adapter": trainable_parameters,
            "plastic_state_elements": int(adapters[0].memory.numel()),
            "optimizer_target_is_adapter_only": optimizer_targets_adapter_only,
        },
        "optimizer_state": {
            "actual_elements_before_step": actual_optimizer_state_elements,
            "adam_moment_elements_if_all_parameters_optimized": 2 * (frozen_parameters + trainable_parameters),
            "adam_moment_elements_with_adapter_only": 2 * trainable_parameters,
            "note": "The run uses explicit plastic writes and does not take an optimizer step; moment counts are theoretical tensor-element counts.",
        },
        "gradient_boundary": {
            "freeze_only": freeze_probe,
            "detached_features": detach_probe,
            "note": "Gradient element counts are an observed backward-graph proxy, not measured FLOPs or wall-clock speed.",
        },
        "core": {
            "unchanged": all(torch.equal(core_after[name], before) for name, before in core_before.items()),
            "max_abs_parameter_drift": max(core_drifts, default=0.0),
        },
        "episodes": episode_results,
        "replay": {
            "identical_outputs": original_replay == loaded_replay,
            "checkpoint_state_digest": saved["state_content_sha256"],
            "loaded_state_digest": loaded["state_content_sha256"],
        },
        "consolidation": {
            "method": "deterministic arithmetic mean of two compatible adapter states",
            "source_adapters": len(adapters),
            "isolated_cross_episode_mean_accuracy": sum(item.accuracy for item in isolated_cross_results) / len(isolated_cross_results),
            "fast_ensemble_mean_accuracy": sum(item.accuracy for item in fast_ensemble_results) / len(fast_ensemble_results),
            "fast_ensemble_mean_squared_error": sum(item.mean_squared_error for item in fast_ensemble_results) / len(fast_ensemble_results),
            "consolidated_mean_accuracy": sum(item.accuracy for item in consolidated_results) / len(consolidated_results),
            "consolidated_mean_squared_error": sum(item.mean_squared_error for item in consolidated_results) / len(consolidated_results),
            "accuracy_change_from_fast_ensemble": (
                sum(item.accuracy for item in consolidated_results) - sum(item.accuracy for item in fast_ensemble_results)
            ) / len(consolidated_results),
            "mse_change_from_fast_ensemble": (
                sum(item.mean_squared_error for item in consolidated_results)
                - sum(item.mean_squared_error for item in fast_ensemble_results)
            ) / len(consolidated_results),
            "retention_per_episode": [
                {
                    "episode_id": episode.episode_id,
                    "accuracy_change": consolidated_result.accuracy - fast_result.accuracy,
                    "mse_change": consolidated_result.mean_squared_error - fast_result.mean_squared_error,
                }
                for episode, fast_result, consolidated_result in zip(episodes, fast_ensemble_results, consolidated_results)
            ],
            "fast_ensemble_per_episode": [_evaluation_dict(item) for item in fast_ensemble_results],
            "per_episode": [_evaluation_dict(item) for item in consolidated_results],
        },
    }


__all__ = [
    "Evaluation",
    "FrozenToyCore",
    "evaluate_episode",
    "module_fingerprint",
    "run_plastic_memory_experiment",
    "write_episode",
]
