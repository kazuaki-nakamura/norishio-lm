from __future__ import annotations

import pytest

pytest.importorskip("torch")

from norishio_lm.plastic_memory_experiment import run_plastic_memory_experiment


def test_experiment_reports_required_mechanism_evidence(tmp_path) -> None:
    result = run_plastic_memory_experiment(tmp_path)

    assert result["device"] == "cpu"
    assert result["core"] == {"unchanged": True, "max_abs_parameter_drift": 0.0}
    assert result["parameters"]["optimizer_target_is_adapter_only"] is True
    assert result["parameters"]["frozen_core"] > 0
    assert result["parameters"]["trainable_adapter"] > 0
    assert result["optimizer_state"]["actual_elements_before_step"] == 0
    assert result["optimizer_state"]["adam_moment_elements_with_adapter_only"] < result["optimizer_state"]["adam_moment_elements_if_all_parameters_optimized"]
    for episode in result["episodes"]:
        assert episode["disabled"]["accuracy"] == 0.5
        assert episode["enabled"]["accuracy"] == 1.0
        assert episode["disabled"]["predictions"] != episode["enabled"]["predictions"]


def test_replay_consolidation_and_detach_boundary_are_measured(tmp_path) -> None:
    result = run_plastic_memory_experiment(tmp_path)

    assert result["replay"]["identical_outputs"] is True
    assert result["replay"]["checkpoint_state_digest"] == result["replay"]["loaded_state_digest"]
    assert result["consolidation"]["source_adapters"] == 2
    assert result["consolidation"]["consolidated_mean_accuracy"] == 1.0
    assert result["consolidation"]["fast_ensemble_mean_accuracy"] == 1.0
    assert result["consolidation"]["accuracy_change_from_fast_ensemble"] == 0.0
    assert result["consolidation"]["mse_change_from_fast_ensemble"] > 0.0
    assert result["consolidation"]["retention_per_episode"] == [
        {"episode_id": "episode-a", "accuracy_change": 0.0, "mse_change": pytest.approx(0.12406250706408173)},
        {"episode_id": "episode-b", "accuracy_change": 0.0, "mse_change": pytest.approx(0.12406250706408173)},
    ]
    assert result["consolidation"]["isolated_cross_episode_mean_accuracy"] == 0.75
    assert result["gradient_boundary"]["freeze_only"]["source_gradient_elements"] > 0
    assert result["gradient_boundary"]["detached_features"]["source_gradient_elements"] == 0


def test_seed_is_explicit_and_rejects_boolean(tmp_path) -> None:
    assert run_plastic_memory_experiment(tmp_path / "valid", seed=11)["seed"] == 11
    with pytest.raises(TypeError, match="seed"):
        run_plastic_memory_experiment(tmp_path / "invalid", seed=True)
