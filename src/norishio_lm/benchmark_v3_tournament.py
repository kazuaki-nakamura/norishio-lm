"""Machine checks for the preregistered Issue #36 Phase 1 contract.

This module validates configuration and terminal-run authentication only.  It
does not train a model, open the final confirmation split, or make a model
comparison claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "benchmark_v3_factor_path" / "tournament.json"
SCHEMA = "norishio.issue36.tournament.v1"
BENCHMARK_FREEZE_SHA = "ae8ea2727558a57f6fa7c0e23ad28fc767dcc859"
BENCHMARK_DIGEST = "930958ca1002b9f566fb13d28e072e99fa8ba5c0493308526586dcc128302d84"
EXPECTED_CONFIG_SHA256 = "ff89911b64d8f6e652ed617f971ff0512dbda07e4c71bd585f5e0f23635fd399"
SEEDS = (7, 17, 29)
ARM_IDS = ("H0L0", "H0L1", "H1L0", "H1L1", "D_AUX", "NO_INPUT")
TOP_LEVEL = {"schema", "status", "benchmark", "input_boundary", "common_model",
             "compute", "training", "randomness", "sampling_schedule",
             "parameter_budget", "arms", "selection", "metrics", "checkpoint_gate"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_sha256(config: Mapping[str, Any]) -> str:
    validate_tournament(config)
    return hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()


def load_tournament(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_tournament(config)
    return config


def _exact(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"frozen tournament mismatch: {name}")


PARAMETER_FORMULAS = {
    "byte_embedding": "260*16",
    "source_linear": "16*32+32",
    "decoder_embedding": "260*32",
    "decoder_gru": "3*(32*32+32*32+2*32)",
    "byte_output": "32*260+260",
    "h0_projection": "32*32+32",
    "factor_hidden": "4*(32*16+16)",
    "factor_class_heads": "2*(16*6+6)+2*(16*4+4)",
    "factor_conditioning": "(6+6+4+4)*32+32",
}


def _base_parameter_count() -> int:
    return (260 * 16 + (16 * 32 + 32) + 260 * 32
            + 3 * (32 * 32 + 32 * 32 + 2 * 32) + (32 * 260 + 260)
            + (32 * 32 + 32) + 4 * (32 * 16 + 16)
            + 2 * (16 * 6 + 6) + 2 * (16 * 4 + 4))


def calculated_parameter_count() -> int:
    """Return the conditioned main-arm count from the explicit formulas."""
    return _base_parameter_count() + (6 + 6 + 4 + 4) * 32 + 32


def calculated_parameter_counts() -> dict[str, int]:
    return {arm: (_base_parameter_count() if arm == "D_AUX" else calculated_parameter_count())
            for arm in ARM_IDS}


ARM_COUNTS = calculated_parameter_counts()


def validate_tournament(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != TOP_LEVEL:
        raise ValueError("unexpected tournament top-level fields")
    _exact(config["schema"], SCHEMA, "schema")
    _exact(config["status"], "preregistered", "status")

    benchmark = config["benchmark"]
    _exact(benchmark.get("version"), "norishio-benchmark-v3.0", "benchmark version")
    _exact(benchmark.get("benchmark_freeze_sha"), BENCHMARK_FREEZE_SHA, "benchmark freeze")
    _exact(benchmark.get("content_digest_sha256"), BENCHMARK_DIGEST, "benchmark digest")
    _exact(benchmark.get("development_split"), "diagnostic-validation", "development split")
    _exact(benchmark.get("final_split"), "final-confirmation", "final split")
    _exact(benchmark.get("final_flag"), "--evaluate-final", "final flag")
    _exact(benchmark.get("maximum_final_invocations"), 1, "final invocation count")

    boundary = config["input_boundary"]
    _exact(boundary.get("allowed_fields"), ["context", "text"], "source allowlist")
    _exact(boundary.get("encoding"), "canonical_json_utf8_bytes", "source encoding")
    _exact({key: boundary.get(key) for key in
            ("vocabulary_size", "pad_id", "bos_id", "eos_id", "sep_id", "byte_offset")},
           {"vocabulary_size": 260, "pad_id": 0, "bos_id": 1,
            "eos_id": 2, "sep_id": 3, "byte_offset": 4}, "byte vocabulary")
    _exact(boundary.get("no_input_source"), {"kind": "constant_bos_sep_source_tensor", "token_ids": [1, 3]}, "NO_INPUT source tensor")
    required_forbidden = {"row_id", "group_id", "split", "template_id", "target", "gold_frame",
                          "full_source_categorical_id", "full_frame_categorical_id"}
    if set(boundary.get("forbidden_features", ())) != required_forbidden:
        raise ValueError("forbidden feature set changed")

    model = config["common_model"]
    _exact(model.get("source_encoder"),
           {"kind": "byte_embedding_masked_mean_linear", "embedding_dim": 16, "latent_dim": 32},
           "source encoder")
    _exact(model.get("decoder"),
           {"kind": "causal_byte_gru", "embedding_dim": 32, "hidden_dim": 32, "layers": 1},
           "decoder")
    _exact(model.get("h0"), {"kind": "latent_linear", "input_dim": 32, "hidden_dim": 32}, "h0")
    _exact(model.get("factor_values"), {"participant": 6, "time": 6, "event": 4, "operator": 4}, "factor vocabulary")
    _exact(model.get("factor_order"), ["participant", "time", "event", "operator"], "factor order")
    _exact(model.get("factor_projections"), {"kind": "per_factor_probability_linear_bias_free", "output_dim": 32}, "factor projections")
    _exact(model.get("factor_heads"), {"kind": "four_auxiliary_cross_entropy", "hidden_dim": 16}, "factor heads")
    _exact(model.get("conditioning"), {"probability_width": 20, "projection": "linear32", "global_gate": "all decoder steps", "prefix_local_gate": "grammar prefix"}, "conditioning")
    _exact(model.get("generation"), {"kind": "greedy_bos_self_history", "max_new_tokens": 96}, "generation")

    _exact(config["compute"], {"device": "cpu", "torch_threads": 1, "deterministic_algorithms": True}, "compute")
    training = config["training"]
    _exact({key: training.get(key) for key in ("steps", "batch_size", "optimizer", "gradient_clip", "scheduler", "loss_weights", "hyperparameter_search")},
           {"steps": 600, "batch_size": 16,
            "optimizer": {"name": "Adam", "lr": .003, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0.0},
            "gradient_clip": {"kind": "global_norm", "max_norm": 1.0}, "scheduler": None,
            "loss_weights": {"lm": 1.0, "participant": 1.0, "time": 1.0, "event": 1.0, "operator": 1.0},
            "hyperparameter_search": False}, "training budget")
    _exact(config["randomness"], {"run_seeds": list(SEEDS), "source_encoder_seed": "run_seed", "decoder_seed": "run_seed_plus_1000", "h0_init_seed": "run_seed_plus_1500", "factor_head_seed": "run_seed_plus_2000", "factor_projection_seed": "run_seed_plus_2500", "batch_schedule_seed": "run_seed", "generation": "greedy"}, "randomness")
    _exact(config["sampling_schedule"], {"kind": "torch_randint_with_replacement", "train_rows": 384, "steps": 600, "batch_size": 16, "seed": "run_seed"}, "sampling schedule")

    arms = config["arms"]
    if not isinstance(arms, list) or tuple(arm.get("id") for arm in arms) != ARM_IDS:
        raise ValueError("arm IDs/order changed")
    counts: list[int] = []
    for arm in arms:
        _exact(arm.get("trainable_parameters"), ARM_COUNTS[arm["id"]], f"{arm['id']} parameters")
        _exact(arm.get("factor_heads"), "shared_four_factor_ce", f"{arm['id']} factor heads")
        _exact(arm.get("loss_structure"), "lm_plus_four_factor_ce", f"{arm['id']} loss structure")
        if arm["id"].startswith("H"):
            if arm.get("activation") not in {"linear", "tanh"} or arm.get("gate_mode") not in {"global", "prefix-local"}:
                raise ValueError("main arm activation/gate changed")
        elif arm["id"] == "D_AUX":
            _exact(arm.get("activation"), "linear", "D_AUX activation")
            _exact(arm.get("gate_mode"), "latent-only", "D_AUX gate")
        else:
            _exact(arm.get("activation"), "linear", "NO_INPUT activation")
            _exact(arm.get("gate_mode"), "constant-source", "NO_INPUT gate")
            _exact(arm.get("source_input"), "constant_bos_sep_source_tensor", "NO_INPUT source")
        counts.append(arm["trainable_parameters"])
    budget = config["parameter_budget"]
    if max(counts) > budget.get("maximum_trainable", 0):
        raise ValueError("parameter maximum exceeded")
    deviation = (max(counts) - min(counts)) / min(counts)
    if deviation > budget.get("maximum_relative_deviation", -1) or deviation > .03:
        raise ValueError("parameter deviation exceeded")
    _exact(budget.get("count_rule"), "sum_numel_requires_grad", "parameter count rule")
    _exact(budget.get("post_result_padding_forbidden"), True, "padding policy")

    _exact(config["selection"], {"development_only": True, "result_driven_changes": False}, "selection policy")
    _exact(config["metrics"], {"single_seed_success_claim": False, "general_llm_claim": False}, "claim policy")
    gate = config["checkpoint_gate"]
    _exact(gate.get("required_terminal_runs"), 18, "terminal run count")
    _exact(gate.get("terminal_statuses"), ["complete", "failed"], "terminal statuses")
    _exact(gate.get("required_complete_hashes"), ["tournament_config_sha256", "benchmark_content_digest_sha256", "initial_state_sha256", "final_state_sha256", "schedule_sha256", "checkpoint_file_sha256"], "checkpoint hashes")
    _exact(gate.get("final_requires_all_terminal"), True, "final terminal gate")
    _exact(gate.get("failed_runs_remain_visible"), True, "failed run visibility")
    _exact(gate.get("retry_requires_identical_frozen_config"), True, "retry config gate")
    actual_digest = hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()
    if EXPECTED_CONFIG_SHA256 and actual_digest != EXPECTED_CONFIG_SHA256:
        raise ValueError("frozen tournament mismatch: canonical tournament digest")
    return {"schema": SCHEMA, "arms": len(arms), "runs": len(arms) * len(SEEDS), "min_parameters": min(counts), "max_parameters": max(counts), "maximum_relative_deviation": deviation, "config_sha256": actual_digest}


def required_run_keys(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    validate_tournament(config)
    return tuple((arm["id"], seed) for arm in config["arms"] for seed in SEEDS)


def _hex64(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def validate_terminal_runs(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    expected = set(required_run_keys(config))
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    seen: set[tuple[str, int]] = set(); complete: list[tuple[str, int]] = []; failed: list[tuple[str, int]] = []
    digest = config_sha256(config)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping): raise ValueError(f"record {index} must be a mapping")
        key = (record.get("arm"), record.get("seed"))
        if key not in expected or key in seen: raise ValueError("unknown or duplicate arm/seed record")
        seen.add(key); status = record.get("status")
        if record.get("tournament_config_sha256") != digest or record.get("benchmark_content_digest_sha256") != BENCHMARK_DIGEST:
            raise ValueError("terminal record digest mismatch")
        if record.get("trainable_parameters") != ARM_COUNTS[key[0]]: raise ValueError("terminal record parameter count mismatch")
        if status == "complete":
            for name in ("initial_state_sha256", "final_state_sha256", "schedule_sha256", "checkpoint_file_sha256"): _hex64(record.get(name), name)
            complete.append(key)
        elif status == "failed":
            for name in ("phase", "error_type", "redacted_message"):
                if not isinstance(record.get(name), str) or not record[name]: raise ValueError(f"failed record requires {name}")
            failed.append(key)
        else: raise ValueError("run status must be complete or failed")
    if seen != expected: raise ValueError("terminal run set is incomplete")
    return {"ready_for_single_final_invocation": bool(complete), "terminal": len(seen), "complete": len(complete), "failed": len(failed), "final_evaluable": [[arm, seed] for arm, seed in complete]}


__all__ = ["ARM_COUNTS", "ARM_IDS", "BENCHMARK_DIGEST", "BENCHMARK_FREEZE_SHA", "CONFIG_PATH", "EXPECTED_CONFIG_SHA256", "PARAMETER_FORMULAS", "SEEDS", "calculated_parameter_count", "calculated_parameter_counts", "canonical", "config_sha256", "load_tournament", "required_run_keys", "validate_terminal_runs", "validate_tournament"]
