"""Machine checks for the preregistered Issue #34 tournament; no training here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "benchmark_v2" / "tournament.json"
SCHEMA = "norishio.issue34.tournament.v1"
BENCHMARK_FREEZE_SHA = "19ce7145495a47b40a178255272050caa615dc79"
BENCHMARK_DIGEST = "012f57dbefb00f7661ed83aa49047d5736ab03c67d94b043292f40bb780b0134"
EXPECTED_CONFIG_SHA256 = "a81458f89613215d412bea41ea48d5338a6a9c6c14e681400ab1272a4bcdd174"
ARM_COUNTS = {"A_G0": 29272, "A_G1": 29272, "B": 29848,
              "C": 29816, "D": 29532, "E": 29272}
SEEDS = (7, 17, 29)
TOP_LEVEL = {"schema", "status", "benchmark", "input_boundary", "common_model",
             "compute", "training", "randomness", "parameter_budget", "arms",
             "local_injection_fsm", "selection", "metrics", "checkpoint_gate"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


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


def calculated_parameter_counts() -> dict[str, int]:
    source = 260 * 16 + (16 * 32 + 32)
    decoder = 260 * 32 + (3 * 32 * 32 + 3 * 32 * 32 + 6 * 32) + (32 * 260 + 260)
    factor_heads = 32 * 20 + 20
    global_projection = 20 * 32 + 32
    independent_representations = 4 * (32 * 8 + 8)
    independent_heads = 2 * (8 * 6 + 6) + 2 * (8 * 4 + 4)
    symbolic_embeddings = 20 * 8
    symbolic_composer = 32 * 32 + 32
    latent_adapter = (32 * 24 + 24) + (24 * 32 + 32)
    base = source + decoder
    return {"A_G0": base + factor_heads + global_projection,
            "A_G1": base + factor_heads + global_projection,
            "B": base + independent_representations + independent_heads + global_projection,
            "C": base + factor_heads + symbolic_embeddings + symbolic_composer,
            "D": base + latent_adapter,
            "E": base + factor_heads + global_projection}


def validate_tournament(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != TOP_LEVEL:
        raise ValueError("unexpected tournament top-level fields")
    _exact(config["schema"], SCHEMA, "schema")
    _exact(config["status"], "preregistered", "status")
    benchmark = config["benchmark"]
    _exact(benchmark.get("benchmark_freeze_sha"), BENCHMARK_FREEZE_SHA, "benchmark freeze")
    _exact(benchmark.get("content_digest_sha256"), BENCHMARK_DIGEST, "benchmark digest")
    _exact(benchmark.get("development_split"), "diagnostic-validation", "development split")
    _exact(benchmark.get("final_split"), "final-holdout", "final split")
    _exact(benchmark.get("final_flag"), "--evaluate-final", "final flag")
    _exact(benchmark.get("maximum_final_invocations"), 1, "final invocation count")
    boundary = config["input_boundary"]
    _exact(boundary.get("allowed_fields"), ["context", "text"], "source allowlist")
    _exact(boundary.get("encoding"), "canonical_json_utf8_bytes", "source encoding")
    forbidden = set(boundary.get("forbidden_features", ()))
    required_forbidden = {"row_id", "group_id", "split", "template_id", "target",
                          "gold_frame", "full_source_categorical_id", "full_frame_categorical_id"}
    if forbidden != required_forbidden:
        raise ValueError("forbidden feature set changed")
    model = config["common_model"]
    _exact(model.get("source_encoder"),
           {"kind": "byte_embedding_masked_mean_linear", "embedding_dim": 16, "latent_dim": 32},
           "source encoder")
    _exact(model.get("decoder"),
           {"kind": "causal_byte_gru", "embedding_dim": 32, "hidden_dim": 32, "layers": 1},
           "decoder")
    _exact(model.get("factor_values"), {"participant": 6, "time": 6, "event": 4, "operator": 4},
           "factor vocabulary")
    _exact(model.get("factor_order"), ["participant", "time", "event", "operator"],
           "factor order")
    _exact(model.get("generation"), {"kind": "greedy_bos_self_history", "max_new_tokens": 96},
           "generation")
    _exact(config["compute"], {"device": "cpu", "torch_threads": 1,
                                      "deterministic_algorithms": True}, "compute")
    training = config["training"]
    _exact({key: training.get(key) for key in ("steps", "batch_size", "optimizer",
                                                "gradient_clip", "scheduler",
                                                "loss_weights", "hyperparameter_search")},
           {"steps": 600, "batch_size": 16,
            "optimizer": {"name": "Adam", "lr": .003, "betas": [.9, .999],
                          "eps": 1e-8, "weight_decay": 0.0},
            "gradient_clip": {"kind": "global_norm", "max_norm": 1.0},
            "scheduler": None,
            "loss_weights": {"lm": 1.0, "participant": 1.0, "time": 1.0,
                             "event": 1.0, "operator": 1.0},
            "hyperparameter_search": False}, "training budget")
    _exact(config["randomness"].get("run_seeds"), list(SEEDS), "run seeds")
    _exact(config["randomness"], {"run_seeds": list(SEEDS),
           "source_encoder_seed": "run_seed", "decoder_seed": "run_seed_plus_1000",
           "factor_path_seed": "run_seed_plus_2000", "batch_schedule_seed": "run_seed",
           "generation": "greedy"}, "randomness")
    _exact(calculated_parameter_counts(), ARM_COUNTS, "calculated parameter counts")
    arms = config["arms"]
    if not isinstance(arms, list) or [arm.get("id") for arm in arms] != list(ARM_COUNTS):
        raise ValueError("arm IDs/order changed")
    if {arm.get("family") for arm in arms} != {"A", "B", "C", "D", "E"}:
        raise ValueError("architecture family set changed")
    counts = []
    for arm in arms:
        expected_count = ARM_COUNTS[arm["id"]]
        _exact(arm.get("trainable_parameters"), expected_count, f"{arm['id']} parameters")
        counts.append(expected_count)
    budget = config["parameter_budget"]
    if max(counts) > budget.get("maximum_trainable", 0):
        raise ValueError("parameter maximum exceeded")
    deviation = (max(counts) - min(counts)) / min(counts)
    if deviation > budget.get("maximum_relative_deviation", -1) or deviation > .03:
        raise ValueError("parameter deviation exceeded")
    fsm = config["local_injection_fsm"]
    if fsm.get("gold_target_span_access") is not False or fsm.get("template_id_access") is not False:
        raise ValueError("local injection leaks gold or template IDs")
    for name in ("state_source", "compiler", "transition", "activation"):
        if not isinstance(fsm.get(name), str) or not fsm[name]:
            raise ValueError(f"local injection FSM requires {name}")
    _exact(config["selection"].get("development_only"), True, "development-only selection")
    _exact(config["selection"].get("result_driven_changes"), False, "result-driven changes")
    _exact(config["metrics"].get("single_seed_success_claim"), False, "single-seed claims")
    _exact(config["metrics"].get("general_llm_claim"), False, "general claims")
    _exact(config["checkpoint_gate"].get("required_terminal_runs"), 18, "terminal run count")
    _exact(config["checkpoint_gate"].get("terminal_statuses"), ["complete", "failed"],
           "terminal statuses")
    actual_digest = hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()
    _exact(actual_digest, EXPECTED_CONFIG_SHA256, "canonical tournament digest")
    return {"schema": SCHEMA, "arms": len(arms), "runs": len(arms) * len(SEEDS),
            "min_parameters": min(counts), "max_parameters": max(counts),
            "maximum_relative_deviation": deviation}


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
    seen: set[tuple[str, int]] = set()
    complete: list[tuple[str, int]] = []
    failed: list[tuple[str, int]] = []
    digest = config_sha256(config)
    benchmark_digest = config["benchmark"]["content_digest_sha256"]
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a mapping")
        key = (record.get("arm"), record.get("seed"))
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate arm/seed record")
        seen.add(key)
        status = record.get("status")
        if status == "complete":
            if record.get("tournament_config_sha256") != digest:
                raise ValueError("checkpoint tournament digest mismatch")
            if record.get("benchmark_content_digest_sha256") != benchmark_digest:
                raise ValueError("checkpoint benchmark digest mismatch")
            if record.get("trainable_parameters") != ARM_COUNTS[key[0]]:
                raise ValueError("checkpoint parameter count mismatch")
            for name in ("initial_state_sha256", "final_state_sha256", "schedule_sha256",
                         "checkpoint_file_sha256"):
                _hex64(record.get(name), name)
            complete.append(key)
        elif status == "failed":
            if record.get("tournament_config_sha256") != digest:
                raise ValueError("failed record tournament digest mismatch")
            if record.get("benchmark_content_digest_sha256") != benchmark_digest:
                raise ValueError("failed record benchmark digest mismatch")
            if record.get("trainable_parameters") != ARM_COUNTS[key[0]]:
                raise ValueError("failed record parameter count mismatch")
            for name in ("phase", "error_type", "redacted_message"):
                if not isinstance(record.get(name), str) or not record[name]:
                    raise ValueError(f"failed record requires {name}")
            failed.append(key)
        else:
            raise ValueError("run status must be complete or failed")
    if seen != expected:
        raise ValueError("terminal run set is incomplete")
    return {"ready_for_single_final_invocation": True, "terminal": len(seen),
            "complete": len(complete), "failed": len(failed),
            "final_evaluable": [[arm, seed] for arm, seed in complete]}


__all__ = ["ARM_COUNTS", "BENCHMARK_DIGEST", "BENCHMARK_FREEZE_SHA", "CONFIG_PATH",
           "EXPECTED_CONFIG_SHA256",
           "calculated_parameter_counts",
           "SEEDS", "config_sha256", "load_tournament", "required_run_keys",
           "validate_terminal_runs", "validate_tournament"]
