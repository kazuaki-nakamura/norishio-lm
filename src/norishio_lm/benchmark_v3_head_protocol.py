"""Pre-training freeze for the bounded Issue #46 head-learning comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from . import benchmark_v3_head_fixture as fixture
from .benchmark_v3_checkpoint import state_sha256, trainable_parameter_count
from .benchmark_v3_model import build_model
from .benchmark_v3_protocol import batch_schedule, schedule_sha256


ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = ROOT / "data" / "benchmark_v3_head_learning" / "experiment-descriptor-v1.json"
SCHEMA = "norishio.issue46.head-learning-protocol.v1"
FREEZE_SCHEMA = "norishio.issue46.head-learning-freeze.v1"
ARMS = ("JOINT", "FACTOR_ONLY")
SEEDS = (7, 17, 29)
RUN_ORDER = tuple((arm, seed) for seed in SEEDS for arm in ARMS)
MAX_ATTEMPTS = 6
MAX_UPDATES = 3600

# All code capable of changing training, row interpretation, or aggregation is
# content-bound at freeze time. The protocol and executor are covered by the
# pre-training Git commit; protocol self-hashing would be circular.
BOUND_CODE = (
    ".gitattributes",
    "data/benchmark_v2/spec.json",
    "data/benchmark_v3_factor_path/spec.json",
    "src/norishio_lm/benchmark_v3_head_fixture.py",
    "src/norishio_lm/benchmark_v3_head_runner.py",
    "src/norishio_lm/benchmark_v3_head_metrics.py",
    "src/norishio_lm/benchmark_v3_model.py",
    "src/norishio_lm/benchmark_v3_runner.py",
    "src/norishio_lm/benchmark_v3_protocol.py",
    "src/norishio_lm/benchmark_v3_fsm.py",
    "src/norishio_lm/benchmark_v2_model.py",
    "src/norishio_lm/benchmark_v3_head_execute.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_hash(relative: str) -> str:
    # Normalizing line endings avoids a checkout-specific Windows hash while
    # still binding the executable text that Python reads.
    return hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _paired_initial_states() -> dict[str, Any]:
    result: dict[str, Any] = {}
    expected_count: int | None = None
    for seed in SEEDS:
        left = build_model("H1L1", seed=seed)
        right = build_model("H1L1", seed=seed)
        states = (left.state_dict(), right.state_dict())
        if tuple(states[0]) != tuple(states[1]) or any(
            not torch.equal(states[0][key], states[1][key]) for key in states[0]
        ):
            raise ValueError("paired JOINT/FACTOR_ONLY initial tensors differ")
        counts = (trainable_parameter_count(left), trainable_parameter_count(right))
        if counts[0] != counts[1] or (expected_count is not None and counts[0] != expected_count):
            raise ValueError("paired H1L1 parameter count differs")
        expected_count = counts[0]
        digest = state_sha256(states[0])
        if digest != state_sha256(states[1]):
            raise ValueError("paired H1L1 initial state digests differ")
        result[str(seed)] = {"state_sha256": digest, "parameter_count": counts[0], "bitwise_equal": True}
    return result


def build_descriptor() -> dict[str, Any]:
    """Derive all fixed settings without opening prior run artifacts."""
    bundle = fixture.build()
    report = fixture.validate(bundle)
    fixture.check_expected(report)
    if len(bundle["train"]) != 384 or len(bundle["confirmation"]) != 96:
        raise ValueError("fixture split cardinality changed")
    train_ids = [row["id"] for row in bundle["train"]]
    confirm_ids = [row["id"] for row in bundle["confirmation"]]
    if len(set(train_ids + confirm_ids)) != 480:
        raise ValueError("fixture row IDs are not unique")
    paired = _paired_initial_states()
    schedules = {str(seed): schedule_sha256(batch_schedule(
        seed=seed, train_rows=384, steps=600, batch_size=16,
    )) for seed in SEEDS}
    return {
        "schema": SCHEMA,
        "issue": 46,
        "scope": "authored_structural_factor_head_transport_only",
        "historical_consumed_runs": "excluded_immutable",
        "fixture": {
            "content_digest_sha256": report["content_digest_sha256"],
            "spec_sha256": report["spec_sha256"],
            "generator_sha256": report["generator_sha256"],
            "train_row_ids_sha256": sha256(train_ids),
            "confirmation_row_ids_sha256": sha256(confirm_ids),
            "train_rows": 384,
            "confirmation_rows": 96,
            "confirmation_support_counts": report["support_classes"]["confirmation"],
            "split_plan": report["split_plan"],
        },
        "arms": {
            "ids": list(ARMS),
            "shared_module_tree": "BenchmarkV3Model(H1L1)",
            "paired_initialization": paired,
            "loss": {"JOINT": "lm_loss + factor_loss", "FACTOR_ONLY": "factor_loss"},
            "factor_loss": "sum of four source-head cross-entropies",
            "decoder_parameters_present_in_both": True,
        },
        "training": {
            "seeds": list(SEEDS),
            "run_order": [{"arm": arm, "seed": seed} for arm, seed in RUN_ORDER],
            "updates_per_run": 600,
            "batch_size": 16,
            "schedule": "seeded CPU torch.randint, with replacement, over 384 row indices",
            "schedule_sha256": schedules,
            "optimizer": {"name": "Adam", "learning_rate": 0.003,
                          "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "gradient_clip": 1.0,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
            "max_attempts": MAX_ATTEMPTS,
            "max_total_optimizer_updates": MAX_UPDATES,
            "retry": False,
            "gpu": False,
            "paid_compute": False,
            "network_data": False,
        },
        "evaluation": {
            "order": ["train_resubstitution", "confirmation"],
            "row_order": "frozen fixture list order in each split",
            "raw_row_fields": ["arm", "seed", "row_id", "split", "support_group",
                               "target_frame", "factor_probability_vectors",
                               "factor_argmax", "available"],
            "metrics": ["factor_correct_count_rate", "class_balanced_accuracy",
                        "per_class_denominators", "missing_unavailable_counts",
                        "four_head_joint_exact_secondary", "paired_seed_delta"],
            "paired_delta": "FACTOR_ONLY - JOINT",
            "confirmation_strata": ["unseen_pair", "seen_pair/unseen_triple"],
            "generation_ranking": None,
        },
        "interpretation": {
            "branch_precedence": ["incomplete", "narrow_factor_follow_up",
                                  "basic_optimization_or_capacity_unresolved",
                                  "joint_objective_interference_leading",
                                  "compositional_generalization_leading",
                                  "fixture_specific_instability_leading",
                                  "mixed_or_inconclusive"],
            "strong_train_balanced_accuracy_min": 0.8,
            "noncollapsed_confirmation_balanced_accuracy_min": 0.5,
            "noncollapsed_distinct_predicted_classes_min": 3,
            "heldout_train_minus_unseen_balanced_accuracy_min": 0.1,
            "similar_arm_abs_confirmation_balanced_accuracy_delta_max": 0.05,
            "paired_improvement": "strictly positive per factor in all 3 seeds on all confirmation rows",
            "weak_factor": "train balanced accuracy below 0.8 in any FACTOR_ONLY seed",
            "old_weakness_absent": "all participant/time train and confirmation balanced accuracies at least 0.8 in both arms and all seeds",
            "no_composite_score": True,
        },
        "code_sha256": {relative: _file_hash(relative) for relative in BOUND_CODE},
    }


def freeze_document() -> dict[str, Any]:
    descriptor = build_descriptor()
    return {"schema": FREEZE_SCHEMA, "descriptor": descriptor,
            "descriptor_sha256": sha256(descriptor)}


def load_protocol() -> dict[str, Any]:
    """Reject drift against the tracked, pre-training descriptor."""
    try:
        saved = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("head-learning freeze descriptor is absent or unreadable") from exc
    expected = freeze_document()
    if canonical_bytes(saved) != canonical_bytes(expected):
        raise ValueError("head-learning fixture, code, or protocol differs from pre-training freeze")
    return saved


def write_freeze() -> Path:
    """Create the descriptor exactly once; never overwrite it after outcomes."""
    document = freeze_document()
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FREEZE_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return FREEZE_PATH


if __name__ == "__main__":
    print(write_freeze())
