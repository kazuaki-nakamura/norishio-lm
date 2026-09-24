"""Pre-training descriptor and fail-closed gate for Issue #48."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from . import benchmark_v3_role_fixture as fixture
from .benchmark_v3_model import FACTOR_ORDER
from .benchmark_v3_protocol import batch_schedule, schedule_sha256
from .benchmark_v3_role_model import ROLE_FACTOR_CODEBOOKS, paired_models

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "benchmark_v3_role_identifiability"
FREEZE_PATH = DATA_DIR / "experiment-descriptor-v1.json"
STATIC_AUDIT_PATH = DATA_DIR / "static-audit.json"
SCHEMA = "norishio.issue48.role-identifiability-protocol.v1"
FREEZE_SCHEMA = "norishio.issue48.role-identifiability-freeze.v1"
ARMS = fixture.ARMS
SEEDS = (7, 17, 29)
RUN_ORDER = tuple((arm, seed) for seed in SEEDS for arm in ARMS)
MAX_ATTEMPTS = 6
MAX_UPDATES = 3_600
MAX_WALL_SECONDS = 7_200

# Every source capable of changing row interpretation, optimization, or raw
# aggregation must be byte-bound before the first attempted learned run.
BOUND_CODE = (
    "data/benchmark_v3_role_identifiability/.gitattributes",
    "docs/results/benchmark-v3-role-identifiability/.gitattributes",
    "src/norishio_lm/benchmark_v3_role_fixture.py",
    "src/norishio_lm/benchmark_v3_role_model.py",
    "src/norishio_lm/benchmark_v3_role_runner.py",
    "src/norishio_lm/benchmark_v3_role_metrics.py",
    "src/norishio_lm/benchmark_v3_role_protocol.py",
    "src/norishio_lm/benchmark_v3_role_execute.py",
    "src/norishio_lm/benchmark_v3_head_metrics.py",
    "src/norishio_lm/benchmark_v3_head_runner.py",
    "src/norishio_lm/benchmark_v3_runner.py",
    "src/norishio_lm/benchmark_v3_model.py",
    "src/norishio_lm/benchmark_v2_model.py",
    "src/norishio_lm/benchmark_v3_protocol.py",
    "src/norishio_lm/benchmark_v3_checkpoint.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _file_hash(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@torch.no_grad()
def _paired_initial_preflight(bundle: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Prove matched inputs/states across every logical row before training."""
    spec = fixture.spec_data()
    expected = spec["codebooks"]["ROLE_DISJOINT"]
    for factor in FACTOR_ORDER:
        actual = ROLE_FACTOR_CODEBOOKS[factor]
        if {chr(byte): digit for byte, digit in actual.items()} != {
                letter: digit for digit, letter in enumerate(expected[factor])}:
            raise ValueError("model role codebook differs from frozen fixture")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result: dict[str, Any] = {}
    for seed in SEEDS:
        left, right, report = paired_models(seed)
        if not report["bitwise_equal"] or report["trainable_parameters"] != 32_120:
            raise ValueError("paired H1L1 initialization differs")
        embedding_rows_checked = 0
        for factor in FACTOR_ORDER:
            for code_byte, digit in ROLE_FACTOR_CODEBOOKS[factor].items():
                digit_token = ord(str(digit)) + fixture.BYTE_OFFSET
                code_token = code_byte + fixture.BYTE_OFFSET
                if not torch.equal(left.encoder.embedding.weight[code_token],
                                   left.encoder.embedding.weight[digit_token]):
                    raise ValueError("role code embedding was not cloned from digit row")
                embedding_rows_checked += 1
        if embedding_rows_checked != 20:
            raise ValueError("not all twenty role value bytes were paired")
        row_count = 0
        max_latent_difference = 0.0
        for split in fixture.SPLITS:
            a_rows = bundle["ROLE_ALIASED"][split]
            b_rows = bundle["ROLE_DISJOINT"][split]
            for a, b in zip(a_rows, b_rows, strict=True):
                a_ids, b_ids = fixture.source_ids(a), fixture.source_ids(b)
                if len(a_ids) != len(b_ids):
                    raise ValueError("paired source masks have different lengths")
                a_tokens = torch.tensor([a_ids], dtype=torch.long)
                b_tokens = torch.tensor([b_ids], dtype=torch.long)
                a_embeddings = left.encoder.embedding(a_tokens)
                b_embeddings = right.encoder.embedding(b_tokens)
                if not torch.equal(a_embeddings, b_embeddings):
                    raise ValueError("paired initial source embedding sequences differ")
                a_latent = left.encode_source(a_tokens)
                b_latent = right.encode_source(b_tokens)
                difference = float((a_latent - b_latent).abs().max())
                max_latent_difference = max(max_latent_difference, difference)
                if difference != 0:
                    raise ValueError("paired initial latent outputs differ")
                row_count += 1
        if row_count != 480:
            raise ValueError("paired initialization omitted a logical row")
        result[str(seed)] = {
            "initial_state_sha256": report["initial_state_sha256"],
            "trainable_parameters": report["trainable_parameters"],
            "state_dict_bitwise_equal": True,
            "cloned_embedding_rows": embedding_rows_checked,
            "paired_rows_with_equal_embedding_sequences": row_count,
            "paired_rows_with_equal_initial_latent": row_count,
            "initial_latent_tolerance": 0.0,
            "max_initial_latent_difference": max_latent_difference,
        }
    return result


def build_descriptor() -> dict[str, Any]:
    bundle = fixture.build()
    manifest = fixture.check_expected(bundle)
    static = fixture.static_audit_report(bundle)
    if json.loads(STATIC_AUDIT_PATH.read_text(encoding="utf-8")) != static:
        raise ValueError("saved pre-training static audit differs from fixture")
    gate = fixture.preflight(bundle)
    paired = _paired_initial_preflight(bundle)
    schedule_hashes = {
        str(seed): schedule_sha256(batch_schedule(
            seed=seed, train_rows=384, steps=600, batch_size=16)) for seed in SEEDS
    }
    source_binding = {
        arm: {split: sha256([{
            "row_id": row["id"], "source_ids": fixture.source_ids(row),
            "source_mask": [True] * len(fixture.source_ids(row)),
            "count_signature": fixture.count_signature(row),
            "frequency_signature": fixture.normalized_frequency_signature(row),
        } for row in bundle[arm][split]]) for split in fixture.SPLITS}
        for arm in ARMS
    }
    return {
        "schema": SCHEMA,
        "issue": 48,
        "scope": "authored_structural_factor_source_role_transport_only",
        "historical_consumed_artifacts": "excluded_immutable",
        "fixture": {
            "manifest_sha256": fixture.manifest_digest(bundle),
            "static_audit_sha256": sha256(static),
            "spec_sha256": manifest["spec_sha256"],
            "row_files_sha256": manifest["row_files_sha256"],
            "source_binding_sha256": source_binding,
            "rows_per_arm": {"train": 384, "confirmation": 96},
            "confirmation_support_counts": {
                support: static[ARMS[0]]["confirmation_by_support"][support]["row_count"]
                for support in fixture.CONFIRMATION_SUPPORTS
            },
            "codebooks": fixture.spec_data()["codebooks"],
            "static_preflight": gate,
        },
        "arms": {
            "ids": list(ARMS),
            "module_tree": "BenchmarkV3Model(H1L1)",
            "loss": "sum of four source factor cross-entropies only",
            "paired_initialization": paired,
            "active_value_embedding_rows": {"ROLE_ALIASED": 6, "ROLE_DISJOINT": 20},
            "gradient_sharing_differs_despite_same_nominal_parameter_count": True,
        },
        "training": {
            "seeds": list(SEEDS),
            "run_order": [{"arm": arm, "seed": seed} for arm, seed in RUN_ORDER],
            "updates_per_run": 600, "batch_size": 16,
            "schedule": "seeded CPU torch.randint with replacement over 384 row indices",
            "schedule_sha256": schedule_hashes,
            "optimizer": {"name": "Adam", "learning_rate": 0.003,
                          "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "gradient_clip": 1.0,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
            "max_attempts": MAX_ATTEMPTS,
            "max_total_optimizer_updates": MAX_UPDATES,
            "max_executor_wall_seconds": MAX_WALL_SECONDS,
            "retry": False, "gpu": False, "paid_compute": False,
            "network_data": False,
        },
        "evaluation": {
            "order": ["train_resubstitution", "confirmation"],
            "raw_row_fields": ["arm", "seed", "row_id", "split", "support_group",
                               "target_frame", "factor_probability_vectors",
                               "factor_argmax", "predicted_frame", "available",
                               "source_ids_sha256", "source_mask_sha256",
                               "count_signature_sha256", "frequency_signature_sha256"],
            "per_seed_denominators": {"train": 384, "confirmation": 96,
                                      "unseen_pair": 48, "seen_pair/unseen_triple": 48},
            "three_seed_denominators": {"train": 1152, "confirmation": 288,
                                        "unseen_pair": 144, "seen_pair/unseen_triple": 144},
            "metric": "atomic and class-balanced per factor; joint exact secondary",
            "generation_ranking": None,
        },
        "interpretation": {
            "strong_train_balanced_accuracy_min": 0.80,
            "rescue_train_margin_over_aliased_ceiling": 0.05,
            "rescue_paired_train_direction": "ROLE_DISJOINT > ROLE_ALIASED at all three seeds",
            "generalization_train_minus_confirmation_stratum_min": 0.15,
            "event_operator_noncollapse_balanced_accuracy_min": 0.50,
            "event_operator_noncollapse_distinct_predicted_classes_min": 2,
            "no_composite_score": True,
        },
        "code_sha256": {relative: _file_hash(relative) for relative in BOUND_CODE},
    }


def freeze_document() -> dict[str, Any]:
    descriptor = build_descriptor()
    return {"schema": FREEZE_SCHEMA, "descriptor": descriptor,
            "descriptor_sha256": sha256(descriptor)}


def load_protocol(*, raw_only: bool = False) -> dict[str, Any]:
    saved = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if (saved.get("schema") != FREEZE_SCHEMA
            or saved.get("descriptor_sha256") != sha256(saved.get("descriptor"))):
        raise ValueError("Issue #48 descriptor or digest is malformed")
    if raw_only:
        descriptor = saved["descriptor"]
        if descriptor["code_sha256"] != {
                relative: _file_hash(relative) for relative in BOUND_CODE}:
            raise ValueError("live execution or aggregation code differs from freeze")
        bundle = fixture.build()
        if (fixture.manifest_digest(bundle) != descriptor["fixture"]["manifest_sha256"]
                or fixture.check_expected(bundle)["static_audit_sha256"] !=
                descriptor["fixture"]["static_audit_sha256"]):
            raise ValueError("live fixture differs from freeze")
        if json.loads(STATIC_AUDIT_PATH.read_text(encoding="utf-8")) != fixture.static_audit_report(bundle):
            raise ValueError("saved static audit differs from fixture")
    elif saved != freeze_document():
        raise ValueError("live Issue #48 protocol differs from committed pre-training freeze")
    return saved


__all__ = ["ARMS", "BOUND_CODE", "DATA_DIR", "FREEZE_PATH", "MAX_ATTEMPTS",
           "MAX_UPDATES", "MAX_WALL_SECONDS", "ROOT", "RUN_ORDER", "SEEDS",
           "STATIC_AUDIT_PATH", "build_descriptor", "canonical_bytes",
           "freeze_document", "load_protocol", "sha256"]
