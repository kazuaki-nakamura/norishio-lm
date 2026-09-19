"""Machine-readable future benchmark-v3 metric and intervention contracts."""
from __future__ import annotations

from typing import Any


SELECTION_CONTRACT: dict[str, Any] = {
    "schema": "norishio.issue39.selection-contract.v1",
    "aggregation": "unweighted_three_seed_mean",
    "direction": "maximize",
    "primary": "all.generation_frame_exact.accuracy",
    "tie_breakers": [
        "all.triple_exact.accuracy",
        "all.pair_exact.accuracy",
        "all.atomic_balanced_accuracy.mean",
        "all.exact_target_text.accuracy",
        "arm_id_ascending",
    ],
    "derived_metrics": {
        "all.atomic_balanced_accuracy.mean": {
            "source_paths": [
                "all.atomic_balanced_accuracy.participant",
                "all.atomic_balanced_accuracy.time",
                "all.atomic_balanced_accuracy.event",
                "all.atomic_balanced_accuracy.operator",
            ],
            "reducer": "unweighted_arithmetic_mean",
        },
    },
}

INTERVENTION_RULE = "alternate_after_baseline_argmax_mod_width"


__all__ = ["INTERVENTION_RULE", "SELECTION_CONTRACT"]
